from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

from . import __version__
from .backends import ARM_SME_BACKEND, backend_for_architecture
from .ir import Node, Op
from .model import rank
from .patterns import Candidate, enumerate_candidates, lanes_for
from .profile import MachineProfile
from .scheduler import ScheduledNode, schedule


RADICES = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64)
PRECISIONS = ("fp32", "fp64")
DIRECTIONS = ("forward", "backward")
STAGES = ("first", "other")


def _compiler_digest() -> str:
    digest = sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def profile_digest(profile: MachineProfile) -> str:
    canonical = json.dumps(profile.to_json(), sort_keys=True,
                           separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()


def _literal(value: float, precision: str) -> str:
    if abs(value) < 5e-16:
        value = 0.0
    text = f"{value:.17g}"
    if "." not in text and "e" not in text:
        text += ".0"
    return text + ("f" if precision == "fp32" else "")


def _kernel_name(radix: int, precision: str, direction: str,
                 stage: str) -> str:
    suffix = "fwd" if direction == "forward" else "bwd"
    return f"mofft_kernel_r{radix}_{precision}_{suffix}_{stage}"


def _coefficient_data(candidate: Candidate) -> tuple[list[float], list[float]]:
    sign = -1.0 if candidate.direction == "forward" else 1.0
    real: list[float] = []
    imag: list[float] = []
    for j in range(candidate.coefficient_rows):
        for k in range(candidate.coefficient_columns):
            angle = sign * 2.0 * math.pi * j * k / candidate.radix
            real.append(math.cos(angle))
            imag.append(math.sin(angle))
    return real, imag


_backend = ARM_SME_BACKEND
_Types = _backend.types
_emit_node = _backend.emit_node


@dataclass(frozen=True)
class _Bundle:
    kind: str
    width: int
    members: tuple[int, ...]


def _row_count(candidate: Candidate, group: int, block: int) -> int:
    global_block = group * candidate.blocks_per_group + block
    row_base = global_block * candidate.output_block_width
    return min(candidate.output_block_width,
               max(0, candidate.output_rows - row_base))


def _partition_vg(nodes: list[Node], contiguous,
                  widths: tuple[int, ...] = (4, 2)) -> list[tuple[Node, ...]]:
    """Prefer VG4, then VG2, leaving unsafe tails as scalar operations."""
    result: list[tuple[Node, ...]] = []
    index = 0
    while index < len(nodes):
        width = 1
        for requested in widths:
            chunk = nodes[index:index + requested]
            if len(chunk) == requested and all(
                    contiguous(chunk[i - 1], chunk[i])
                    for i in range(1, requested)):
                width = requested
                break
        if width > 1:
            result.append(tuple(nodes[index:index + width]))
        index += width
    return result


def _build_bundles(candidate: Candidate, scheduled: list[ScheduledNode],
                   group: int) -> tuple[dict[int, _Bundle], set[int]]:
    """Prove and form the VG operations that will actually be emitted."""
    position = {item.node.id: index for index, item in enumerate(scheduled)}
    bundles: dict[int, _Bundle] = {}
    suppressed: set[int] = set()

    # Multi-vector loads advance each destination by the architectural VL,
    # not by the number of active lanes. Therefore every member must be a
    # full vector and its coefficient-table block must be exactly adjacent.
    load_groups: dict[tuple[str, str, str], list[Node]] = {}
    for item in scheduled:
        node = item.node
        if (node.op == Op.LOAD and node.attr("role") == "coefficient" and
                node.attr("layout") == "full"):
            block = int(node.attr("block", "0"))
            if (_row_count(candidate, group, block) ==
                    candidate.output_block_width == lanes_for(
                        candidate.precision, candidate.target)):
                key = (node.attr("component", ""), node.attr("index", ""),
                       node.attr("layout", ""))
                load_groups.setdefault(key, []).append(node)
    for nodes in load_groups.values():
        nodes.sort(key=lambda node: int(node.attr("block", "0")))
        chunks = _partition_vg(
            nodes, lambda a, b: int(b.attr("block", "0")) ==
            int(a.attr("block", "0")) + 1, widths=(2,))
        for chunk in chunks:
            leader = min(chunk, key=lambda node: position[node.id])
            ordered = tuple(node.id for node in chunk)
            bundles[leader.id] = _Bundle("load", len(chunk), ordered)
            suppressed.update(node_id for node_id in ordered
                              if node_id != leader.id)

    # A VG ZA read groups sub-vectors within one ZA tile; it does not combine
    # independently accumulated FMOPA tiles. Such reads are represented
    # explicitly by matrix_twiddle EXTRACT_H nodes in the IR and are never
    # inferred here from consecutive tile numbers.
    return bundles, suppressed


def _scheduled_body(candidate: Candidate, scheduled: list[ScheduledNode],
                    name: str, group: int,
                    omit_row_epilogue: bool = False,
                    twiddle_broadcast: bool = False,
                    suffix: str = "",
                    column_predicate: str = "pg_cols",
                    base_name: str = "base",
                    suppressed_nodes: frozenset[int] = frozenset()) -> str:
    bundles, bundle_suppressed = _build_bundles(candidate, scheduled, group)
    suppressed = bundle_suppressed | set(suppressed_nodes)
    row_nodes = {item.node.id for item in scheduled
                 if ((item.node.op in (Op.MATRIX_EXTRACT_H, Op.MATRIX_EXTRACT_V) and
                      item.node.attr("phase") != "matrix_twiddle") or
                     item.node.op == Op.STORE2)}
    changed = True
    while changed:
        changed = False
        for item in scheduled:
            if item.node.id not in row_nodes and any(
                    dependency in row_nodes for dependency in item.node.inputs):
                row_nodes.add(item.node.id)
                changed = True
    # vg4 ZA-extract: a direct/like-terms _finish block is exactly two EXTRACT_H
    # from consecutive even tiles (real, imag) plus one STORE2 with no
    # even-odd/like-terms recombination in the row epilogue. For those blocks,
    # replace the single-vector per-row extract loop with a vg4 loop that reads
    # four rows per tile per instruction. Skipped only when the row epilogue is
    # itself omitted (transposed output uses the vertical epilogue instead).
    vg4_skip: set[int] = set()
    finish_blocks: dict[int, tuple[int, int, int]] = {}
    if not omit_row_epilogue:
        for block in range(candidate.blocks_per_group):
            if _row_count(candidate, group, block) == 0:
                continue
            extracts: list[tuple[int, int]] = []
            stores: list[int] = []
            other = False
            for item in scheduled:
                if item.node.id not in row_nodes:
                    continue
                block_attr = item.node.attr("block")
                if block_attr is None or int(block_attr) != block:
                    continue
                node = item.node
                if node.op == Op.MATRIX_EXTRACT_H and node.attr("phase") != "matrix_twiddle":
                    extracts.append((node.id, int(node.attr("tile"))))
                elif node.op == Op.STORE2:
                    stores.append(node.id)
                else:
                    other = True
            if other or len(extracts) != 2 or len(stores) != 1:
                continue
            tiles = sorted(tile for _, tile in extracts)
            if tiles[1] != tiles[0] + 1 or tiles[0] % 2 != 0:
                continue
            store_node = candidate.graph.nodes[stores[0]]
            half = int(store_node.attr("output_half", "0"))
            ho = (candidate.radix // 2) if half else 0
            finish_blocks[block] = (tiles[0], tiles[1], ho)
            vg4_skip.update(node_id for node_id, _ in extracts)
            vg4_skip.add(stores[0])
    matrix_lines: list[str] = []
    row_lines: dict[int, list[str]] = {
        block: [] for block in range(candidate.blocks_per_group)}
    cursor_index = {"input": 0, "twiddle": 0}
    for item in scheduled:
        block_attr = item.node.attr("block")
        if (block_attr is not None and
                _row_count(candidate, group, int(block_attr)) == 0):
            continue
        if item.node.id in suppressed:
            continue
        if item.node.id in vg4_skip:
            continue
        if omit_row_epilogue and item.node.id in row_nodes:
            continue
        prefix: list[str] = []
        address_override = None
        if item.node.op == Op.LOAD2:
            role = item.node.attr("role")
            index = int(item.node.attr("index", "0"))
            delta = index - cursor_index[role]
            cursor_index[role] = index
            cursor = f"{role}_cursor{suffix}"
            stride = "row_stride" if role == "input" else "twiddle_stride"
            if delta:
                prefix.append(f"{cursor} += (ptrdiff_t){delta} * "
                              f"(ptrdiff_t){stride};")
            address_override = cursor
        emitted = prefix + (
            _backend.emit_bundle(
                bundles[item.node.id].kind, bundles[item.node.id].width,
                bundles[item.node.id].members, candidate, name, group, suffix)
            if item.node.id in bundles else
            _emit_node(item.node, candidate, name, group, address_override,
                       twiddle_broadcast, suffix, column_predicate,
                       base_name))
        for line in emitted:
            if item.node.id in row_nodes:
                block = int(item.node.attr("block", "0"))
                target = row_lines[block]
            else:
                target = matrix_lines
            target.append(f"      /* ir:{item.node.id} round:{item.round} */ {line}")
    for block, lines in row_lines.items():
        if not lines:
            continue
        matrix_lines.append(
            f"      for (size_t row = 0; row < row_count_{block}; ++row) {{")
        matrix_lines.extend("  " + line for line in lines)
        matrix_lines.append("      }")
    for block, (real_tile, imag_tile, ho) in finish_blocks.items():
        matrix_lines.extend(_backend.emit_finish_rows(
            candidate, block, real_tile, imag_tile, ho, suffix,
            column_predicate, base_name))
    return "\n".join(matrix_lines)


def _actual_vg_stats(candidate: Candidate,
                     scheduled: list[ScheduledNode]) -> dict[str, int]:
    stats = {"vg2": 0, "vg4": 0}
    for group in range(candidate.matrix_iterations):
        bundles, _ = _build_bundles(candidate, scheduled, group)
        for bundle in bundles.values():
            stats[f"vg{bundle.width}"] += 1
        for item in scheduled:
            node = item.node
            if node.op == Op.MATRIX_FMA or (
                    node.op == Op.MATRIX_EXTRACT_H and
                    node.attr("phase") == "matrix_twiddle"):
                width = int(node.attr("width", "2"))
                stats[f"vg{width}"] += 1
    return stats


def _emit_kernel(candidate: Candidate,
                 scheduled: list[ScheduledNode],
                 name_override: str | None = None,
                 transposed_output: bool = False,
                 coefficient_owner: str | None = None,
                 twiddle_broadcast: bool = False,
                 direct_input: bool = False) -> str:
    t = _Types(candidate.precision)
    name = (name_override or
            _kernel_name(candidate.radix, candidate.precision,
                         candidate.direction, candidate.stage))
    table_name = coefficient_owner or name
    real, imag = _coefficient_data(candidate)
    size = candidate.coefficient_rows * candidate.coefficient_columns
    arrays = "" if coefficient_owner else (
        f"static const {t.ctype} {table_name}_wr[{size}] "
        "__attribute__((aligned(128))) = {\n  "
        + ", ".join(_literal(x, candidate.precision) for x in real)
        + "\n};\n"
        f"static const {t.ctype} {table_name}_wi[{size}] "
        "__attribute__((aligned(128))) = {\n  "
        + ", ".join(_literal(x, candidate.precision) for x in imag)
        + "\n};\n")
    if candidate.stage == "first":
        parameters = (f"const {t.complex} *input, size_t row_stride, "
                      f"{t.complex} *output, size_t batch")
    else:
        parameters = (f"const {t.complex} *input, size_t row_stride, "
                      f"size_t input_repeat, "
                      f"const {t.complex} *twiddles, size_t twiddle_stride, "
                      f"size_t twiddle_repeat, {t.complex} *output, "
                      f"size_t batch")
    def iteration_blocks(suffix: str = "", base_name: str = "base",
                         input_offset: str = "0") -> str:
        blocks: list[str] = []
        for group in range(candidate.matrix_iterations):
            declarations: list[str] = []
            for block in range(candidate.blocks_per_group):
                global_block = group * candidate.blocks_per_group + block
                row_base = global_block * candidate.output_block_width
                remaining = max(0, candidate.output_rows - row_base)
                row_count = min(candidate.output_block_width, remaining)
                declarations.extend([
                    f"      const size_t row_base_{block} = {row_base};",
                    f"      const size_t row_count_{block} = {row_count};",
                    f"      svbool_t pg_rows_{block} = svwhilelt_b{t.bits}("
                    f"(uint64_t)0, (uint64_t)row_count_{block});"])
            if direct_input:
                offset = "" if input_offset == "0" else f" + {input_offset}"
                declarations.append(
                    f"      const size_t input_base{suffix} = "
                    f"input_group_base + group_offset{offset};")
                # Prefetch the following iteration, not merely the following
                # row. The offset matters in the second half of a pipelined
                # pair, where this targets the first iteration of the next
                # pair.
                if candidate.radix <= 32:
                    pf_lines = [
                        f"      if (group_offset{offset} + vl < "
                        "input_repeat) {"]
                    for k in range(candidate.radix):
                        pf_lines.append(
                            f"        __builtin_prefetch((const void *)&input["
                            f"input_base{suffix} + vl + (size_t){k} * "
                            f"row_stride], 0, 3);")
                    pf_lines.append("      }")
                    declarations.extend(pf_lines)
            else:
                declarations.append(
                    f"      const size_t input_base{suffix} = {base_name};")
            declarations.append(
                f"      const {t.complex} *input_cursor{suffix} = "
                f"input + input_base{suffix};")
            if candidate.stage == "other":
                if direct_input:
                    twiddle_base = "group_index"
                else:
                    twiddle_base = ("twiddle_group_index"
                                    if twiddle_broadcast else base_name)
                declarations.append(
                    f"      const size_t twiddle_base{suffix} = "
                    f"{twiddle_base};")
                declarations.append(
                    f"      const {t.complex} *twiddle_cursor{suffix} = "
                    f"twiddles + twiddle_base{suffix};")
            body = _scheduled_body(
                candidate, scheduled, table_name, group,
                omit_row_epilogue=transposed_output,
                twiddle_broadcast=twiddle_broadcast, suffix=suffix,
                column_predicate=f"pg_cols{suffix}", base_name=base_name)
            if transposed_output:
                body += "\n" + _backend.emit_vertical_epilogue(
                    candidate, group)
            blocks.append("\n    {\n" + "\n".join(declarations) +
                          "\n      svzero_za();\n" + body + "\n    }")
        return "".join(blocks)

    blocks = iteration_blocks()
    pipeline_two = direct_input and candidate.batch_pipeline_depth == 2
    if direct_input:
        if pipeline_two:
            paired = iteration_blocks() + iteration_blocks(
                "_b", "base_b", "vl")
            loop = f"""  for (size_t group_index = 0, group_base = 0,
              input_group_base = 0;
       group_base < batch;
       ++group_index, group_base += input_repeat,
           input_group_base += {candidate.radix} * input_repeat) {{
    size_t group_offset = 0;
    for (; group_offset + vl < input_repeat &&
           group_base + group_offset + vl < batch;
         group_offset += 2 * vl) {{
      const size_t base = group_base + group_offset;
      const size_t base_b = base + vl;
      size_t active = input_repeat - group_offset;
      if (active > vl) active = vl;
      if (active > batch - base) active = batch - base;
      size_t active_b = input_repeat - group_offset - vl;
      if (active_b > vl) active_b = vl;
      if (active_b > batch - base_b) active_b = batch - base_b;
      svbool_t pg_cols = svwhilelt_b{t.bits}((uint64_t)0,
                                             (uint64_t)active);
      svbool_t pg_cols_b = svwhilelt_b{t.bits}((uint64_t)0,
                                               (uint64_t)active_b);
{paired}
    }}
    for (; group_offset < input_repeat && group_base + group_offset < batch;
         group_offset += vl) {{
      const size_t base = group_base + group_offset;
      size_t active = input_repeat - group_offset;
      if (active > vl) active = vl;
      if (active > batch - base) active = batch - base;
      svbool_t pg_cols = svwhilelt_b{t.bits}((uint64_t)0,
                                             (uint64_t)active);
{blocks}
    }}
  }}"""
            loop_header = loop
            loop_footer = ""
        else:
            loop_header = f"""  for (size_t group_index = 0, group_base = 0,
              input_group_base = 0;
       group_base < batch;
       ++group_index, group_base += input_repeat,
           input_group_base += {candidate.radix} * input_repeat) {{
    for (size_t group_offset = 0; group_offset < input_repeat;
         group_offset += vl) {{
      const size_t base = group_base + group_offset;
      size_t active = input_repeat - group_offset;
      if (active > vl) active = vl;
      if (active > batch - base) active = batch - base;
      svbool_t pg_cols = svwhilelt_b{t.bits}((uint64_t)0,
                                             (uint64_t)active);"""
            loop_footer = "    }\n  }"
    elif twiddle_broadcast:
        loop_header = f"""  for (size_t twiddle_group_index = 0,
              group_base = 0;
       group_base < batch;
       ++twiddle_group_index, group_base += twiddle_repeat) {{
    for (size_t group_offset = 0; group_offset < twiddle_repeat;
         group_offset += vl) {{
      const size_t base = group_base + group_offset;
      size_t active = twiddle_repeat - group_offset;
      if (active > vl) active = vl;
      if (active > batch - base) active = batch - base;
      svbool_t pg_cols = svwhilelt_b{t.bits}((uint64_t)0,
                                             (uint64_t)active);"""
        loop_footer = "    }\n  }"
    else:
        loop_header = f"""  for (size_t base = 0; base < batch; base += vl) {{
    svbool_t pg_cols = svwhilelt_b{t.bits}((uint64_t)0,
                                           (uint64_t)(batch - base));"""
        loop_footer = "  }"
    return f"""
{arrays}
/* graph-sha256: {candidate.graph.digest()} */
__attribute__((aligned(128)))
void {name}({parameters}) __arm_streaming __arm_inout("za") {{
  const size_t vl = {t.cnt};
{loop_header}
{'' if pipeline_two else blocks}
{loop_footer}
}}
"""


def emit_candidate_source(candidate: Candidate, name: str,
                          context: str = "normal") -> str:
    """Emit one forced candidate for standalone compile/differential tests."""
    valid = ("normal", "transposed", "broadcast", "direct_broadcast")
    if context not in valid:
        raise ValueError(f"unknown kernel context {context!r}")
    if candidate.stage == "first" and context not in ("normal", "transposed"):
        raise ValueError(f"{context} is not a first-stage context")
    if candidate.stage == "other" and context not in (
            "normal", "broadcast", "direct_broadcast"):
        raise ValueError(f"{context} is not an other-stage context")
    scheduled = schedule(candidate.graph)
    preamble = ('#include <stddef.h>\n#include <stdint.h>\n'
                '#include <arm_sve.h>\n#include <arm_sme.h>\n'
                '#include "mofft.h"\n')
    return preamble + _emit_kernel(
        candidate, scheduled, name,
        transposed_output=context == "transposed",
        twiddle_broadcast=context in ("broadcast", "direct_broadcast"),
        direct_input=context == "direct_broadcast")


def _declaration(candidate: Candidate, name_override: str | None = None) -> str:
    t = _Types(candidate.precision)
    name = name_override or _kernel_name(candidate.radix, candidate.precision,
                                         candidate.direction, candidate.stage)
    if candidate.stage == "first":
        args = f"const {t.complex} *, size_t, {t.complex} *, size_t"
    else:
        args = (f"const {t.complex} *, size_t, size_t, "
                f"const {t.complex} *, size_t, "
                f"size_t, {t.complex} *, size_t")
    return (f'__attribute__((aligned(128))) void {name}({args}) '
            '__arm_streaming __arm_inout("za");')


def _emit_registry(radices: tuple[int, ...],
                   costs: dict[tuple[int, str, str], float],
                   bucket_contexts: set[tuple[int, str, str, str, str, str]],
                   profile: MachineProfile) -> str:
    lines = ['#include "mofft_generated_kernels.h"', '']
    def call(name: str, radix: int, precision: str, direction: str,
             stage: str, context: str, arguments: str) -> str:
        small = (radix, precision, direction, stage, context,
                 "small") in bucket_contexts
        medium = (radix, precision, direction, stage, context,
                  "medium") in bucket_contexts
        rmedium = (radix, precision, direction, stage, context,
                   "rmedium") in bucket_contexts
        rlarge = (radix, precision, direction, stage, context,
                  "rlarge") in bucket_contexts
        rhuge = (radix, precision, direction, stage, context,
                 "rhuge") in bucket_contexts
        # Locality-repeat dispatch applies to the broadcast and
        # direct_broadcast contexts, where the kernel variant is chosen by the
        # live repeat granularity rather than the batch count.  Bands:
        # (32,256]=rmedium, (256,4096]=rlarge, >4096=rhuge; <=32 falls through
        # to the unsuffixed default lowering.  This is mutually exclusive with
        # the batch-bucket path in practice (a wisdom entry is either
        # batch-bucketed or locality-bucketed), so when locality buckets exist
        # they take precedence.
        if context in ("broadcast", "direct_broadcast") and (
                rmedium or rlarge or rhuge):
            var = "tw_repeat" if context == "broadcast" else "input_repeat"
            branches = []
            for rbucket, present, lo, hi in (
                    ("rmedium", rmedium, 32, 256),
                    ("rlarge", rlarge, 256, 4096),
                    ("rhuge", rhuge, 4096, None)):
                if not present:
                    continue
                cond = (f"{var} > {lo} && {var} <= {hi}" if hi is not None
                        else f"{var} > {lo}")
                branches.append(f"if ({cond}) {name}_{rbucket}({arguments})")
            return (" else ".join(branches)
                    + f" else {name}({arguments});")
        if small and medium:
            return (f'if (batch <= 32) {name}_small({arguments}); '
                    f'else if (batch <= 256) {name}_medium({arguments}); '
                    f'else {name}({arguments});')
        if small:
            return (f'if (batch <= 32) {name}_small({arguments}); '
                    f'else {name}({arguments});')
        if medium:
            return (f'if (batch <= 256) {name}_medium({arguments}); '
                    f'else {name}({arguments});')
        return f'{name}({arguments});'

    for precision in PRECISIONS:
        t = _Types(precision)
        lines.extend([
            '__attribute__((aligned(128)))',
            f'void mofft_dispatch_first_{precision}(int radix, int direction, '
            f'const {t.complex} *input, size_t stride, {t.complex} *output, '
            'size_t batch) __arm_streaming __arm_inout("za") {',
            '  switch (radix) {'])
        for radix in radices:
            fwd = _kernel_name(radix, precision, "forward", "first")
            bwd = _kernel_name(radix, precision, "backward", "first")
            args = 'input, stride, output, batch'
            fwd_call = call(fwd, radix, precision, "forward", "first",
                            "normal", args)
            bwd_call = call(bwd, radix, precision, "backward", "first",
                            "normal", args)
            lines.append(f'    case {radix}: if (direction < 0) {{ {fwd_call} }} else {{ {bwd_call} }} return;')
        lines.extend(['    default: return;', '  }', '}', ''])
        lines.extend([
            '__attribute__((aligned(128)))',
            f'void mofft_dispatch_first_transposed_{precision}(int radix, int direction, '
            f'const {t.complex} *input, size_t stride, {t.complex} *output, '
            'size_t batch) __arm_streaming __arm_inout("za") {',
            '  switch (radix) {'])
        for radix in radices:
            fwd = _kernel_name(radix, precision, "forward", "first") + "_t"
            bwd = _kernel_name(radix, precision, "backward", "first") + "_t"
            args = 'input, stride, output, batch'
            fwd_call = call(fwd, radix, precision, "forward", "first",
                            "transposed", args)
            bwd_call = call(bwd, radix, precision, "backward", "first",
                            "transposed", args)
            lines.append(f'    case {radix}: if (direction < 0) {{ {fwd_call} }} else {{ {bwd_call} }} return;')
        lines.extend(['    default: return;', '  }', '}', ''])
        lines.extend([
            '__attribute__((aligned(128)))',
            f'void mofft_dispatch_other_{precision}(int radix, int direction, '
            f'const {t.complex} *input, size_t stride, size_t input_repeat, '
            f'const {t.complex} *tw, '
            f'size_t tw_stride, size_t tw_repeat, {t.complex} *output, '
            f'size_t batch) '
            '__arm_streaming __arm_inout("za") {',
            '  switch (radix) {'])
        for radix in radices:
            fwd = _kernel_name(radix, precision, "forward", "other")
            bwd = _kernel_name(radix, precision, "backward", "other")
            fwd_broadcast = fwd + "_broadcast"
            bwd_broadcast = bwd + "_broadcast"
            fwd_direct = fwd + "_direct_broadcast"
            bwd_direct = bwd + "_direct_broadcast"
            args = ('input, stride, input_repeat, tw, tw_stride, tw_repeat, '
                    'output, batch')
            fd = call(fwd_direct, radix, precision, "forward", "other",
                      "direct_broadcast", args)
            bd = call(bwd_direct, radix, precision, "backward", "other",
                      "direct_broadcast", args)
            fb = call(fwd_broadcast, radix, precision, "forward", "other",
                      "broadcast", args)
            bb = call(bwd_broadcast, radix, precision, "backward", "other",
                      "broadcast", args)
            fn = call(fwd, radix, precision, "forward", "other", "normal", args)
            bn = call(bwd, radix, precision, "backward", "other", "normal", args)
            lines.append(f'    case {radix}: if (input_repeat != 0) {{ if (direction < 0) {{ {fd} }} else {{ {bd} }} }} else if (tw_repeat != 1) {{ if (direction < 0) {{ {fb} }} else {{ {bb} }} }} else {{ if (direction < 0) {{ {fn} }} else {{ {bn} }} }} return;')
        lines.extend(['    default: return;', '  }', '}', ''])
    lines.extend(['double mofft_generated_radix_cost_stage(int radix, int precision_bits, int first_stage) {',
                  '  switch (radix) {'])
    for radix in radices:
        f32_first = costs[(radix, "fp32", "first")]
        f32_other = costs[(radix, "fp32", "other")]
        f64_first = costs[(radix, "fp64", "first")]
        f64_other = costs[(radix, "fp64", "other")]
        lines.append(
            f'    case {radix}: return precision_bits == 32 ? '
            f'(first_stage ? {f32_first:.17g} : {f32_other:.17g}) : '
            f'(first_stage ? {f64_first:.17g} : {f64_other:.17g});')
    lines.extend(['    default: return 1.0e300;', '  }', '}', '',
                  'double mofft_generated_radix_cost(int radix, int precision_bits) {',
                  '  return mofft_generated_radix_cost_stage(radix, precision_bits, 0);',
                  '}', '',
                  'double mofft_generated_memory_cost(size_t working_set_bytes,',
                  '                                   size_t read_bytes,',
                  '                                   size_t write_bytes) {'])
    for _, level in sorted(profile.memory_hierarchy.levels.items(),
                           key=lambda item: item[1].capacity_bytes):
        lines.extend([
            f'  if (working_set_bytes <= {level.capacity_bytes}ULL) {{',
            f'    double read_cost = (double)read_bytes / '
            f'{level.read_bandwidth_bytes_per_cost_unit:.17g};',
            f'    double write_cost = (double)write_bytes / '
            f'{level.write_bandwidth_bytes_per_cost_unit:.17g};',
        ])
        lines.extend([
            f'    double mixed_cost = (double)(read_bytes + write_bytes) / '
            f'{level.effective_mixed_bandwidth:.17g};',
            '    double bandwidth_cost = read_cost > write_cost ? '
            'read_cost : write_cost;',
            '    if (mixed_cost > bandwidth_cost) '
            'bandwidth_cost = mixed_cost;',
            f'    return bandwidth_cost + {level.latency_cost:.17g};',
        ])
        lines.extend([
            '  }'])
    lines.extend(['  return 1.0e300;', '}', ''])
    lines.extend([
        'double mofft_generated_transpose_cost(size_t working_set_bytes,',
        '                                      size_t data_bytes,',
        '                                      size_t columns, size_t batch,',
        '                                      size_t complex_bytes,',
        '                                      int blocked) {',
    ])
    if profile.layout_cases:
        lines.extend([
            '  double best_score = 1.0e300;',
            '  double best_relative_bandwidth = 1.0;',
        ])
        for case in profile.layout_cases:
            case_blocked = int(case.operation == "blocked_transpose")
            lines.extend([
                f'  if (blocked == {case_blocked}) {{',
                f'    double column_ratio = columns > {case.columns}ULL ?',
                f'        (double)columns / {case.columns}.0 :',
                f'        (double){case.columns} / (double)columns;',
                f'    double batch_ratio = batch > {case.batch}ULL ?',
                f'        (double)batch / {case.batch}.0 :',
                f'        (double){case.batch} / (double)batch;',
                f'    double working_set_ratio = working_set_bytes > '
                f'{case.working_set_bytes}ULL ?',
                f'        (double)working_set_bytes / {case.working_set_bytes}.0 :',
                f'        (double){case.working_set_bytes} / '
                '(double)working_set_bytes;',
                '    double score = column_ratio + batch_ratio + '
                'working_set_ratio;',
                '    if (score < best_score) {',
                '      best_score = score;',
                f'      best_relative_bandwidth = '
                f'{case.relative_bandwidth:.17g};',
                '    }',
                '  }',
            ])
        lines.extend([
            '  return mofft_generated_memory_cost(working_set_bytes,',
            '                                     data_bytes, data_bytes) /',
            '         best_relative_bandwidth;',
        ])
    else:
        lines.extend([
            '  size_t cache_line_bytes = mofft_generated_cache_line_bytes();',
            '  size_t useful_line_bytes = batch > SIZE_MAX / complex_bytes ?',
            '      SIZE_MAX : batch * complex_bytes;',
            '  size_t read_bytes = data_bytes;',
            '  (void)columns;',
            '  if (!blocked && useful_line_bytes < cache_line_bytes) {',
            '    read_bytes = data_bytes > SIZE_MAX / cache_line_bytes ?',
            '        SIZE_MAX : data_bytes * cache_line_bytes;',
            '    if (read_bytes != SIZE_MAX) {',
            '      read_bytes = read_bytes > SIZE_MAX - useful_line_bytes + 1 ?',
            '          SIZE_MAX : (read_bytes + useful_line_bytes - 1) /',
            '                         useful_line_bytes;',
            '    }',
            '  }',
            '  return mofft_generated_memory_cost(working_set_bytes,',
            '                                     read_bytes, data_bytes);',
        ])
    lines.extend(['}', ''])
    lines.extend([
        'size_t mofft_generated_cache_line_bytes(void) {',
        f'  return (size_t){profile.memory_hierarchy.cache_line_bytes};',
        '}', ''])
    return "\n".join(lines)


def emit(profile: MachineProfile, output: str | Path,
         radices: tuple[int, ...] = RADICES,
         kernel_wisdom: dict | None = None,
         temp_tile_rotation: str = "auto") -> dict:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_backend = backend_for_architecture(profile.architecture)
    if selected_backend is not _backend:
        raise ValueError(
            f"{selected_backend.name} requires its own source generator")
    target = selected_backend.target(profile)
    selections: list[dict] = []
    sources: dict[str, list[str]] = {precision: [] for precision in PRECISIONS}
    declarations: list[str] = []
    costs: dict[tuple[int, str, str], float] = {}
    bucket_contexts: set[tuple[int, str, str, str, str, str]] = set()
    wisdom_index: dict[tuple[int, str, str, str, str, str], dict] = {}
    if kernel_wisdom is not None:
        if kernel_wisdom.get("compiler_sha256") != _compiler_digest():
            raise ValueError("kernel wisdom was measured with a different compiler")
        if kernel_wisdom.get("profile_sha256") != profile_digest(profile):
            raise ValueError("kernel wisdom was measured with a different profile")
        for entry in kernel_wisdom.get("entries", []):
            batch = entry.get("batch")
            locality_repeat = entry.get("locality_repeat")
            if locality_repeat is not None:
                # Locality-repeat wisdom: the variant was measured at a fixed
                # input/twiddle repeat granularity.  Bucket by the dispatch
                # band that granularity falls in so the registry can select it
                # at runtime from the live repeat count, not the batch count.
                lr = int(locality_repeat)
                if lr <= 256:
                    bucket = "rmedium"
                elif lr <= 4096:
                    bucket = "rlarge"
                else:
                    bucket = "rhuge"
            else:
                bucket = entry.get("batch_bucket")
                if bucket is None:
                    if batch is not None and int(batch) <= 32:
                        bucket = "small"
                    elif batch is not None and int(batch) <= 256:
                        bucket = "medium"
                    else:
                        bucket = "default"
            if bucket not in ("small", "medium", "default",
                              "rmedium", "rlarge", "rhuge"):
                raise ValueError(f"unknown kernel wisdom bucket {bucket!r}")
            key = (int(entry["radix"]), entry["precision"],
                   entry["direction"], entry["stage"],
                   entry.get("context", "normal"), bucket)
            current = wisdom_index.get(key)
            if current is None or float(entry["median_nanoseconds"]) < float(
                    current["median_nanoseconds"]):
                wisdom_index[key] = entry
    # Keep the complete forward power-of-two path ahead of tunable mixed-radix
    # code. Changing a non-power candidate then cannot move otherwise identical
    # power kernels to different instruction-cache sets.
    emission_radices = tuple(
        radix for radix in radices if radix & (radix - 1) == 0) + tuple(
        radix for radix in radices if radix & (radix - 1) != 0)
    for precision in PRECISIONS:
        for direction in DIRECTIONS:
            for radix in emission_radices:
                for stage in STAGES:
                    candidates = enumerate_candidates(radix, precision,
                                                      stage, direction,
                                                      temp_tile_rotation,
                                                      target)
                    ranked = rank(candidates, profile)
                    def select(context: str, bucket: str = "default"):
                        selected_item, selected_cost = ranked[0]
                        reason = "analytical-model"
                        measured = None
                        wisdom = wisdom_index.get(
                            (radix, precision, direction, stage, context,
                             bucket))
                        if wisdom is None:
                            return selected_item, selected_cost, reason, measured
                        matches = [item for item in ranked if
                                   item[0].pattern.value == wisdom["pattern"] and
                                   item[0].matrix_vg_width == int(
                                       wisdom["matrix_vg_width"]) and
                                   item[0].rotate_temp_tiles == bool(
                                       wisdom.get("rotate_temp_tiles", False)) and
                                   item[0].batch_pipeline_depth == int(
                                       wisdom.get("batch_pipeline_depth", 1))]
                        if not matches:
                            raise ValueError(
                                "kernel wisdom selects a missing candidate for "
                                f"r{radix}/{precision}/{direction}/{stage}/"
                                f"{context}/{bucket}")
                        selected_item, selected_cost = matches[0]
                        return (selected_item, selected_cost,
                                "empirical-kernel-wisdom",
                                float(wisdom["median_nanoseconds"]))

                    selected, cost, selection_reason, measured_nanoseconds = (
                        select("normal"))
                    scheduled = schedule(selected.graph)
                    kernel_source = _emit_kernel(selected, scheduled)
                    vg_stats = _actual_vg_stats(selected, scheduled)
                    sources[precision].append(kernel_source)
                    declarations.append(_declaration(selected))
                    transposed_digest = None
                    broadcast_digest = None
                    direct_broadcast_digest = None
                    variants = {}
                    default_contexts: dict[
                        str, tuple[Candidate, str, str, dict]] = {
                        "normal": (selected, normal_name := _kernel_name(
                            radix, precision, direction, stage), normal_name,
                            {})}
                    if stage == "first":
                        transposed_name = normal_name + "_t"
                        transposed, transposed_cost, transposed_reason, transposed_ns = (
                            select("transposed"))
                        transposed_schedule = schedule(transposed.graph)
                        transposed_source = _emit_kernel(
                            transposed, transposed_schedule, transposed_name,
                            transposed_output=True,
                            coefficient_owner=(normal_name if
                                transposed.graph.digest() == selected.graph.digest()
                                else None))
                        sources[precision].append(transposed_source)
                        declarations.append(_declaration(transposed,
                                                         transposed_name))
                        transposed_digest = sha256(
                            transposed_source.encode()).hexdigest()
                        variants["transposed"] = {
                            "pattern": transposed.pattern.value,
                            "matrix_vg_width": transposed.matrix_vg_width,
                            "rotate_temp_tiles": transposed.rotate_temp_tiles,
                            "batch_pipeline_depth":
                                transposed.batch_pipeline_depth,
                            "selection_reason": transposed_reason,
                            "measured_nanoseconds": transposed_ns,
                            "cost": asdict(transposed_cost),
                            "graph_sha256": transposed.graph.digest(),
                            "emission_sha256": transposed_digest,
                        }
                        default_contexts["transposed"] = (
                            transposed, transposed_name,
                            (normal_name if transposed.graph.digest() ==
                             selected.graph.digest() else transposed_name),
                            {"transposed_output": True})
                    else:
                        broadcast_name = normal_name + "_broadcast"
                        broadcast, broadcast_cost, broadcast_reason, broadcast_ns = (
                            select("broadcast"))
                        broadcast_schedule = schedule(broadcast.graph)
                        broadcast_source = _emit_kernel(
                            broadcast, broadcast_schedule, broadcast_name,
                            coefficient_owner=(normal_name if
                                broadcast.graph.digest() == selected.graph.digest()
                                else None),
                            twiddle_broadcast=True)
                        sources[precision].append(broadcast_source)
                        declarations.append(_declaration(broadcast,
                                                         broadcast_name))
                        broadcast_digest = sha256(
                            broadcast_source.encode()).hexdigest()
                        direct_name = normal_name + "_direct_broadcast"
                        direct, direct_cost, direct_reason, direct_ns = select(
                            "direct_broadcast")
                        direct_schedule = schedule(direct.graph)
                        direct_source = _emit_kernel(
                            direct, direct_schedule, direct_name,
                            coefficient_owner=(normal_name if
                                direct.graph.digest() == selected.graph.digest()
                                else None),
                            twiddle_broadcast=True, direct_input=True)
                        sources[precision].append(direct_source)
                        declarations.append(_declaration(direct,
                                                         direct_name))
                        direct_broadcast_digest = sha256(
                            direct_source.encode()).hexdigest()
                        variants["broadcast"] = {
                            "pattern": broadcast.pattern.value,
                            "matrix_vg_width": broadcast.matrix_vg_width,
                            "rotate_temp_tiles": broadcast.rotate_temp_tiles,
                            "batch_pipeline_depth":
                                broadcast.batch_pipeline_depth,
                            "selection_reason": broadcast_reason,
                            "measured_nanoseconds": broadcast_ns,
                            "cost": asdict(broadcast_cost),
                            "graph_sha256": broadcast.graph.digest(),
                            "emission_sha256": broadcast_digest,
                        }
                        variants["direct_broadcast"] = {
                            "pattern": direct.pattern.value,
                            "matrix_vg_width": direct.matrix_vg_width,
                            "rotate_temp_tiles": direct.rotate_temp_tiles,
                            "batch_pipeline_depth":
                                direct.batch_pipeline_depth,
                            "selection_reason": direct_reason,
                            "measured_nanoseconds": direct_ns,
                            "cost": asdict(direct_cost),
                            "graph_sha256": direct.graph.digest(),
                            "emission_sha256": direct_broadcast_digest,
                        }
                        default_contexts["broadcast"] = (
                            broadcast, broadcast_name,
                            (normal_name if broadcast.graph.digest() ==
                             selected.graph.digest() else broadcast_name),
                            {"twiddle_broadcast": True})
                        default_contexts["direct_broadcast"] = (
                            direct, direct_name,
                            (normal_name if direct.graph.digest() ==
                             selected.graph.digest() else direct_name),
                            {"twiddle_broadcast": True,
                             "direct_input": True})
                    bucket_variants = {}
                    for context, (default_candidate, default_name,
                                  coefficient_owner,
                                  lowering) in default_contexts.items():
                        for bucket, max_batch in (("small", 32),
                                                  ("medium", 256)):
                            bucket_key = (radix, precision, direction, stage,
                                          context, bucket)
                            if bucket_key not in wisdom_index:
                                continue
                            item, item_cost, item_reason, item_ns = select(
                                context, bucket)
                            item_name = default_name + f"_{bucket}"
                            item_schedule = schedule(item.graph)
                            item_source = _emit_kernel(
                                item, item_schedule, item_name,
                                coefficient_owner=(coefficient_owner if
                                    item.graph.digest() ==
                                    default_candidate.graph.digest() else None),
                                **lowering)
                            sources[precision].append(item_source)
                            declarations.append(_declaration(item, item_name))
                            item_digest = sha256(
                                item_source.encode()).hexdigest()
                            bucket_contexts.add((radix, precision, direction,
                                                 stage, context, bucket))
                            bucket_variants[f"{context}_{bucket}"] = {
                                "max_batch": max_batch,
                                "pattern": item.pattern.value,
                                "matrix_vg_width": item.matrix_vg_width,
                                "rotate_temp_tiles": item.rotate_temp_tiles,
                                "batch_pipeline_depth":
                                    item.batch_pipeline_depth,
                                "selection_reason": item_reason,
                                "measured_nanoseconds": item_ns,
                                "cost": asdict(item_cost),
                                "graph_sha256": item.graph.digest(),
                                "emission_sha256": item_digest,
                            }
                        if context in ("broadcast", "direct_broadcast"):
                            for rbucket, band_lo, band_hi in (
                                    ("rmedium", 32, 256),
                                    ("rlarge", 256, 4096),
                                    ("rhuge", 4096, None)):
                                bucket_key = (radix, precision, direction,
                                              stage, context, rbucket)
                                if bucket_key not in wisdom_index:
                                    continue
                                item, item_cost, item_reason, item_ns = select(
                                    context, rbucket)
                                item_name = default_name + f"_{rbucket}"
                                item_schedule = schedule(item.graph)
                                item_source = _emit_kernel(
                                    item, item_schedule, item_name,
                                    coefficient_owner=(coefficient_owner if
                                        item.graph.digest() ==
                                        default_candidate.graph.digest()
                                        else None),
                                    **lowering)
                                sources[precision].append(item_source)
                                declarations.append(_declaration(item, item_name))
                                item_digest = sha256(
                                    item_source.encode()).hexdigest()
                                bucket_contexts.add((radix, precision, direction,
                                                     stage, context, rbucket))
                                bucket_variants[f"{context}_default_{rbucket}"] = {
                                    "locality_band": [band_lo, band_hi],
                                    "pattern": item.pattern.value,
                                    "matrix_vg_width": item.matrix_vg_width,
                                    "rotate_temp_tiles": item.rotate_temp_tiles,
                                    "batch_pipeline_depth":
                                        item.batch_pipeline_depth,
                                    "selection_reason": item_reason,
                                    "measured_nanoseconds": item_ns,
                                    "cost": asdict(item_cost),
                                    "graph_sha256": item.graph.digest(),
                                    "emission_sha256": item_digest,
                                }
                    # Complete-plan modeling supplies transform-scale traffic
                    # and combines it with this compute lower bound once per
                    # stage.  Exporting the kernel-local total here would
                    # apply max(compute, memory) twice at different scales.
                    costs[(radix, precision, stage)] = min(
                        costs.get((radix, precision, stage), float("inf")),
                        cost.compute_cost)
                    selections.append({
                        "radix": radix, "precision": precision,
                        "direction": direction, "stage": stage,
                        "target": selected.target.name,
                        "matrix_mapping": selected.mapping.value,
                        "pattern": selected.pattern.value,
                        "matrix_vg_width": selected.matrix_vg_width,
                        "rotate_temp_tiles": selected.rotate_temp_tiles,
                        "batch_pipeline_depth":
                            selected.batch_pipeline_depth,
                        "selection_reason": selection_reason,
                        "measured_nanoseconds": measured_nanoseconds,
                        "cost": asdict(cost),
                        "graph_sha256": selected.graph.digest(),
                        "emission_sha256": sha256(
                            kernel_source.encode()).hexdigest(),
                        "transposed_emission_sha256": transposed_digest,
                        "broadcast_emission_sha256": broadcast_digest,
                        "direct_broadcast_emission_sha256":
                            direct_broadcast_digest,
                        "variants": variants,
                        "bucket_variants": bucket_variants,
                        "coefficient_shape": [selected.coefficient_rows,
                                              selected.coefficient_columns, 2],
                        "emitted_node_ids": [item.node.id for item in scheduled],
                        "schedule": {
                            "nodes": len(scheduled),
                            "rounds": max((item.round for item in scheduled),
                                          default=0) + 1,
                            "spills": sum(item.spill for item in scheduled),
                            "vg2": vg_stats["vg2"],
                            "vg4": vg_stats["vg4"],
                            "matrix_tiles": sorted({
                                item.matrix_tile for item in scheduled
                                if item.node.op == Op.OUTER_PRODUCT_ACCUM}),
                        },
                    })
    header = """#ifndef MOFFT_GENERATED_KERNELS_H
#define MOFFT_GENERATED_KERNELS_H
#include <stddef.h>
#include <arm_sme.h>
#include "mofft.h"
""" + "\n".join(declarations) + """
__attribute__((aligned(128)))
void mofft_dispatch_first_fp32(int, int, const mofft_complex_f32 *, size_t,
                               mofft_complex_f32 *, size_t) __arm_streaming __arm_inout("za");
__attribute__((aligned(128)))
void mofft_dispatch_first_fp64(int, int, const mofft_complex_f64 *, size_t,
                               mofft_complex_f64 *, size_t) __arm_streaming __arm_inout("za");
__attribute__((aligned(128)))
void mofft_dispatch_first_transposed_fp32(
    int, int, const mofft_complex_f32 *, size_t,
    mofft_complex_f32 *, size_t) __arm_streaming __arm_inout("za");
__attribute__((aligned(128)))
void mofft_dispatch_first_transposed_fp64(
    int, int, const mofft_complex_f64 *, size_t,
    mofft_complex_f64 *, size_t) __arm_streaming __arm_inout("za");
__attribute__((aligned(128)))
void mofft_dispatch_other_fp32(int, int, const mofft_complex_f32 *, size_t,
                               size_t, const mofft_complex_f32 *, size_t,
                               size_t, mofft_complex_f32 *, size_t) __arm_streaming __arm_inout("za");
__attribute__((aligned(128)))
void mofft_dispatch_other_fp64(int, int, const mofft_complex_f64 *, size_t,
                               size_t, const mofft_complex_f64 *, size_t,
                               size_t, mofft_complex_f64 *, size_t) __arm_streaming __arm_inout("za");
double mofft_generated_radix_cost(int radix, int precision_bits);
double mofft_generated_radix_cost_stage(int radix, int precision_bits,
                                        int first_stage);
double mofft_generated_memory_cost(size_t working_set_bytes,
                                   size_t read_bytes, size_t write_bytes);
double mofft_generated_transpose_cost(size_t working_set_bytes,
                                      size_t data_bytes, size_t columns,
                                      size_t batch, size_t complex_bytes,
                                      int blocked);
size_t mofft_generated_cache_line_bytes(void);
#endif
"""
    (output_path / "mofft_generated_kernels.h").write_text(header)
    preamble = ('#include "mofft_generated_kernels.h"\n#include <stdint.h>\n'
                '#include <arm_sve.h>\n#include <arm_sme.h>\n')
    for precision, chunks in sources.items():
        (output_path / f"mofft_kernels_{precision}.c").write_text(
            preamble + "\n".join(chunks))
    (output_path / "mofft_kernel_registry.c").write_text(
        _emit_registry(radices, costs, bucket_contexts, profile))
    manifest = {
        "schema_version": 4,
        "generator_version": __version__,
        "compiler_sha256": _compiler_digest(),
        "profile": profile.to_json(),
        "kernel_wisdom": ({
            "formal_measurement": bool(kernel_wisdom.get(
                "formal_measurement", False)),
            "compiler_sha256": kernel_wisdom["compiler_sha256"],
            "profile_sha256": kernel_wisdom["profile_sha256"],
            "entry_count": len(kernel_wisdom.get("entries", [])),
        } if kernel_wisdom is not None else None),
        "radices": list(radices),
        "temp_tile_rotation": temp_tile_rotation,
        "emitted_kernel_count": len(declarations),
        "selections": selections,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["input_sha256"] = sha256(canonical.encode()).hexdigest()
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
