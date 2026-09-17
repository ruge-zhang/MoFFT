"""ARM SME ACLE lowering for MoFFT's neutral matrix IR."""

from __future__ import annotations

import math

from ..ir import Node, Op, ValueKind
from ..patterns import Candidate
from ..profile import MachineProfile
from ..targets.arm_sme import arm_sme_target
from ..targets.base import TargetCapabilities


class ArmSMETypes:
    def __init__(self, precision: str):
        self.ctype = "float" if precision == "fp32" else "double"
        self.complex = ("mofft_complex_f32" if precision == "fp32"
                        else "mofft_complex_f64")
        self.bits = "32" if precision == "fp32" else "64"
        self.short = "f32" if precision == "fp32" else "f64"
        self.vector = ("svfloat32_t" if precision == "fp32"
                       else "svfloat64_t")
        self.pair = ("svfloat32x2_t" if precision == "fp32"
                     else "svfloat64x2_t")
        self.quad = ("svfloat32x4_t" if precision == "fp32"
                     else "svfloat64x4_t")
        self.cnt = "svcntw()" if precision == "fp32" else "svcntd()"


class ArmSMEBackend:
    name = "arm-sme"

    @staticmethod
    def target(profile: MachineProfile) -> TargetCapabilities:
        return arm_sme_target(profile.streaming_vector_bits)

    @staticmethod
    def types(precision: str) -> ArmSMETypes:
        return ArmSMETypes(precision)

    @staticmethod
    def value_name(node_id: int, suffix: str = "") -> str:
        return f"v{node_id}{suffix}"

    @staticmethod
    def pair_name(node_id: int, suffix: str = "") -> str:
        return f"p{node_id}{suffix}"

    @staticmethod
    def _coefficient_phase(
            candidate: Candidate, j: int, quadrant: int) -> int:
        direction = -1 if candidate.direction == "forward" else 1
        return (direction * j * quadrant) % 4

    def emit_node(
            self, node: Node, candidate: Candidate, table_name: str,
            group: int, address_override: str | None = None,
            twiddle_broadcast: bool = False, suffix: str = "",
            column_predicate: str = "pg_cols",
            base_name: str = "base") -> list[str]:
        t = self.types(candidate.precision)
        short = t.short
        value = self.value_name
        pair = self.pair_name
        op = node.op
        if op == Op.MATRIX_ALLOC:
            return []
        if op == Op.MATRIX_CLEAR:
            base_mask = "0x11ULL" if candidate.precision == "fp32" else "1ULL"
            return [f"svzero_mask_za({base_mask} << {node.attr('tile')});"]
        if op == Op.BROADCAST:
            return [f"{t.vector} {value(node.id, suffix)} = "
                    f"svdup_n_{short}({node.attr('value')});"]
        if op == Op.LOAD2:
            role = node.attr("role")
            if address_override is not None:
                address = address_override
            elif role == "input":
                address = (f"&input[{node.attr('index')} * row_stride + "
                           f"{base_name}]")
            elif role == "twiddle":
                address = (f"&twiddles[{node.attr('index')} * twiddle_stride "
                           f"+ {base_name}]")
            else:
                raise ValueError(f"unknown LOAD2 role {role}")
            if role == "twiddle" and twiddle_broadcast:
                return [f"{t.pair} {pair(node.id, suffix)} = "
                        f"svcreate2_{short}("
                        f"svdup_n_{short}(({address})->real), "
                        f"svdup_n_{short}(({address})->imag));"]
            return [f"{t.pair} {pair(node.id, suffix)} = "
                    f"svld2_{short}({column_predicate}, "
                    f"(const {t.ctype} *){address});"]
        if op == Op.GET:
            source = pair(node.inputs[0], suffix)
            source_kind = candidate.graph.nodes[node.inputs[0]].kind
            get_width = 4 if source_kind == ValueKind.VECTOR_GROUP4 else 2
            return [f"{t.vector} {value(node.id, suffix)} = "
                    f"svget{get_width}_{short}({source}, "
                    f"{node.attr('component')});"]
        if op == Op.PACK:
            width = int(node.attr("width", "2"))
            packed_type = t.quad if width == 4 else t.pair
            arguments = ", ".join(value(item, suffix)
                                  for item in node.inputs)
            return [f"{packed_type} {pair(node.id, suffix)} = "
                    f"svcreate{width}_{short}({arguments});"]
        if op == Op.LOAD:
            if node.attr("role") != "coefficient":
                raise ValueError(f"unknown LOAD role {node.attr('role')}")
            array = "wr" if node.attr("component") == "real" else "wi"
            block = int(node.attr("block", "0"))
            global_block = group * candidate.blocks_per_group + block
            row_base = global_block * candidate.output_block_width
            if node.attr("layout") == "quarter":
                reuse_period = math.ceil(
                    candidate.coefficient_columns /
                    candidate.output_block_width)
                source_block = (
                    (group * candidate.blocks_per_group + block) %
                    reuse_period)
                offset = str(source_block * candidate.output_block_width)
            else:
                offset = str(row_base)
            stride = candidate.coefficient_columns
            address = (f"&{table_name}_{array}[{node.attr('index')} * "
                       f"{stride} + {offset}]")
            return [f"{t.vector} {value(node.id, suffix)} = "
                    f"svld1_{short}(pg_rows_{block}, {address});"]
        if op == Op.COEFF_RECONSTRUCT:
            component = node.attr("component")
            j = int(node.attr("index", "0"))
            block = int(node.attr("block", "0"))
            global_block = group * candidate.blocks_per_group + block
            row_base = global_block * candidate.output_block_width
            quadrant = row_base // candidate.coefficient_columns
            phase = self._coefficient_phase(candidate, j, quadrant)
            raw_r = value(node.inputs[0], suffix)
            raw_i = value(node.inputs[1], suffix)
            pg_rows = f"pg_rows_{block}"
            if phase == 0:
                expression = raw_r if component == "real" else raw_i
            elif phase == 1:
                expression = (f"svneg_{short}_x({pg_rows}, {raw_i})"
                              if component == "real" else raw_r)
            elif phase == 2:
                raw = raw_r if component == "real" else raw_i
                expression = f"svneg_{short}_x({pg_rows}, {raw})"
            else:
                expression = (raw_i if component == "real" else
                              f"svneg_{short}_x({pg_rows}, {raw_r})")
            return [f"{t.vector} {value(node.id, suffix)} = {expression};"]
        if op in (Op.ADD, Op.SUB, Op.MUL):
            intrinsic = {Op.ADD: "svadd", Op.SUB: "svsub",
                         Op.MUL: "svmul"}[op]
            return [f"{t.vector} {value(node.id, suffix)} = "
                    f"{intrinsic}_{short}_m({column_predicate}, "
                    f"{value(node.inputs[0], suffix)}, "
                    f"{value(node.inputs[1], suffix)});"]
        if op == Op.NEG:
            block = int(node.attr("block", "0"))
            return [f"{t.vector} {value(node.id, suffix)} = "
                    f"svneg_{short}_x(pg_rows_{block}, "
                    f"{value(node.inputs[0], suffix)});"]
        if op == Op.OUTER_PRODUCT_ACCUM:
            intrinsic = ("svmops" if node.attr("subtract") == "1"
                         else "svmopa")
            tile = node.attr("tile")
            block = node.attr("block")
            return [f"{intrinsic}_za{t.bits}_{short}_m("
                    f"{tile}, pg_rows_{block}, {column_predicate}, "
                    f"{value(node.inputs[1], suffix)}, "
                    f"{value(node.inputs[2], suffix)});"]
        if op == Op.MATRIX_FMA:
            width = int(node.attr("width", "2"))
            matrix_group = int(node.attr("group", "0"))
            tile = int(node.attr("tile", "0"))
            index = tile + (
                4 if candidate.precision == "fp32" else 8) * matrix_group
            intrinsic = ("svmls" if node.attr("subtract") == "1"
                         else "svmla")
            return [f"{intrinsic}_za{t.bits}_{short}_vg1x{width}({index}, "
                    f"{pair(node.inputs[1], suffix)}, "
                    f"{pair(node.inputs[2], suffix)});"]
        if op == Op.MATRIX_EXTRACT_H:
            tile = node.attr("tile")
            if node.attr("phase") == "matrix_twiddle":
                width = int(node.attr("width", "2"))
                packed_type = t.quad if width == 4 else t.pair
                return [f"{packed_type} {pair(node.id, suffix)} = "
                        f"svread_hor_za{t.bits}_{short}_vg{width}({tile}, "
                        f"{node.attr('slice')});"]
            return [f"{t.vector} {value(node.id, suffix)} = "
                    f"svread_hor_za{t.bits}_{short}_m("
                    f"svdup_n_{short}(0), {column_predicate}, {tile}, row);"]
        if op == Op.FMLA_VECTOR:
            intrinsic = ("svmls" if node.attr("subtract") == "1"
                         else "svmla")
            return [f"{t.vector} {value(node.id, suffix)} = "
                    f"{intrinsic}_{short}_m({column_predicate}, "
                    f"{value(node.inputs[0], suffix)}, "
                    f"{value(node.inputs[1], suffix)}, "
                    f"{value(node.inputs[2], suffix)});"]
        if op == Op.STORE2:
            block = node.attr("block")
            half_offset = (candidate.radix // 2
                           if node.attr("output_half", "0") == "1" else 0)
            return [f"svst2_{short}({column_predicate}, "
                    f"({t.ctype} *)&output[(row_base_{block} + row + "
                    f"{half_offset}) * batch + {base_name}], "
                    f"svcreate2_{short}("
                    f"{value(node.inputs[0], suffix)}, "
                    f"{value(node.inputs[1], suffix)}));"]
        raise ValueError(f"no ARM SME ACLE lowering for {op.value}")

    def emit_bundle(
            self, kind: str, width: int, members: tuple[int, ...],
            candidate: Candidate, table_name: str, group: int,
            suffix: str = "") -> list[str]:
        if kind != "load":
            raise ValueError(f"unknown ARM SME bundle kind {kind}")
        t = self.types(candidate.precision)
        nodes = [candidate.graph.nodes[node_id] for node_id in members]
        packed_type = t.quad if width == 4 else t.pair
        packed_name = f"vg_{kind}_{members[0]}{suffix}"
        first = nodes[0]
        array = "wr" if first.attr("component") == "real" else "wi"
        block = int(first.attr("block", "0"))
        global_block = group * candidate.blocks_per_group + block
        offset = global_block * candidate.output_block_width
        address = (f"&{table_name}_{array}[{first.attr('index')} * "
                   f"{candidate.coefficient_columns} + {offset}]")
        lines = [
            f"{packed_type} {packed_name} = svld1_{t.short}_x{width}("
            f"svreinterpret_c(pg_rows_{block}), {address});"]
        for component, node in enumerate(nodes):
            lines.append(
                f"{t.vector} {self.value_name(node.id, suffix)} = "
                f"svget{width}_{t.short}({packed_name}, {component});")
        return lines

    def emit_finish_rows(
            self, candidate: Candidate, block: int, real_tile: int,
            imag_tile: int, half_offset: int, suffix: str = "",
            column_predicate: str = "pg_cols",
            base_name: str = "base") -> list[str]:
        """Use grouped ZA reads for a direct/like-terms row epilogue."""
        t = self.types(candidate.precision)
        bits, short = t.bits, t.short
        lines = [
            f"      for (size_t row = 0; row + 4 <= row_count_{block}; row += 4) {{",
            f"        svfloat{bits}x4_t _req{suffix} = svread_hor_za{bits}_{short}_vg4({real_tile}, (uint64_t)row);",
            f"        svfloat{bits}x4_t _imq{suffix} = svread_hor_za{bits}_{short}_vg4({imag_tile}, (uint64_t)row);"]
        # svget4 requires a compile-time constant index.
        for lane in range(4):
            lines.append(
                f"        svst2_{short}({column_predicate}, ({t.ctype} *)&output["
                f"(row_base_{block} + row + {lane} + {half_offset}) * batch + {base_name}], "
                f"svcreate2_{short}(svget4_{short}(_req{suffix}, {lane}), "
                f"svget4_{short}(_imq{suffix}, {lane})));"
            )
        lines.extend([
            "      }",
            f"      for (size_t row = (row_count_{block} / 4) * 4; row < row_count_{block}; ++row) {{",
            f"        svfloat{bits}_t _r{suffix} = svread_hor_za{bits}_{short}_m(svdup_n_{short}(0), {column_predicate}, {real_tile}, row);",
            f"        svfloat{bits}_t _i{suffix} = svread_hor_za{bits}_{short}_m(svdup_n_{short}(0), {column_predicate}, {imag_tile}, row);",
            f"        svst2_{short}({column_predicate}, ({t.ctype} *)&output[(row_base_{block} + row + {half_offset}) * batch + {base_name}], svcreate2_{short}(_r{suffix}, _i{suffix}));",
            "      }"])
        return lines

    @staticmethod
    def _row_count(candidate: Candidate, group: int, block: int) -> int:
        global_block = group * candidate.blocks_per_group + block
        row_base = global_block * candidate.output_block_width
        return min(candidate.output_block_width,
                   max(0, candidate.output_rows - row_base))

    def emit_vertical_epilogue(
            self, candidate: Candidate, group: int) -> str:
        """Store ZA columns as batch-major complex rows."""
        t = self.types(candidate.precision)
        even_odd = "even_odd" in candidate.pattern.value
        direct_even_odd_halves = (
            candidate.radix == 32 and
            candidate.pattern.value == "like_terms_even_odd"
        )
        lines = ["      const size_t active_cols =",
                 "          batch - base < vl ? batch - base : vl;",
                 "      for (size_t col = 0; col < active_cols; ++col) {"]
        for block in range(candidate.blocks_per_group):
            if self._row_count(candidate, group, block) == 0:
                continue
            global_block = group * candidate.blocks_per_group + block
            row_base = global_block * candidate.output_block_width
            zero = f"svdup_n_{t.short}(0)"
            if even_odd:
                values = []
                for component in range(4):
                    name = f"vertical_{block}_{component}"
                    tile = 4 * block + component
                    lines.append(
                        f"        {t.vector} {name} = "
                        f"svread_ver_za{t.bits}_{t.short}_m({zero}, "
                        f"pg_rows_{block}, {tile}, (uint32_t)col);")
                    values.append(name)
                if direct_even_odd_halves:
                    # The radix-32 LT+EO graph accumulates the low and high
                    # halves directly into tiles 0/1 and 2/3.  Reapplying the
                    # generic Even-Odd sum/difference here would transform the
                    # already finished outputs a second time.
                    lines.extend([
                        f"        svst2_{t.short}(pg_rows_{block}, ({t.ctype} *)&output["
                        f"(base + col) * {candidate.radix} + {row_base}], "
                        f"svcreate2_{t.short}({values[0]}, {values[1]}));",
                        f"        svst2_{t.short}(pg_rows_{block}, ({t.ctype} *)&output["
                        f"(base + col) * {candidate.radix} + "
                        f"{row_base + candidate.radix // 2}], "
                        f"svcreate2_{t.short}({values[2]}, {values[3]}));",
                    ])
                else:
                    lines.extend([
                        f"        {t.vector} vertical_low_r_{block} = "
                        f"svadd_{t.short}_m(pg_rows_{block}, {values[0]}, {values[2]});",
                        f"        {t.vector} vertical_low_i_{block} = "
                        f"svadd_{t.short}_m(pg_rows_{block}, {values[1]}, {values[3]});",
                        f"        {t.vector} vertical_high_r_{block} = "
                        f"svsub_{t.short}_m(pg_rows_{block}, {values[0]}, {values[2]});",
                        f"        {t.vector} vertical_high_i_{block} = "
                        f"svsub_{t.short}_m(pg_rows_{block}, {values[1]}, {values[3]});",
                        f"        svst2_{t.short}(pg_rows_{block}, ({t.ctype} *)&output["
                        f"(base + col) * {candidate.radix} + {row_base}], "
                        f"svcreate2_{t.short}(vertical_low_r_{block}, "
                        f"vertical_low_i_{block}));",
                        f"        svst2_{t.short}(pg_rows_{block}, ({t.ctype} *)&output["
                        f"(base + col) * {candidate.radix} + "
                        f"{row_base + candidate.radix // 2}], "
                        f"svcreate2_{t.short}(vertical_high_r_{block}, "
                        f"vertical_high_i_{block}));",
                    ])
            else:
                real_tile = 2 * block
                imag_tile = real_tile + 1
                lines.extend([
                    f"        {t.vector} vertical_r_{block} = "
                    f"svread_ver_za{t.bits}_{t.short}_m({zero}, pg_rows_{block}, "
                    f"{real_tile}, (uint32_t)col);",
                    f"        {t.vector} vertical_i_{block} = "
                    f"svread_ver_za{t.bits}_{t.short}_m({zero}, pg_rows_{block}, "
                    f"{imag_tile}, (uint32_t)col);",
                    f"        svst2_{t.short}(pg_rows_{block}, ({t.ctype} *)&output["
                    f"(base + col) * {candidate.radix} + {row_base}], "
                    f"svcreate2_{t.short}(vertical_r_{block}, vertical_i_{block}));",
                ])
        lines.append("      }")
        return "\n".join(lines)


ARM_SME_BACKEND = ArmSMEBackend()
