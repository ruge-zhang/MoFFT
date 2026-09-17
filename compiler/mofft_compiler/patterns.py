from __future__ import annotations

from dataclasses import dataclass
import math

from .ir import Graph, Op, ValueKind
from .mapping import MatrixMapping
from .rewrites import (Pattern, algebraic_patterns,
                       evaluate_pattern as evaluate_pattern)
from .targets.arm_sme import ARM_SME_512
from .targets.base import TargetCapabilities


@dataclass(frozen=True)
class Candidate:
    radix: int
    precision: str
    stage: str
    direction: str
    pattern: Pattern
    graph: Graph
    matrix_iterations: int
    post_iterations: int
    coefficient_rows: int
    coefficient_columns: int
    output_rows: int
    output_block_width: int
    output_blocks: int
    blocks_per_group: int
    matrix_vg_width: int
    rotate_temp_tiles: bool
    mapping: MatrixMapping
    target: TargetCapabilities
    emittable: bool = True
    batch_pipeline_depth: int = 1

    @property
    def vector_reuse(self) -> bool:
        return self.pattern in (Pattern.VECTOR_REUSE,
                                Pattern.LIKE_TERMS_VECTOR_REUSE)

    @property
    def like_terms(self) -> bool:
        return self.pattern in (Pattern.LIKE_TERMS,
                                Pattern.LIKE_TERMS_EVEN_ODD,
                                Pattern.LIKE_TERMS_VECTOR_REUSE)

    @property
    def even_odd(self) -> bool:
        return self.pattern in (Pattern.EVEN_ODD,
                                Pattern.LIKE_TERMS_EVEN_ODD)


def lanes_for(precision: str,
              target: TargetCapabilities = ARM_SME_512) -> int:
    return target.lanes(precision)


def matrix_tiles_for(precision: str,
                     target: TargetCapabilities = ARM_SME_512) -> int:
    return target.accumulator_tiles(precision)


def legal_patterns(radix: int, precision: str = "fp32",
                   target: TargetCapabilities = ARM_SME_512
                   ) -> tuple[Pattern, ...]:
    """Patterns that are feasible to lower for ``radix`` on a target.

    Algebraic legality comes from :mod:`rewrites`; this layer adds the target's
    profitability/feasibility threshold without changing the mathematics.
    """
    lanes = lanes_for(precision, target)
    algebraic = set(algebraic_patterns(radix))
    result = [Pattern.DIRECT, Pattern.LIKE_TERMS]
    # At exactly one architectural vector the quarter-wave reconstruction is
    # still profitable and fully legal.  Radix-16 FP32 is the important M5
    # boundary case: excluding equality removes a useful VG4/vector-reuse
    # design from the search space.
    if Pattern.VECTOR_REUSE in algebraic and radix >= lanes:
        result.extend((Pattern.VECTOR_REUSE,
                       Pattern.LIKE_TERMS_VECTOR_REUSE))
    if Pattern.EVEN_ODD in algebraic and radix >= lanes:
        result.extend((Pattern.EVEN_ODD,
                       Pattern.LIKE_TERMS_EVEN_ODD))
    return tuple(result)


def _load_complex(graph: Graph, role: str, index: int) -> tuple[int, int]:
    pair = graph.add(Op.LOAD2, kind=ValueKind.VECTOR_PAIR,
                     predicate="cols", role=role, index=index)
    return (graph.add(Op.GET, (pair,), kind=ValueKind.VECTOR, component=0),
            graph.add(Op.GET, (pair,), kind=ValueKind.VECTOR, component=1))


def _load_input(graph: Graph, j: int, stage: str) -> tuple[int, int]:
    xr, xi = _load_complex(graph, "input", j)
    if stage == "first":
        return xr, xi
    tr, ti = _load_complex(graph, "twiddle", j)
    rr = graph.add(Op.MUL, (xr, tr), kind=ValueKind.VECTOR,
                   predicate="cols", phase="twiddle")
    rr = graph.add(Op.FMLA_VECTOR, (rr, xi, ti), kind=ValueKind.VECTOR,
                   predicate="cols", subtract=1, phase="twiddle")
    ii = graph.add(Op.MUL, (xr, ti), kind=ValueKind.VECTOR,
                   predicate="cols", phase="twiddle")
    ii = graph.add(Op.FMLA_VECTOR, (ii, xi, tr), kind=ValueKind.VECTOR,
                   predicate="cols", subtract=0, phase="twiddle")
    return rr, ii


def _prepare_inputs(graph: Graph, radix: int, precision: str, stage: str,
                    temp_tile_base: int | None,
                    num_temp_tiles: int,
                    matrix_vg_width: int,
                    input_order: list[int]) -> list[tuple[int, int]]:
    if stage == "first" or temp_tile_base is None:
        return [_load_input(graph, j, stage) for j in range(radix)]
    width = matrix_vg_width
    group_kind = (ValueKind.VECTOR_GROUP4 if width == 4
                  else ValueKind.VECTOR_PAIR)
    inputs: list[tuple[int, int] | None] = [None] * radix
    zero_vector = graph.add(Op.BROADCAST, kind=ValueKind.VECTOR, value=0)
    one_vector = graph.add(Op.BROADCAST, kind=ValueKind.VECTOR, value=1)
    # Rotate the temporary accumulator tile across width-groups so each group
    # accumulates into its own tile. The old code reused one tile, forcing
    # every group to wait for the previous group's extract before re-zeroing
    # (sequential). With distinct tiles, group i's loads / FmlaZa can issue
    # while group i-1's result is still being read and fed to the output
    # outer products, overlapping loads with matrix compute. When width-groups
    # exceed the available tiles, the rotation cycles; clearing a tile then
    # depends on the previous extract of that same tile (a pipeline of depth
    # num_temp_tiles), not on
    # all previous extracts. The backend decides how these logical tiles map
    # to architectural matrix state.
    matrix_allocations: dict[int, int] = {}
    tile_extracts: dict[int, list[int]] = {}
    for group_index, start in enumerate(range(0, radix, width)):
        temp_tile = temp_tile_base + (group_index % num_temp_tiles)
        xr: list[int] = []
        xi: list[int] = []
        tr: list[int] = []
        ti: list[int] = []
        for lane in range(width):
            order_index = start + lane
            if order_index < radix:
                j = input_order[order_index]
                ar, ai = _load_complex(graph, "input", j)
                br, bi = _load_complex(graph, "twiddle", j)
            else:
                ar = ai = bi = zero_vector
                br = one_vector
            xr.append(ar); xi.append(ai); tr.append(br); ti.append(bi)
        pxr = graph.add(Op.PACK, xr, kind=group_kind, width=width)
        pxi = graph.add(Op.PACK, xi, kind=group_kind, width=width)
        ptr = graph.add(Op.PACK, tr, kind=group_kind, width=width)
        pti = graph.add(Op.PACK, ti, kind=group_kind, width=width)
        if temp_tile not in matrix_allocations:
            matrix_allocations[temp_tile] = graph.add(
                Op.MATRIX_ALLOC, kind=ValueKind.MATRIX, tile=temp_tile,
                role="twiddle")
        temp = graph.add(
            Op.MATRIX_CLEAR,
            (matrix_allocations[temp_tile],
             *tile_extracts.get(temp_tile, [])),
            kind=ValueKind.MATRIX, tile=temp_tile, phase="twiddle")
        temp = graph.add(Op.MATRIX_FMA, (temp, pxr, ptr),
                         kind=ValueKind.MATRIX, tile=temp_tile,
                         group=0, subtract=0, width=width)
        temp = graph.add(Op.MATRIX_FMA, (temp, pxi, pti),
                         kind=ValueKind.MATRIX, tile=temp_tile,
                         group=0, subtract=1, width=width)
        temp = graph.add(Op.MATRIX_FMA, (temp, pxr, pti),
                         kind=ValueKind.MATRIX, tile=temp_tile,
                         group=1, subtract=0, width=width)
        temp = graph.add(Op.MATRIX_FMA, (temp, pxi, ptr),
                         kind=ValueKind.MATRIX, tile=temp_tile,
                         group=1, subtract=0, width=width)
        current_extracts: list[int] = []
        for lane in range(width):
            order_index = start + lane
            if order_index >= radix:
                break
            j = input_order[order_index]
            if precision == "fp64" and width == 4:
                slice_index = (lane // 2) * 4
                real_component = (lane % 2) * 2
            elif precision == "fp32" and width == 2:
                slice_index = lane * 8
                real_component = 0
            else:
                slice_index = lane * 4
                real_component = 0
            packed = graph.add(Op.MATRIX_EXTRACT_H, (temp,), kind=group_kind,
                               tile=temp_tile, slice=slice_index,
                               phase="matrix_twiddle", width=width)
            current_extracts.append(packed)
            real = graph.add(Op.GET, (packed,), kind=ValueKind.VECTOR,
                             component=real_component)
            imag = graph.add(Op.GET, (packed,), kind=ValueKind.VECTOR,
                             component=real_component + 1)
            inputs[j] = (real, imag)
        tile_extracts[temp_tile] = current_extracts
    return [value for value in inputs if value is not None]


def _load_coefficient(graph: Graph, j: int, block: int,
                      vector_reuse: bool,
                      reuse_period_blocks: int = 1) -> tuple[int, int]:
    layout = "quarter" if vector_reuse else "full"
    source_block = block % reuse_period_blocks if vector_reuse else block
    raw_real = graph.add(Op.LOAD, kind=ValueKind.VECTOR,
                         predicate=f"rows:{source_block}", role="coefficient",
                         component="real", index=j, block=source_block,
                         layout=layout)
    raw_imag = graph.add(Op.LOAD, kind=ValueKind.VECTOR,
                         predicate=f"rows:{source_block}", role="coefficient",
                         component="imag", index=j, block=source_block,
                         layout=layout)
    if not vector_reuse:
        return raw_real, raw_imag
    real = graph.add(Op.COEFF_RECONSTRUCT, (raw_real, raw_imag),
                     kind=ValueKind.VECTOR, predicate=f"rows:{block}",
                     component="real", index=j, block=block)
    imag = graph.add(Op.COEFF_RECONSTRUCT, (raw_real, raw_imag),
                     kind=ValueKind.VECTOR, predicate=f"rows:{block}",
                     component="imag", index=j, block=block)
    return real, imag


def _accumulate(graph: Graph, accumulators: list[int], block: int,
                wr: int, wi: int, xr: int, xi: int,
                tile_base: int | None = None) -> None:
    real_tile = 2 * block if tile_base is None else tile_base
    imag_tile = real_tile + 1
    pred = f"rows:{block},cols"
    accumulators[real_tile] = graph.add(
        Op.OUTER_PRODUCT_ACCUM, (accumulators[real_tile], wr, xr),
        kind=ValueKind.MATRIX, predicate=pred, block=block,
        tile=real_tile, subtract=0)
    accumulators[real_tile] = graph.add(
        Op.OUTER_PRODUCT_ACCUM, (accumulators[real_tile], wi, xi),
        kind=ValueKind.MATRIX, predicate=pred, block=block,
        tile=real_tile, subtract=1)
    accumulators[imag_tile] = graph.add(
        Op.OUTER_PRODUCT_ACCUM, (accumulators[imag_tile], wr, xi),
        kind=ValueKind.MATRIX, predicate=pred, block=block,
        tile=imag_tile, subtract=0)
    accumulators[imag_tile] = graph.add(
        Op.OUTER_PRODUCT_ACCUM, (accumulators[imag_tile], wi, xr),
        kind=ValueKind.MATRIX, predicate=pred, block=block,
        tile=imag_tile, subtract=0)


def _accumulate_all(graph: Graph, accumulators: list[int], blocks: int,
                    j: int, xr: int, xi: int, vector_reuse: bool,
                    reuse_period_blocks: int) -> None:
    for block in range(blocks):
        wr, wi = _load_coefficient(graph, j, block, vector_reuse,
                                   reuse_period_blocks)
        _accumulate(graph, accumulators, block, wr, wi, xr, xi)


def _build_direct(graph: Graph, radix: int, blocks: int,
                  inputs: list[tuple[int, int]],
                  vector_reuse: bool,
                  reuse_period_blocks: int) -> list[int]:
    accumulators = [graph.add(Op.MATRIX_ALLOC, kind=ValueKind.MATRIX, tile=i)
                    for i in range(2 * blocks)]
    for j in range(radix):
        xr, xi = inputs[j]
        _accumulate_all(graph, accumulators, blocks, j, xr, xi, vector_reuse,
                        reuse_period_blocks)
    return accumulators


def _build_like_terms(graph: Graph, radix: int, blocks: int,
                      inputs: list[tuple[int, int]],
                      vector_reuse: bool,
                      reuse_period_blocks: int) -> list[int]:
    accumulators = [graph.add(Op.MATRIX_ALLOC, kind=ValueKind.MATRIX, tile=i)
                    for i in range(2 * blocks)]
    one = graph.add(Op.BROADCAST, kind=ValueKind.VECTOR, value=1)
    zero = graph.add(Op.BROADCAST, kind=ValueKind.VECTOR, value=0)
    x0r, x0i = inputs[0]
    for block in range(blocks):
        _accumulate(graph, accumulators, block, one, zero, x0r, x0i)
    for j in range(1, (radix + 1) // 2):
        xr0, xi0 = inputs[j]
        xr1, xi1 = inputs[radix - j]
        sx = graph.add(Op.ADD, (xr0, xr1), kind=ValueKind.VECTOR,
                       predicate="cols")
        si = graph.add(Op.ADD, (xi0, xi1), kind=ValueKind.VECTOR,
                       predicate="cols")
        dx = graph.add(Op.SUB, (xr0, xr1), kind=ValueKind.VECTOR,
                       predicate="cols")
        di = graph.add(Op.SUB, (xi0, xi1), kind=ValueKind.VECTOR,
                       predicate="cols")
        for block in range(blocks):
            wr, wi = _load_coefficient(graph, j, block, vector_reuse,
                                       reuse_period_blocks)
            real_tile, imag_tile = 2 * block, 2 * block + 1
            pred = f"rows:{block},cols"
            accumulators[real_tile] = graph.add(
                Op.OUTER_PRODUCT_ACCUM, (accumulators[real_tile], wr, sx),
                kind=ValueKind.MATRIX, predicate=pred, block=block,
                tile=real_tile, subtract=0)
            accumulators[real_tile] = graph.add(
                Op.OUTER_PRODUCT_ACCUM, (accumulators[real_tile], wi, di),
                kind=ValueKind.MATRIX, predicate=pred, block=block,
                tile=real_tile, subtract=1)
            accumulators[imag_tile] = graph.add(
                Op.OUTER_PRODUCT_ACCUM, (accumulators[imag_tile], wr, si),
                kind=ValueKind.MATRIX, predicate=pred, block=block,
                tile=imag_tile, subtract=0)
            accumulators[imag_tile] = graph.add(
                Op.OUTER_PRODUCT_ACCUM, (accumulators[imag_tile], wi, dx),
                kind=ValueKind.MATRIX, predicate=pred, block=block,
                tile=imag_tile, subtract=0)
    if radix % 2 == 0:
        xr, xi = inputs[radix // 2]
        _accumulate_all(graph, accumulators, blocks, radix // 2,
                        xr, xi, vector_reuse, reuse_period_blocks)
    return accumulators


def _finish(graph: Graph, accumulators: list[int], blocks: int) -> None:
    for block in range(blocks):
        real = graph.add(Op.MATRIX_EXTRACT_H, (accumulators[2 * block],),
                         kind=ValueKind.VECTOR, predicate="cols",
                         tile=2 * block, block=block, row="row")
        imag = graph.add(Op.MATRIX_EXTRACT_H, (accumulators[2 * block + 1],),
                         kind=ValueKind.VECTOR, predicate="cols",
                         tile=2 * block + 1, block=block, row="row")
        graph.add(Op.STORE2, (real, imag), kind=ValueKind.MEMORY,
                  predicate="cols", block=block, row="row")


def _build_radix32_like_terms_even_odd(
        graph: Graph, blocks: int,
        inputs: list[tuple[int, int]]) -> list[int]:
    # Accumulate the two output halves directly so that the epilogue does not
    # have to reconstruct them with row-wise additions and subtractions.
    radix = 32
    accumulators = [graph.add(Op.MATRIX_ALLOC, kind=ValueKind.MATRIX, tile=i)
                    for i in range(4 * blocks)]

    def accumulate_pair(block: int, base: int, wr: int, wi: int,
                        sx: int, si: int, dx: int, di: int,
                        negate: bool = False) -> None:
        pred = f"rows:{block},cols"
        accumulators[base] = graph.add(
            Op.OUTER_PRODUCT_ACCUM, (accumulators[base], wr, sx),
            kind=ValueKind.MATRIX, predicate=pred, block=block,
            tile=base, subtract=1 if negate else 0)
        accumulators[base] = graph.add(
            Op.OUTER_PRODUCT_ACCUM, (accumulators[base], wi, di),
            kind=ValueKind.MATRIX, predicate=pred, block=block,
            tile=base, subtract=0 if negate else 1)
        accumulators[base + 1] = graph.add(
            Op.OUTER_PRODUCT_ACCUM, (accumulators[base + 1], wr, si),
            kind=ValueKind.MATRIX, predicate=pred, block=block,
            tile=base + 1, subtract=1 if negate else 0)
        accumulators[base + 1] = graph.add(
            Op.OUTER_PRODUCT_ACCUM, (accumulators[base + 1], wi, dx),
            kind=ValueKind.MATRIX, predicate=pred, block=block,
            tile=base + 1, subtract=1 if negate else 0)

    def add_direct_input(j: int, xr: int, xi: int) -> None:
        for block in range(blocks):
            wr, wi = _load_coefficient(graph, j, block, False)
            _accumulate(graph, accumulators, block, wr, wi, xr, xi,
                        tile_base=4 * block)
            _accumulate(graph, accumulators, block, wr, wi, xr, xi,
                        tile_base=4 * block + 2)

    add_direct_input(0, *inputs[0])
    for j in range(1, (radix + 1) // 2):
        xr0, xi0 = inputs[j]
        xr1, xi1 = inputs[radix - j]
        sx = graph.add(Op.ADD, (xr0, xr1), kind=ValueKind.VECTOR,
                       predicate="cols")
        si = graph.add(Op.ADD, (xi0, xi1), kind=ValueKind.VECTOR,
                       predicate="cols")
        dx = graph.add(Op.SUB, (xr0, xr1), kind=ValueKind.VECTOR,
                       predicate="cols")
        di = graph.add(Op.SUB, (xi0, xi1), kind=ValueKind.VECTOR,
                       predicate="cols")
        for block in range(blocks):
            wr, wi = _load_coefficient(graph, j, block, False)
            base = 4 * block
            accumulate_pair(block, base, wr, wi, sx, si, dx, di)
            accumulate_pair(block, base + 2, wr, wi, sx, si, dx, di,
                            negate=bool(j % 2))
    add_direct_input(radix // 2, *inputs[radix // 2])
    return accumulators


def _build_even_odd(graph: Graph, radix: int, blocks: int,
                    inputs: list[tuple[int, int]],
                    like_terms: bool) -> list[int]:
    if like_terms and radix == 32:
        return _build_radix32_like_terms_even_odd(graph, blocks, inputs)

    accumulators = [graph.add(Op.MATRIX_ALLOC, kind=ValueKind.MATRIX, tile=i)
                    for i in range(4 * blocks)]

    def add_input(j: int, xr: int, xi: int) -> None:
        for block in range(blocks):
            wr, wi = _load_coefficient(graph, j, block, False)
            base = 4 * block + (0 if j % 2 == 0 else 2)
            _accumulate(graph, accumulators, block, wr, wi, xr, xi,
                        tile_base=base)

    if not like_terms:
        for j in range(radix):
            add_input(j, *inputs[j])
        return accumulators

    add_input(0, *inputs[0])
    for j in range(1, (radix + 1) // 2):
        xr0, xi0 = inputs[j]
        xr1, xi1 = inputs[radix - j]
        sx = graph.add(Op.ADD, (xr0, xr1), kind=ValueKind.VECTOR,
                       predicate="cols")
        si = graph.add(Op.ADD, (xi0, xi1), kind=ValueKind.VECTOR,
                       predicate="cols")
        dx = graph.add(Op.SUB, (xr0, xr1), kind=ValueKind.VECTOR,
                       predicate="cols")
        di = graph.add(Op.SUB, (xi0, xi1), kind=ValueKind.VECTOR,
                       predicate="cols")
        for block in range(blocks):
            wr, wi = _load_coefficient(graph, j, block, False)
            base = 4 * block + (0 if j % 2 == 0 else 2)
            pred = f"rows:{block},cols"
            accumulators[base] = graph.add(
                Op.OUTER_PRODUCT_ACCUM, (accumulators[base], wr, sx),
                kind=ValueKind.MATRIX, predicate=pred, block=block,
                tile=base, subtract=0)
            accumulators[base] = graph.add(
                Op.OUTER_PRODUCT_ACCUM, (accumulators[base], wi, di),
                kind=ValueKind.MATRIX, predicate=pred, block=block,
                tile=base, subtract=1)
            accumulators[base + 1] = graph.add(
                Op.OUTER_PRODUCT_ACCUM, (accumulators[base + 1], wr, si),
                kind=ValueKind.MATRIX, predicate=pred, block=block,
                tile=base + 1, subtract=0)
            accumulators[base + 1] = graph.add(
                Op.OUTER_PRODUCT_ACCUM, (accumulators[base + 1], wi, dx),
                kind=ValueKind.MATRIX, predicate=pred, block=block,
                tile=base + 1, subtract=0)
    add_input(radix // 2, *inputs[radix // 2])
    return accumulators


def _finish_even_odd(graph: Graph, accumulators: list[int],
                     blocks: int) -> None:
    for block in range(blocks):
        values = [graph.add(Op.MATRIX_EXTRACT_H, (accumulators[4 * block + i],),
                            kind=ValueKind.VECTOR, predicate="cols",
                            tile=4 * block + i, block=block, row="row")
                  for i in range(4)]
        low_r = graph.add(Op.ADD, (values[0], values[2]),
                          kind=ValueKind.VECTOR, predicate="cols", block=block)
        low_i = graph.add(Op.ADD, (values[1], values[3]),
                          kind=ValueKind.VECTOR, predicate="cols", block=block)
        high_r = graph.add(Op.SUB, (values[0], values[2]),
                           kind=ValueKind.VECTOR, predicate="cols", block=block)
        high_i = graph.add(Op.SUB, (values[1], values[3]),
                           kind=ValueKind.VECTOR, predicate="cols", block=block)
        graph.add(Op.STORE2, (low_r, low_i), kind=ValueKind.MEMORY,
                  predicate="cols", block=block, output_half=0, row="row")
        graph.add(Op.STORE2, (high_r, high_i), kind=ValueKind.MEMORY,
                  predicate="cols", block=block, output_half=1, row="row")


def _finish_even_odd_direct(graph: Graph, accumulators: list[int],
                            blocks: int) -> None:
    for block in range(blocks):
        values = [graph.add(Op.MATRIX_EXTRACT_H, (accumulators[4 * block + i],),
                            kind=ValueKind.VECTOR, predicate="cols",
                            tile=4 * block + i, block=block, row="row")
                  for i in range(4)]
        graph.add(Op.STORE2, (values[0], values[1]), kind=ValueKind.MEMORY,
                  predicate="cols", block=block, output_half=0, row="row")
        graph.add(Op.STORE2, (values[2], values[3]), kind=ValueKind.MEMORY,
                  predicate="cols", block=block, output_half=1, row="row")


def build_candidate(radix: int, precision: str, stage: str,
                    direction: str, pattern: Pattern,
                    matrix_vg_width: int = 4,
                    rotate_temp_tiles: bool = True,
                    target: TargetCapabilities = ARM_SME_512,
                    mapping: MatrixMapping =
                    MatrixMapping.SPLIT_COMPLEX_OUTER,
                    batch_pipeline_depth: int = 1) -> Candidate:
    if not target.supports(mapping):
        raise ValueError(f"{target.name} does not support {mapping.value}")
    if mapping != MatrixMapping.SPLIT_COMPLEX_OUTER:
        raise ValueError(f"mapping lowering is not implemented: "
                         f"{mapping.value}")
    if pattern not in legal_patterns(radix, precision, target):
        raise ValueError(f"illegal {pattern.value} for radix {radix} {precision}")
    if stage not in ("first", "other"):
        raise ValueError(f"unknown stage: {stage}")
    if matrix_vg_width not in target.matrix_group_widths:
        raise ValueError(f"unsupported matrix VG width: {matrix_vg_width}")
    if batch_pipeline_depth not in (1, 2):
        raise ValueError("batch pipeline depth must be 1 or 2")
    if batch_pipeline_depth == 2 and not (
            stage == "other" and precision == "fp64" and
            radix in (8, 16, 32)):
        raise ValueError("batch pipeline depth 2 is only validated for "
                         "FP64 radix-8/16/32 other-stage kernels")
    vector_reuse = pattern in (Pattern.VECTOR_REUSE,
                               Pattern.LIKE_TERMS_VECTOR_REUSE)
    like_terms = pattern in (Pattern.LIKE_TERMS,
                             Pattern.LIKE_TERMS_VECTOR_REUSE,
                             Pattern.LIKE_TERMS_EVEN_ODD)
    lanes = lanes_for(precision, target)
    even_odd = pattern in (Pattern.EVEN_ODD, Pattern.LIKE_TERMS_EVEN_ODD)
    coefficient_columns = (radix // 2 if even_odd else
                           (radix // 4 if vector_reuse else radix))
    output_rows = radix // 2 if even_odd else radix
    block_width = min(lanes, coefficient_columns)
    output_blocks = math.ceil(output_rows / block_width)
    tiles_per_block = 4 if even_odd else 2
    tile_count = matrix_tiles_for(precision, target)
    blocks_per_group = min(output_blocks, tile_count // tiles_per_block)
    temp_tile: int | None = None
    num_temp_tiles = 0
    if stage == "other":
        matrix_blocks = (tile_count - 1) // tiles_per_block
        if matrix_blocks >= 1:
            blocks_per_group = min(output_blocks, matrix_blocks)
            # Radix-32 FP64 Direct and Like-Terms have four output blocks.
            # Splitting them evenly across two passes leaves four temporary
            # ZA tiles for twiddle preparation in each pass.  The former 3+1
            # split left only two temporaries in the first pass and then
            # repeated the complete input preparation for one output block.
            if (radix == 32 and precision == "fp64" and
                    pattern in (Pattern.DIRECT, Pattern.LIKE_TERMS)):
                blocks_per_group = min(output_blocks, 2)
            temp_tile = blocks_per_group * tiles_per_block
            available_temp_tiles = tile_count - temp_tile
            num_temp_tiles = (available_temp_tiles
                              if rotate_temp_tiles else 1)
    reuse_period_blocks = math.ceil(coefficient_columns / block_width)
    graph = Graph(precision)
    input_order = list(range(radix))
    chunked_like_terms = (
        stage == "other" and pattern == Pattern.LIKE_TERMS and
        matrix_vg_width == 4 and
        (radix == 32 or (radix == 64 and rotate_temp_tiles)))
    if chunked_like_terms:
        # Visit symmetric inputs in VG4-sized chunks.  This keeps the address
        # streams mostly monotonic while preserving the Like-Terms pairing.
        # For r64, restrict this lowering to the rotating-tile candidate: the
        # single-tile variant does not hide the longer VG4 dependency chain.
        input_order = []
        for start in range(0, radix // 2, 4):
            input_order.extend(range(start, start + 4))
            input_order.extend(range(radix - 1 - start,
                                     radix - 5 - start, -1))
    elif like_terms:
        input_order = [0]
        for j in range(1, (radix + 1) // 2):
            input_order.extend((j, radix - j))
        if radix % 2 == 0:
            input_order.append(radix // 2)
    prepared_inputs = _prepare_inputs(graph, radix, precision, stage, temp_tile,
                                      num_temp_tiles,
                                      matrix_vg_width, input_order)
    if even_odd:
        accumulators = _build_even_odd(graph, radix, blocks_per_group,
                                       prepared_inputs, like_terms)
        (_finish_even_odd_direct if like_terms and radix == 32
         else _finish_even_odd)(
            graph, accumulators, blocks_per_group)
    else:
        accumulators = (_build_like_terms(graph, radix, blocks_per_group,
                                         prepared_inputs, vector_reuse,
                                         reuse_period_blocks)
                        if like_terms else
                        _build_direct(graph, radix, blocks_per_group,
                                      prepared_inputs, vector_reuse,
                                      reuse_period_blocks))
        _finish(graph, accumulators, blocks_per_group)
    graph.validate()
    coefficient_rows = ((radix // 2 + 1 if radix % 2 == 0
                         else (radix + 1) // 2)
                        if like_terms else radix)
    groups = math.ceil(output_blocks / blocks_per_group)
    return Candidate(
        radix=radix, precision=precision, stage=stage, direction=direction,
        pattern=pattern, graph=graph, matrix_iterations=groups,
        post_iterations=1, coefficient_rows=coefficient_rows,
        coefficient_columns=coefficient_columns, output_rows=output_rows,
        output_block_width=block_width, output_blocks=output_blocks,
        blocks_per_group=blocks_per_group,
        matrix_vg_width=matrix_vg_width,
        rotate_temp_tiles=stage == "other" and rotate_temp_tiles,
        mapping=mapping, target=target, emittable=True,
        batch_pipeline_depth=batch_pipeline_depth)


def enumerate_candidates(radix: int, precision: str, stage: str,
                         direction: str,
                         temp_tile_rotation: str = "auto",
                         target: TargetCapabilities = ARM_SME_512
                         ) -> list[Candidate]:
    if temp_tile_rotation not in ("auto", "on", "off"):
        raise ValueError(f"unknown temporary-tile rotation mode: "
                         f"{temp_tile_rotation}")
    widths = (target.matrix_group_widths if stage == "other" else
              (max(target.matrix_group_widths),))
    rotations = ((False, True) if stage == "other" and
                 temp_tile_rotation == "auto" else
                 (temp_tile_rotation == "on",))
    pipeline_depths = ((1, 2) if stage == "other" and
                       precision == "fp64" and radix in (8, 16, 32) else (1,))
    return [build_candidate(
                radix, precision, stage, direction, pattern, width, rotate,
                target=target, batch_pipeline_depth=pipeline_depth)
            for pattern in legal_patterns(radix, precision, target)
            for width in widths
            for rotate in rotations
            for pipeline_depth in pipeline_depths]
