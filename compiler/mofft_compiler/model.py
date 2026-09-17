from __future__ import annotations

from dataclasses import dataclass
import math

from .patterns import Candidate
from .profile import MachineProfile
from .scheduler import schedule


@dataclass(frozen=True)
class Cost:
    matrix_cost: float
    post_cost: float
    register_pressure_cost: float
    compute_cost: float
    memory_cost: float
    working_set_bytes: int
    cache_footprint_bytes: int
    cache_line_utilization: float
    traffic_bytes: float
    arithmetic_intensity: float
    memory_level: str
    total_cost: float
    bottlenecks: tuple[str, ...]


def estimate(candidate: Candidate, profile: MachineProfile) -> Cost:
    # Matrix-extract results consumed by the scalar row epilogue execute inside
    # a loop over every output row.  Counting their DAG nodes once severely
    # underestimates matrix reads, arithmetic and stores (especially Even-Odd,
    # whose four partial tiles are combined per row).  Compute average counts
    # per matrix group here; `iterations` below scales them to the full radix.
    users: dict[int, list[int]] = {node.id: [] for node in candidate.graph.nodes}
    for node in candidate.graph.nodes:
        for dependency in node.inputs:
            users[dependency].append(node.id)
    row_nodes = {
        node.id for node in candidate.graph.nodes
        if node.op.value in ("matrix_extract_horizontal", "matrix_extract_vertical") and
        node.attr("phase") != "matrix_twiddle"
    }
    frontier = list(row_nodes)
    while frontier:
        for user in users[frontier.pop()]:
            if user not in row_nodes:
                row_nodes.add(user)
                frontier.append(user)

    iterations = max(candidate.matrix_iterations, 1)
    row_multipliers: dict[int, float] = {}
    for block in range(candidate.blocks_per_group):
        executed_rows = 0
        for group in range(iterations):
            global_block = group * candidate.blocks_per_group + block
            remaining = max(
                0, candidate.output_rows -
                global_block * candidate.output_block_width)
            executed_rows += min(candidate.output_block_width, remaining)
        row_multipliers[block] = executed_rows / iterations

    counts: dict[str, float] = {}
    post_counts: dict[str, float] = {}
    for node in candidate.graph.nodes:
        multiplier = 1.0
        if node.id in row_nodes:
            multiplier = row_multipliers.get(
                int(node.attr("block", "0")),
                float(candidate.output_block_width))
            post_counts[node.op.value] = (
                post_counts.get(node.op.value, 0.0) + multiplier)
        counts[node.op.value] = counts.get(node.op.value, 0.0) + multiplier
    # The scheduler assigns each outer product an accumulator-chain round:
    # independent operations share a round, while dependent ones chain across
    # rounds (latency-bound). count/throughput assumes full ILP saturation and
    # therefore under-estimates deep accumulator chains such as the direct DFT's
    # repeated za[r] += fmop(...) accumulation. Use the chain depth as a
    # latency-bound floor so the model can distinguish the direct DFT (deep
    # chain, latency-bound) from simplified expressions (shallower chains).
    scheduled = schedule(candidate.graph)
    outer_product_rounds = [
        item.round for item in scheduled
        if item.node.op.value == "outer_product_accumulate"]
    outer_product_chain_depth = (
        max(outer_product_rounds) + 1 if outer_product_rounds else 0)
    matrix_fma_per_tile: dict[int, int] = {}
    for item in scheduled:
        if item.node.op.value == "matrix_fma":
            tile = int(item.node.attr("tile", "0"))
            matrix_fma_per_tile[tile] = matrix_fma_per_tile.get(tile, 0) + 1
    matrix_fma_chain_depth = max(matrix_fma_per_tile.values(), default=0)
    resource_costs: dict[str, float] = {}
    post_resources: dict[str, float] = {}
    for op, count in counts.items():
        key = candidate.target.instruction_model_key(
            op, candidate.matrix_vg_width)
        if key is None:
            continue
        if op == "matrix_extract_horizontal":
            matrix_extracts = sum(
                node.op.value == "matrix_extract_horizontal" and
                node.attr("phase") == "matrix_twiddle"
                for node in candidate.graph.nodes)
            scalar_extracts = count - matrix_extracts
            group_extract_key = candidate.target.instruction_model_key(
                "matrix_extract_group", candidate.matrix_vg_width)
            for extract_count, extract_key in (
                    (matrix_extracts, group_extract_key),
                    (scalar_extracts, key)):
                if extract_count == 0 or extract_key is None:
                    continue
                precision_key = f"{extract_key}_{candidate.precision}"
                resolved = (precision_key if precision_key in profile.instructions
                            else extract_key)
                if (resolved not in profile.instructions and
                        extract_key == group_extract_key):
                    scalar_precision_key = f"{key}_{candidate.precision}"
                    resolved = (scalar_precision_key
                                if scalar_precision_key in profile.instructions
                                else key)
                if resolved not in profile.instructions:
                    continue
                model = profile.instructions[resolved]
                operation_cost = extract_count / max(model.throughput, 1e-9)
                resource_costs[model.resource] = (
                    resource_costs.get(model.resource, 0.0) + operation_cost)
                if extract_key == key:
                    post_resources[model.resource] = (
                        post_resources.get(model.resource, 0.0) +
                        operation_cost)
            continue
        precision_key = f"{key}_{candidate.precision}"
        if precision_key in profile.instructions:
            key = precision_key
        if key not in profile.instructions:
            continue
        model = profile.instructions[key]
        operation_cost = count / max(model.throughput, 1e-9)
        if op == "outer_product_accumulate" and outer_product_chain_depth:
            operation_cost = max(
                operation_cost, outer_product_chain_depth * model.latency)
        elif op == "matrix_fma" and matrix_fma_chain_depth:
            operation_cost = max(
                operation_cost, matrix_fma_chain_depth * model.latency)
        resource_costs[model.resource] = (
            resource_costs.get(model.resource, 0.0) + operation_cost)
        post_count = post_counts.get(op, 0.0)
        if post_count:
            post_cost_for_op = post_count / max(model.throughput, 1e-9)
            post_resources[model.resource] = (
                post_resources.get(model.resource, 0.0) + post_cost_for_op)
    matrix_resources = dict(resource_costs)
    for resource, resource_cost in post_resources.items():
        matrix_resources[resource] = max(
            0.0, matrix_resources[resource] - resource_cost)
    if candidate.vector_reuse:
        # Quarter-wave coefficient reconstruction is part of the matrix phase.
        # Register-only real/imaginary swaps are free; sign-changing phases
        # emit FNEG and consume the SVE arithmetic resource.  Fold that work
        # into the resource maximum instead of adding it after the maximum:
        # the generated FNEG instructions can overlap independent FMOP/load
        # work and are not a separate serial epilogue.
        negations = 0
        for group in range(candidate.matrix_iterations):
            for block in range(candidate.blocks_per_group):
                global_block = group * candidate.blocks_per_group + block
                row_base = global_block * candidate.output_block_width
                quadrant = row_base // candidate.coefficient_columns
                for node in candidate.graph.nodes:
                    if (node.op.value == "coeff_reconstruct" and
                            int(node.attr("block", "0")) == block):
                        j = int(node.attr("index", "0"))
                        phase = ((-1 if candidate.direction == "forward" else 1)
                                 * j * quadrant) % 4
                        component = node.attr("component")
                        if phase == 2 or (phase == 1 and component == "real") or (
                                phase == 3 and component == "imag"):
                            negations += 1
        add_key = f"sve_add_{candidate.precision}"
        if add_key not in profile.instructions:
            add_key = "sve_add"
        if add_key in profile.instructions:
            neg_model = profile.instructions[add_key]
            negation_cost = (negations / iterations /
                              max(neg_model.throughput, 1e-9))
            matrix_resources[neg_model.resource] = (
                matrix_resources.get(neg_model.resource, 0.0) +
                negation_cost)
            resource_costs[neg_model.resource] = (
                resource_costs.get(neg_model.resource, 0.0) +
                negation_cost)
    matrix_cost = max(matrix_resources.values(), default=0.0)
    # The row epilogue is a dependent pipeline (matrix read -> arithmetic ->
    # store), not a set of freely overlapping matrix resources.  Summing its
    # resource stages matches the generated control/data flow and prevents
    # Even-Odd's mandatory recombination from being treated as free beneath
    # matrix-read throughput.
    post_cost = sum(post_resources.values())
    spill_count = sum(item.spill for item in scheduled)
    spill_cost = 0.0
    for operation in ("sve_load", "sve_store"):
        key = (f"{operation}_{candidate.precision}"
               if f"{operation}_{candidate.precision}" in profile.instructions
               else operation)
        if key in profile.instructions:
            spill_cost += spill_count / max(
                profile.instructions[key].throughput, 1e-9)
    spill_cost *= iterations
    compute_cost = ((matrix_cost + post_cost) * iterations + spill_cost)
    scalar_bytes = 4 if candidate.precision == "fp32" else 8
    vector_bytes = profile.streaming_vector_bits // 8
    coefficient_component_bytes = (candidate.coefficient_rows *
                                   candidate.coefficient_columns * scalar_bytes)
    coefficient_bytes = 2 * coefficient_component_bytes
    complex_vector_bytes = 2 * vector_bytes
    # Separate real/imaginary input, output, and (for later stages) twiddle
    # streams occupy distinct cache lines. Model their line-rounded footprint,
    # not merely the sum of useful bytes, so small/tail-heavy candidates do not
    # receive impossible perfect cache utilization.
    resident_streams = 4 if candidate.stage == "first" else 6
    resident_stream_bytes = candidate.radix * vector_bytes
    working_set_bytes = (coefficient_bytes +
                         resident_streams * resident_stream_bytes)
    cache_line_bytes = profile.memory_hierarchy.cache_line_bytes

    def line_footprint(byte_count: int) -> int:
        return ((byte_count + cache_line_bytes - 1) // cache_line_bytes *
                cache_line_bytes)

    cache_footprint_bytes = (
        2 * line_footprint(coefficient_component_bytes) +
        resident_streams * line_footprint(resident_stream_bytes))
    cache_line_utilization = (working_set_bytes /
                              max(cache_footprint_bytes, 1))
    memory_level_name, memory_level = profile.memory_hierarchy.level_for(
        cache_footprint_bytes)
    read_bytes = 0.0
    write_bytes = 0.0
    for op, count in counts.items():
        if op == "load":
            read_bytes += count * vector_bytes
        elif op == "load2":
            read_bytes += count * complex_vector_bytes
        elif op == "store":
            write_bytes += count * vector_bytes
        elif op == "store2":
            write_bytes += count * complex_vector_bytes
    read_bytes *= iterations
    write_bytes *= iterations
    spill_bytes = spill_count * vector_bytes * iterations
    read_bytes += spill_bytes
    write_bytes += spill_bytes
    # Line under-utilization consumes bandwidth even though it does not add
    # useful FFT data. Sequential full-vector streams remain close to one;
    # coefficient tails and small arrays are charged for their transferred
    # cache-line footprint.
    transferred_read_bytes = read_bytes / max(cache_line_utilization, 1e-9)
    transferred_write_bytes = write_bytes / max(cache_line_utilization, 1e-9)
    mixed_cost = ((transferred_read_bytes + transferred_write_bytes) /
                  max(memory_level.effective_mixed_bandwidth, 1e-9))
    memory_cost = max(
        transferred_read_bytes /
        max(memory_level.read_bandwidth_bytes_per_cost_unit, 1e-9),
        transferred_write_bytes /
        max(memory_level.write_bandwidth_bytes_per_cost_unit, 1e-9),
        mixed_cost,
    ) + memory_level.latency_cost
    lanes = profile.lanes(candidate.precision)
    flop_weights = {
        "add": lanes, "sub": lanes, "mul": lanes,
        "fmla_vector": 2 * lanes,
        "outer_product_accumulate": 2 * lanes * lanes,
        "matrix_fma": 2 * lanes * candidate.matrix_vg_width,
    }
    floating_point_operations = sum(
        count * flop_weights.get(op, 0.0) for op, count in counts.items()
    ) * iterations
    arithmetic_intensity = floating_point_operations / max(
        transferred_read_bytes + transferred_write_bytes, 1e-9)
    total = max(compute_cost, memory_cost)
    maximum = max(resource_costs.values(), default=0.0)
    bottleneck_list = sorted(k for k, v in resource_costs.items()
                             if math.isclose(v, maximum, rel_tol=0.05))
    if memory_cost >= .95 * total:
        bottleneck_list.append(f"memory-{memory_level_name}")
    return Cost(matrix_cost, post_cost, spill_cost, compute_cost, memory_cost,
                working_set_bytes, cache_footprint_bytes,
                cache_line_utilization, read_bytes + write_bytes,
                arithmetic_intensity, memory_level_name, total,
                tuple(sorted(set(bottleneck_list))))


def rank(candidates: list[Candidate],
         profile: MachineProfile) -> list[tuple[Candidate, Cost]]:
    ranked = [(candidate, estimate(candidate, profile))
              for candidate in candidates if candidate.emittable]
    # The lower-bound model can tie VG2 and VG4 even though VG4 emits fewer
    # grouped matrix operations.  For moderate radices, Direct, Like-Terms,
    # and Even-Odd remain below the register-pressure point where that shorter
    # instruction stream wins on Apple SME2.  Combined rewrites retain the
    # conservative VG2 preference because their extra temporaries make VG4
    # pressure visible.  Larger radices keep the conservative default except
    # radix-64 Like-Terms, whose chunked VG4 lowering reduces address and
    # register pressure.  Measured wisdom can override either tie-break.
    def vg_tiebreak(candidate: Candidate) -> int:
        prefer_wide = (
            (candidate.radix <= 32 and candidate.pattern.value in {
                "direct", "like_terms", "even_odd"
            }) or
            (candidate.radix == 64 and
             candidate.pattern.value == "like_terms")
        )
        return (-candidate.matrix_vg_width if prefer_wide
                else candidate.matrix_vg_width)

    # Rotation only uses temporary matrix tiles proven available by candidate
    # construction. If the lower bound ties, prefer its shorter dependency
    # chain; empirical wisdom can still select the single-tile control.
    return sorted(ranked, key=lambda item: (item[1].total_cost,
                                             item[0].pattern.value,
                                             vg_tiebreak(item[0]),
                                             not item[0].rotate_temp_tiles,
                                             item[0].batch_pipeline_depth))
