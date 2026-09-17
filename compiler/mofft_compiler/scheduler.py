from __future__ import annotations

from dataclasses import dataclass

from .ir import Graph, Node, Op, ValueKind


@dataclass(frozen=True)
class ScheduledNode:
    node: Node
    round: int
    vector_register: int | None
    matrix_tile: int | None
    spill: bool
    fusion: int


def schedule(graph: Graph, vector_limit: int = 32,
             matrix_tile_limit: int | None = None) -> list[ScheduledNode]:
    """Pressure-aware outer-product scheduling with stable tie breaks."""
    graph.validate()
    matrix_tile_limit = matrix_tile_limit or (
        4 if graph.precision == "fp32" else 8)
    users: dict[int, list[int]] = {node.id: [] for node in graph.nodes}
    indegree = {node.id: len(node.inputs) for node in graph.nodes}
    for node in graph.nodes:
        for dependency in node.inputs:
            users[dependency].append(node.id)
    remaining_uses = {node_id: len(node_users)
                      for node_id, node_users in users.items()}

    # For each producer, remember which accumulator chains it can unblock. This
    # lets the scheduler expose a different tile before extending the same
    # accumulator chain, which determines the outer-product latency floor.
    target_tiles: dict[int, set[int]] = {node.id: set() for node in graph.nodes}
    distance_to_outer_product: dict[int, int] = {
        node.id: len(graph.nodes) + 1 for node in graph.nodes}
    first_outer_product: dict[int, int] = {
        node.id: len(graph.nodes) + 1 for node in graph.nodes}
    for node in reversed(graph.nodes):
        if node.op == Op.OUTER_PRODUCT_ACCUM:
            target_tiles[node.id].add(int(node.attr("tile", "0")))
            distance_to_outer_product[node.id] = 0
            first_outer_product[node.id] = node.id
        for user in users[node.id]:
            target_tiles[node.id].update(target_tiles[user])
            distance_to_outer_product[node.id] = min(
                distance_to_outer_product[node.id],
                distance_to_outer_product[user] + 1)
            first_outer_product[node.id] = min(
                first_outer_product[node.id], first_outer_product[user])

    ready = {node.id for node in graph.nodes if indegree[node.id] == 0}
    live_vectors: dict[int, tuple[int, ...]] = {}
    free_vectors = set(range(vector_limit))
    outer_product_round: dict[int, int] = {}
    output: list[ScheduledNode] = []
    last_tile: int | None = None

    op_priority = {
        Op.MATRIX_ALLOC: -2, Op.MATRIX_CLEAR: -1, Op.BROADCAST: -1,
        Op.LOAD2: 0, Op.GET: 1, Op.PACK: 1, Op.LOAD: 2,
        Op.ADD: 3, Op.SUB: 3, Op.MUL: 3, Op.NEG: 3,
        Op.COEFF_RECONSTRUCT: 3,
        Op.FMLA_VECTOR: 4, Op.MATRIX_FMA: 4,
        Op.MATRIX_EXTRACT_H: 6, Op.MATRIX_EXTRACT_V: 6,
        Op.STORE: 7, Op.STORE2: 7,
    }

    def score(node_id: int) -> tuple[int, int, int, int, int]:
        node = graph.nodes[node_id]
        if node.op == Op.MATRIX_ALLOC:
            return (-3, 0, 0, 0, node_id)
        if node.op == Op.OUTER_PRODUCT_ACCUM:
            tile = int(node.attr("tile", "0"))
            return (0 if last_tile is None or tile != last_tile else 2,
                    node_id,
                    outer_product_round.get(node.inputs[0], -1) + 1, 0,
                    node_id)
        exposes_other = (last_tile is not None and
                         any(tile != last_tile for tile in target_tiles[node_id]))
        return (1 if exposes_other else 3, first_outer_product[node_id],
                distance_to_outer_product[node_id],
                op_priority.get(node.op, 5),
                node_id)

    while ready:
        node_id = min(ready, key=score)
        ready.remove(node_id)
        node = graph.nodes[node_id]
        # Inputs whose final use is this instruction can be coalesced with the
        # destination tuple. Release them before assigning the result, as a
        # physical SSA allocator would.
        for dependency in node.inputs:
            remaining_uses[dependency] -= 1
            if remaining_uses[dependency] == 0 and dependency in live_vectors:
                free_vectors.update(live_vectors.pop(dependency))
        register = None
        spill = False
        matrix_tile = None
        if node.kind in (ValueKind.VECTOR, ValueKind.VECTOR_PAIR,
                         ValueKind.VECTOR_GROUP4):
            width = (4 if node.kind == ValueKind.VECTOR_GROUP4 else
                     (2 if node.kind == ValueKind.VECTOR_PAIR else 1))
            # LOAD2 and matrix tuple reads are containers whose GET nodes
            # name the actual component registers. Counting both the tuple
            # and every GET would double-count the same architectural Z regs.
            if node.op in (Op.LOAD2, Op.MATRIX_EXTRACT_H):
                width = 0
            # This is a pressure model, not the final physical allocator.
            # ACLE tuple intrinsics impose contiguity at instruction selection;
            # globally renumberable live values must not be called spills merely
            # because this greedy walk has fragmented its temporary numbering.
            if width == 0:
                pass
            elif len(free_vectors) < width:
                spill = True
            else:
                allocated = tuple(sorted(free_vectors)[:width])
                register = allocated[0]
                free_vectors.difference_update(allocated)
                live_vectors[node_id] = allocated
        elif node.kind == ValueKind.MATRIX:
            requested = node.attr("tile")
            if requested is not None:
                matrix_tile = int(requested)
                if matrix_tile >= matrix_tile_limit:
                    raise ValueError(
                        f"matrix tile {matrix_tile} exceeds limit "
                        f"{matrix_tile_limit}")

        round_number = 0
        if node.op == Op.OUTER_PRODUCT_ACCUM:
            round_number = outer_product_round.get(node.inputs[0], -1) + 1
            outer_product_round[node.id] = round_number
            last_tile = matrix_tile
        output.append(ScheduledNode(node, round_number, register, matrix_tile,
                                    spill, 1))

        for user in users[node_id]:
            indegree[user] -= 1
            if indegree[user] == 0:
                ready.add(user)

    if len(output) != len(graph.nodes):
        raise ValueError("DAG contains a cycle or a missing dependency")

    # Fusion is deliberately not inferred from schedule adjacency. The
    # emitter forms concrete groups only after proving address or matrix-tile
    # slice contiguity, predicate compatibility and full-vector width.
    return output
