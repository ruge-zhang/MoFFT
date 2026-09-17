from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from typing import Iterable


class ValueKind(str, Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    VECTOR_PAIR = "vector_pair"
    VECTOR_GROUP4 = "vector_group4"
    MATRIX = "matrix"
    PREDICATE = "predicate"
    MEMORY = "memory"


class Op(str, Enum):
    LOAD = "load"
    LOAD2 = "load2"
    STORE = "store"
    STORE2 = "store2"
    GET = "get"
    PACK = "pack"
    BROADCAST = "broadcast"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    NEG = "neg"
    COEFF_RECONSTRUCT = "coeff_reconstruct"
    OUTER_PRODUCT_ACCUM = "outer_product_accumulate"
    FMLA_VECTOR = "fmla_vector"
    MATRIX_FMA = "matrix_fma"
    MATRIX_EXTRACT_H = "matrix_extract_horizontal"
    MATRIX_EXTRACT_V = "matrix_extract_vertical"
    MATRIX_ALLOC = "matrix_alloc"
    MATRIX_CLEAR = "matrix_clear"


@dataclass(frozen=True)
class Node:
    id: int
    op: Op
    inputs: tuple[int, ...]
    kind: ValueKind
    precision: str
    predicate: str | None = None
    attrs: tuple[tuple[str, str], ...] = ()

    def attr(self, name: str, default: str | None = None) -> str | None:
        return dict(self.attrs).get(name, default)


@dataclass
class Graph:
    precision: str
    nodes: list[Node] = field(default_factory=list)
    _cse: dict[tuple, int] = field(default_factory=dict, init=False, repr=False)

    def add(self, op: Op, inputs: Iterable[int] = (), *, kind: ValueKind,
            predicate: str | None = None, **attrs: object) -> int:
        input_tuple = tuple(inputs)
        attr_tuple = tuple(sorted((key, str(value)) for key, value in attrs.items()))
        key = (op.value, input_tuple, kind.value, self.precision, predicate, attr_tuple)
        if op not in (Op.STORE, Op.STORE2, Op.MATRIX_ALLOC, Op.MATRIX_CLEAR, Op.OUTER_PRODUCT_ACCUM,
                      Op.MATRIX_FMA) and key in self._cse:
            return self._cse[key]
        node_id = len(self.nodes)
        node = Node(node_id, op, input_tuple, kind, self.precision,
                    predicate, attr_tuple)
        self.nodes.append(node)
        if op not in (Op.STORE, Op.STORE2, Op.MATRIX_ALLOC, Op.MATRIX_CLEAR, Op.OUTER_PRODUCT_ACCUM,
                      Op.MATRIX_FMA):
            self._cse[key] = node_id
        return node_id

    def digest(self) -> str:
        serial = [
            {
                "id": n.id, "op": n.op.value, "inputs": n.inputs,
                "kind": n.kind.value, "precision": n.precision,
                "predicate": n.predicate, "attrs": n.attrs,
            }
            for n in self.nodes
        ]
        return sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()

    def op_counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for node in self.nodes:
            result[node.op.value] = result.get(node.op.value, 0) + 1
        return result

    def validate(self) -> None:
        for node in self.nodes:
            for dependency in node.inputs:
                if dependency < 0 or dependency >= node.id:
                    raise ValueError(
                        f"node {node.id} has non-DAG dependency {dependency}")
