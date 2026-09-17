"""Backend interface between scheduled matrix IR and source emission."""

from __future__ import annotations

from typing import Protocol

from ..ir import Node
from ..patterns import Candidate
from ..profile import MachineProfile
from ..targets.base import TargetCapabilities


class BackendTypes(Protocol):
    ctype: str
    complex: str
    bits: str
    short: str
    vector: str
    pair: str
    quad: str
    cnt: str


class MatrixBackend(Protocol):
    name: str

    def target(self, profile: MachineProfile) -> TargetCapabilities: ...

    def types(self, precision: str) -> BackendTypes: ...

    def value_name(self, node_id: int, suffix: str = "") -> str: ...

    def pair_name(self, node_id: int, suffix: str = "") -> str: ...

    def emit_node(
            self, node: Node, candidate: Candidate, table_name: str,
            group: int, address_override: str | None = None,
            twiddle_broadcast: bool = False, suffix: str = "",
            column_predicate: str = "pg_cols",
            base_name: str = "base") -> list[str]: ...

    def emit_bundle(
            self, kind: str, width: int, members: tuple[int, ...],
            candidate: Candidate, table_name: str, group: int,
            suffix: str = "") -> list[str]: ...

    def emit_finish_rows(
            self, candidate: Candidate, block: int, real_tile: int,
            imag_tile: int, half_offset: int, suffix: str = "",
            column_predicate: str = "pg_cols",
            base_name: str = "base") -> list[str]: ...

    def emit_vertical_epilogue(
            self, candidate: Candidate, group: int) -> str: ...
