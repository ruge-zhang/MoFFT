from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InstructionModel:
    throughput: float
    latency: float
    saturation_ilp: int
    resource: str
    inferred: bool = True


@dataclass(frozen=True)
class MemoryLevel:
    capacity_bytes: int
    read_bandwidth_bytes_per_cost_unit: float
    write_bandwidth_bytes_per_cost_unit: float
    latency_cost: float
    inferred: bool = True
    mixed_bandwidth_bytes_per_cost_unit: float | None = None

    @property
    def effective_mixed_bandwidth(self) -> float:
        if self.mixed_bandwidth_bytes_per_cost_unit is not None:
            return self.mixed_bandwidth_bytes_per_cost_unit
        read = max(self.read_bandwidth_bytes_per_cost_unit, 1e-9)
        write = max(self.write_bandwidth_bytes_per_cost_unit, 1e-9)
        return 2.0 / (1.0 / read + 1.0 / write)


@dataclass(frozen=True)
class MemoryHierarchy:
    cache_line_bytes: int
    levels: dict[str, MemoryLevel]

    def level_for(self, working_set_bytes: int) -> tuple[str, MemoryLevel]:
        ordered = sorted(self.levels.items(),
                         key=lambda item: item[1].capacity_bytes)
        for name, level in ordered:
            if working_set_bytes <= level.capacity_bytes:
                return name, level
        return ordered[-1]


@dataclass(frozen=True)
class LayoutCase:
    operation: str
    working_set_bytes: int
    columns: int
    batch: int
    relative_bandwidth: float
    inferred: bool = False


def _default_memory_hierarchy() -> MemoryHierarchy:
    # Conservative fallback for legacy profiles. Values are explicitly
    # inferred and are replaced by mofft-calibrate memory sweeps.
    return MemoryHierarchy(128, {
        "l1d": MemoryLevel(64 * 1024, 64.0, 48.0, 4.0, True),
        "l2": MemoryLevel(4 * 1024 * 1024, 32.0, 24.0, 16.0, True),
        "dram": MemoryLevel(1 << 62, 12.0, 10.0, 100.0, True),
    })


@dataclass(frozen=True)
class MachineProfile:
    schema_version: int
    name: str
    architecture: str
    streaming_vector_bits: int
    instructions: dict[str, InstructionModel]
    source: str
    measured: bool
    measurement_metadata: dict[str, Any]
    memory_hierarchy: MemoryHierarchy
    layout_cases: tuple[LayoutCase, ...] = ()
    cost_unit: str = "normalized-time"

    @classmethod
    def load(cls, path: str | Path) -> "MachineProfile":
        raw: dict[str, Any] = json.loads(Path(path).read_text())
        instructions = {
            name: InstructionModel(**values)
            for name, values in raw["instructions"].items()
        }
        memory_raw = raw.get("memory_hierarchy")
        if memory_raw is None:
            memory_hierarchy = _default_memory_hierarchy()
        else:
            levels = {}
            for name, values in memory_raw["levels"].items():
                normalized = dict(values)
                for kind in ("read", "write", "mixed"):
                    current = f"{kind}_bandwidth_bytes_per_cost_unit"
                    legacy = f"{kind}_bandwidth_bytes_per_cycle"
                    if current not in normalized and legacy in normalized:
                        normalized[current] = normalized.pop(legacy)
                if "latency_cost" not in normalized:
                    normalized["latency_cost"] = normalized.pop(
                        "latency_cycles")
                levels[name] = MemoryLevel(**normalized)
            memory_hierarchy = MemoryHierarchy(
                cache_line_bytes=int(memory_raw["cache_line_bytes"]),
                levels=levels)
        layout_cases = tuple(
            LayoutCase(**values)
            for values in raw.get("layout_model", {}).get("cases", []))
        metadata = raw.get("measurement_metadata", {})
        cost_unit = raw.get("cost_unit")
        if cost_unit is None:
            throughput_unit = str(metadata.get("throughput_unit", ""))
            cost_unit = ("nanoseconds" if "/ns" in throughput_unit
                         else "normalized-time")
        return cls(
            schema_version=raw["schema_version"],
            name=raw["name"],
            architecture=raw["architecture"],
            streaming_vector_bits=raw["streaming_vector_bits"],
            instructions=instructions,
            source=raw["source"],
            measured=raw["measured"],
            measurement_metadata=metadata,
            memory_hierarchy=memory_hierarchy,
            layout_cases=layout_cases,
            cost_unit=cost_unit,
        )

    def lanes(self, precision: str) -> int:
        bits = 32 if precision == "fp32" else 64
        return self.streaming_vector_bits // bits

    def to_json(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "name": self.name,
            "architecture": self.architecture,
            "streaming_vector_bits": self.streaming_vector_bits,
            "instructions": {
                key: {
                    "throughput": value.throughput,
                    "latency": value.latency,
                    "saturation_ilp": value.saturation_ilp,
                    "resource": value.resource,
                    "inferred": value.inferred,
                }
                for key, value in sorted(self.instructions.items())
            },
            "source": self.source,
            "measured": self.measured,
            "cost_unit": self.cost_unit,
            "memory_hierarchy": {
                "cache_line_bytes": self.memory_hierarchy.cache_line_bytes,
                "levels": {
                    name: {
                        "capacity_bytes": level.capacity_bytes,
                        "read_bandwidth_bytes_per_cost_unit":
                            level.read_bandwidth_bytes_per_cost_unit,
                        "write_bandwidth_bytes_per_cost_unit":
                            level.write_bandwidth_bytes_per_cost_unit,
                        "mixed_bandwidth_bytes_per_cost_unit":
                            level.mixed_bandwidth_bytes_per_cost_unit,
                        "latency_cost": level.latency_cost,
                        "inferred": level.inferred,
                    }
                    for name, level in sorted(
                        self.memory_hierarchy.levels.items(),
                        key=lambda item: item[1].capacity_bytes)
                },
            },
        }
        if self.measurement_metadata:
            result["measurement_metadata"] = self.measurement_metadata
        if self.layout_cases:
            result["layout_model"] = {
                "bandwidth_basis": "relative-to-sequential-copy",
                "cases": [
                    {
                        "operation": case.operation,
                        "working_set_bytes": case.working_set_bytes,
                        "columns": case.columns,
                        "batch": case.batch,
                        "relative_bandwidth": case.relative_bandwidth,
                        "inferred": case.inferred,
                    }
                    for case in self.layout_cases
                ],
            }
        return result
