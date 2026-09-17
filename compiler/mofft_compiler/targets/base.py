"""Instruction-set capability contract used by matrix mapping and costing."""

from __future__ import annotations

from dataclasses import dataclass

from ..mapping import MatrixMapping, MatrixPrimitive


@dataclass(frozen=True)
class TargetCapabilities:
    name: str
    vector_bits: int
    vector_registers: int
    accumulator_tiles_fp32: int
    accumulator_tiles_fp64: int
    matrix_group_widths: tuple[int, ...]
    mappings: tuple[MatrixMapping, ...]
    instruction_model_keys: tuple[tuple[str, str], ...] = ()

    def lanes(self, precision: str) -> int:
        if precision == "fp32":
            bits = 32
        elif precision == "fp64":
            bits = 64
        else:
            raise ValueError(f"unsupported precision: {precision}")
        if self.vector_bits % bits:
            raise ValueError(
                f"{self.name} vector width does not support {precision}")
        return self.vector_bits // bits

    def accumulator_tiles(self, precision: str) -> int:
        if precision == "fp32":
            return self.accumulator_tiles_fp32
        if precision == "fp64":
            return self.accumulator_tiles_fp64
        raise ValueError(f"unsupported precision: {precision}")

    def supports(self, mapping: MatrixMapping) -> bool:
        return mapping in self.mappings

    def supports_primitive(self, primitive: MatrixPrimitive) -> bool:
        return any(mapping.primitive == primitive for mapping in self.mappings)

    def instruction_model_key(
            self, operation: str,
            matrix_group_width: int | None = None) -> str | None:
        template = dict(self.instruction_model_keys).get(operation)
        if template is None:
            return None
        if "{matrix_group_width}" in template:
            if matrix_group_width is None:
                raise ValueError(
                    f"{operation} requires a matrix group width")
            return template.format(matrix_group_width=matrix_group_width)
        return template
