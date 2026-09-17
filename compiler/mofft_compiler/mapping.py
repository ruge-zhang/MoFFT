"""Target-neutral matrix mappings for rewritten FFT expressions."""

from __future__ import annotations

from enum import Enum
from typing import Sequence


class ComplexMatrixLayout(str, Enum):
    """How complex values are represented in a matrix-engine mapping."""

    SPLIT = "split-complex"
    INTERLEAVED = "interleaved-complex"


class MatrixPrimitive(str, Enum):
    """Reduction direction exposed by the target matrix engine."""

    OUTER_PRODUCT = "outer-product"
    INNER_PRODUCT = "inner-product"


class MatrixMapping(str, Enum):
    SPLIT_COMPLEX_OUTER = "split-complex-outer-product"
    INTERLEAVED_COMPLEX_OUTER = "interleaved-complex-outer-product"
    COMPLEX_INNER = "complex-inner-product"

    @property
    def primitive(self) -> MatrixPrimitive:
        if self == MatrixMapping.COMPLEX_INNER:
            return MatrixPrimitive.INNER_PRODUCT
        return MatrixPrimitive.OUTER_PRODUCT

    @property
    def complex_layout(self) -> ComplexMatrixLayout:
        if self == MatrixMapping.INTERLEAVED_COMPLEX_OUTER:
            return ComplexMatrixLayout.INTERLEAVED
        return ComplexMatrixLayout.SPLIT


def evaluate_outer_product_mapping(
        coefficients: Sequence[Sequence[complex]],
        inputs: Sequence[Sequence[complex]],
        mapping: MatrixMapping) -> list[list[complex]]:
    """Evaluate the numerical contract of an outer-product mapping.

    ``inputs[j][b]`` is complex batch ``b`` for input row ``j`` and
    ``coefficients[k][j]`` maps input row ``j`` to output row ``k``.  This
    executable contract is intentionally independent of vector width and of
    any target intrinsic spelling.  Backends can therefore validate a packing
    or lowering against the same semantics before it is admitted to tuning.
    """
    if mapping.primitive != MatrixPrimitive.OUTER_PRODUCT:
        raise ValueError(f"{mapping.value} is not an outer-product mapping")
    output_rows = len(coefficients)
    input_rows = len(inputs)
    if output_rows == 0 or input_rows == 0:
        return []
    batch = len(inputs[0])
    if any(len(row) != input_rows for row in coefficients):
        raise ValueError("coefficient matrix has inconsistent input width")
    if any(len(row) != batch for row in inputs):
        raise ValueError("input rows have inconsistent batch width")

    result = [[0j for _ in range(batch)] for _ in range(output_rows)]
    if mapping.complex_layout == ComplexMatrixLayout.SPLIT:
        for k, row in enumerate(coefficients):
            for j, weight in enumerate(row):
                wr, wi = weight.real, weight.imag
                for b, value in enumerate(inputs[j]):
                    result[k][b] += complex(
                        wr * value.real - wi * value.imag,
                        wr * value.imag + wi * value.real)
        return result

    # An interleaved column vector is [real_0, imag_0, real_1, imag_1, ...].
    # Multiplication by a complex coefficient is two real outer products:
    #   wr * x + wi * rotate90(x), rotate90([r, i]) == [-i, r].
    for k, row in enumerate(coefficients):
        for j, weight in enumerate(row):
            wr, wi = weight.real, weight.imag
            packed = [component for value in inputs[j]
                      for component in (value.real, value.imag)]
            rotated = [component for value in inputs[j]
                       for component in (-value.imag, value.real)]
            accumulated = [wr * value + wi * turn
                           for value, turn in zip(packed, rotated)]
            for b in range(batch):
                result[k][b] += complex(accumulated[2 * b],
                                        accumulated[2 * b + 1])
    return result
