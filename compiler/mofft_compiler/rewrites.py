"""Target-independent FFT algebra and rewrite definitions.

This module deliberately knows nothing about ZA tiles, vector-group widths,
or a concrete instruction set.  A matrix mapping/backend may apply stricter
profitability and feasibility filters after these algebraic preconditions.
"""

from __future__ import annotations

import cmath
from enum import Enum
import math


class Pattern(str, Enum):
    DIRECT = "direct"
    LIKE_TERMS = "like_terms"
    EVEN_ODD = "even_odd"
    LIKE_TERMS_EVEN_ODD = "like_terms_even_odd"
    VECTOR_REUSE = "vector_reuse"
    LIKE_TERMS_VECTOR_REUSE = "like_terms_vector_reuse"


def algebraic_patterns(radix: int) -> tuple[Pattern, ...]:
    """Return rewrites that are mathematically defined for ``radix``.

    This is intentionally broader than a backend's candidate set.  For
    example, Vector-Reuse is valid for every radix divisible by four even when
    a particular target is too narrow for the mapping to be profitable.
    """
    if radix < 2:
        raise ValueError(f"unsupported radix: {radix}")
    result = [Pattern.DIRECT, Pattern.LIKE_TERMS]
    if radix % 4 == 0:
        result.extend((Pattern.VECTOR_REUSE,
                       Pattern.LIKE_TERMS_VECTOR_REUSE))
    if radix % 2 == 0:
        result.extend((Pattern.EVEN_ODD,
                       Pattern.LIKE_TERMS_EVEN_ODD))
    return tuple(result)


def evaluate_pattern(values: list[complex], direction: str,
                     pattern: Pattern) -> list[complex]:
    """Evaluate one algebraic rewrite without a target-specific lowering."""
    radix = len(values)
    if pattern not in algebraic_patterns(radix):
        raise ValueError(f"{pattern.value} is not defined for radix {radix}")
    if direction not in ("forward", "backward"):
        raise ValueError(f"unknown direction: {direction}")
    sign = -1.0 if direction == "forward" else 1.0

    def coefficient(j: int, k: int) -> complex:
        if pattern not in (Pattern.VECTOR_REUSE,
                           Pattern.LIKE_TERMS_VECTOR_REUSE):
            return cmath.exp(sign * 2j * math.pi * j * k / radix)
        quarter = radix // 4
        source = k % quarter
        quadrant = k // quarter
        raw = cmath.exp(sign * 2j * math.pi * j * source / radix)
        phase = cmath.exp(sign * .5j * math.pi * j * quadrant)
        return raw * phase

    def paired_row(k: int, parity: int | None = None) -> complex:
        total = values[0] if parity in (None, 0) else 0j
        for j in range(1, (radix + 1) // 2):
            if parity is not None and j % 2 != parity:
                continue
            w = coefficient(j, k)
            total += values[j] * w + values[radix - j] * w.conjugate()
        if radix % 2 == 0 and parity in (None, (radix // 2) % 2):
            total += values[radix // 2] * coefficient(radix // 2, k)
        return total

    if pattern in (Pattern.EVEN_ODD, Pattern.LIKE_TERMS_EVEN_ODD):
        low: list[complex] = []
        high: list[complex] = []
        for k in range(radix // 2):
            if pattern == Pattern.LIKE_TERMS_EVEN_ODD:
                even = paired_row(k, 0)
                odd = paired_row(k, 1)
            else:
                even = sum(values[j] * coefficient(j, k)
                           for j in range(0, radix, 2))
                odd = sum(values[j] * coefficient(j, k)
                          for j in range(1, radix, 2))
            low.append(even + odd)
            high.append(even - odd)
        return low + high
    if pattern in (Pattern.LIKE_TERMS,
                   Pattern.LIKE_TERMS_VECTOR_REUSE):
        return [paired_row(k) for k in range(radix)]
    return [sum(value * coefficient(j, k)
                for j, value in enumerate(values))
            for k in range(radix)]
