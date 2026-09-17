"""ARM SME matrix-engine capabilities.

ACLE spelling and instruction emission remain in the SME emitter while it is
being split into a backend package.  This module is the single capability
source used by target-independent candidate construction.
"""

from __future__ import annotations

from ..mapping import MatrixMapping
from .base import TargetCapabilities


def arm_sme_target(vector_bits: int = 512) -> TargetCapabilities:
    return TargetCapabilities(
        name="arm-sme",
        vector_bits=vector_bits,
        vector_registers=32,
        accumulator_tiles_fp32=4,
        accumulator_tiles_fp64=8,
        matrix_group_widths=(2, 4),
        # Interleaved-complex outer products have a target-neutral numerical
        # contract, but no validated SME packing/lowering yet. Do not advertise
        # a mapping until the backend can emit and differentially test it.
        mappings=(MatrixMapping.SPLIT_COMPLEX_OUTER,),
        instruction_model_keys=(
            ("outer_product_accumulate", "fmopa"),
            ("add", "sve_add"), ("sub", "sve_add"),
            ("mul", "sve_fmla"), ("neg", "sve_add"),
            ("fmla_vector", "sve_fmla"),
            ("matrix_fma", "sme2_fmla_vg{matrix_group_width}"),
            ("matrix_extract_horizontal", "za_extract"),
            ("matrix_extract_group", "za_extract_vg{matrix_group_width}"),
            ("matrix_extract_vertical", "za_extract"),
            ("load", "sve_load"), ("load2", "sve_load"),
            ("store", "sve_store"), ("store2", "sve_store"),
        ),
    )


ARM_SME_512 = arm_sme_target()
