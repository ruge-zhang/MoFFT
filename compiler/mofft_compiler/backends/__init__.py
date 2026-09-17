"""Instruction-set backends for lowering the neutral matrix IR."""

from .arm_sme import ARM_SME_BACKEND
from .base import MatrixBackend


def backend_for_architecture(architecture: str) -> MatrixBackend:
    normalized = architecture.lower()
    if (normalized.startswith(("arm64", "aarch64")) and
            "sme" in normalized):
        return ARM_SME_BACKEND
    raise ValueError(f"no matrix backend for architecture {architecture!r}")


__all__ = ["ARM_SME_BACKEND", "MatrixBackend", "backend_for_architecture"]
