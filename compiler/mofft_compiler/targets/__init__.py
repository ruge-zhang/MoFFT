"""Target capability descriptions and backend-specific lowerings."""

from .arm_sme import arm_sme_target
from .base import TargetCapabilities

__all__ = ["TargetCapabilities", "arm_sme_target"]
