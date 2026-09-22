"""
Simulation package for the Perpetual Inventory Challenge.
Models the hidden state of retail inventory, unobserved error mechanisms,
censored POS sales observations, and historical audit policies.
"""

from .generator import InventorySimulator
from .error_mechanisms import ErrorMechanismEngine
from .censoring import apply_pos_censoring
from .audit_policy import LegacyABCAuditPolicy

__all__ = [
    "InventorySimulator",
    "ErrorMechanismEngine",
    "apply_pos_censoring",
    "LegacyABCAuditPolicy",
]
