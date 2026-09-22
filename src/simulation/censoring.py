"""
Censoring Module: Simulates Point-of-Sale (POS) transaction recording under physical availability constraints.

Crux:
Demand D(i,t) is latent and unconstrained.
Observed POS Sales S(i,t) is constrained by physical on-shelf availability:
    S(i,t) = min(D(i,t), AvailableOnShelf(i,t))
    LostSales(i,t) = max(0, D(i,t) - AvailableOnShelf(i,t))
"""

from typing import Tuple
import numpy as np


def apply_pos_censoring(
    latent_demand: np.ndarray,
    true_on_hand: np.ndarray,
    misplaced_units: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Censors latent demand based on physically accessible inventory.

    Parameters
    ----------
    latent_demand : np.ndarray
        Integer array of true customer purchase intents.
    true_on_hand : np.ndarray
        Integer array of actual physical inventory units in store.
    misplaced_units : np.ndarray, optional
        Units physically in the building but absent from designated shelf.

    Returns
    -------
    pos_sales : np.ndarray
        Observed POS scan sales.
    lost_sales : np.ndarray
        True unserved demand due to stockouts.
    """
    if misplaced_units is None:
        misplaced_units = np.zeros_like(true_on_hand)

    # Accessible on-shelf stock is physical stock minus misplaced units
    accessible_stock = np.maximum(0, true_on_hand - misplaced_units)
    
    # POS sales cannot exceed what was accessible to shoppers
    pos_sales = np.minimum(latent_demand, accessible_stock)
    lost_sales = np.maximum(0, latent_demand - accessible_stock)

    return pos_sales, lost_sales
