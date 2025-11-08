# =================== CG CEL: run + one-figure plot (3 lines, per-n tqdm progress) [FAST w/ ID cache, memory-safe] ===================
import math, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

# -------------------- 1) CEL (exact water-filling; same results as binary search) --------------------
def cel_allocation(claims, estate):
    c = np.asarray(claims, dtype=np.float64)
    D = float(c.sum())
    if estate <= 0:
        return np.zeros_like(c)
    if estate >= D:
        return c.copy()

    idx = np.argsort(-c)        # descending
    cs  = c[idx]
    ps  = np.cumsum(cs)
    n   = len(cs)

    lam = None
    for k in range(1, n + 1):
        lam_k = (ps[k - 1] - estate) / k
        if cs[k - 1] > lam_k and (k == n or lam_k >= cs[k]):
            lam = lam_k; break
    if lam is None:                         # numeric guard
        lam = (ps[-1] - estate) / n

    as_ = np.maximum(0.0, cs - lam)
    a   = np.zeros_like(as_)
    a[idx] = as_
    s = a.sum()
    if s > 0 and abs(s - estate) > 1e-12:  # tiny renorm (same result as before)
        a *= (estate / s)
    return a

def cg_allocation(claims, estate):
    """
    Contested Garment rule: uses CEA if E <= D/2, otherwise uses CEL.
    For CEL: each agent gets half their claim first, then CEL on remaining.
    """
    c = np.asarray(claims, dtype=np.float64)
    D = float(c.sum())

    if estate >= D:
        return c.copy()

    # CG rule: if E <= D/2, use CEA on half-claims; otherwise use CEL
    if estate <= D / 2:
        # CEA regime
        half_claims = c / 2
        sorted_idx = np.argsort(half_claims)
        allocation = np.zeros_like(c)
        remaining = estate

        for i in range(len(c)):
            n_remaining = len(c) - i
            if n_remaining == 0:
                break
            equal_share = remaining / n_remaining
            idx = sorted_idx[i]
            allocation[idx] = min(half_claims[idx], equal_share)
            remaining -= allocation[idx]

        return allocation
    else:
        # CEL regime: each gets half their claim first, then run CEL on the rest
        half_claims = c / 2
        remaining_estate = estate - D / 2
        cel_result = cel_allocation(half_claims, remaining_estate)
        return half_claims + cel_result
