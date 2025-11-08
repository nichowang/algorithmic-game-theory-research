# ================================================================
#   Quick predictor – how many 0.01 floors in the optimal row?
#   * n agents = 4
#   * Estate  E = 40
#   * Floor value = 0.01
# ================================================================

import numpy as np
import itertools, math

def cea_allocation(claims, estate, tol=1e-12, iters=60):
    c = np.asarray(claims, float)
    lo, hi = 0.0, c.max()
    for _ in range(iters):
        lam = (lo + hi) / 2
        if np.minimum(c, lam).sum() > estate:
            hi = lam
        else:
            lo = lam
    return np.minimum(c, hi)
def impartial_division(R, eps=1e-9):
    R = np.asarray(R, float); n = R.shape[0]

    # φ_{ab}
    phi = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b: continue
            ratios = [R[l,a] / (R[l,b] + eps)
                      for l in range(n) if l not in (a, b) and R[l,b] > eps]
            phi[a,b] = (sum(ratios) / len(ratios)) if ratios else 1.0

    # helper ψ_{res,k,i}
    def psi(res, k, i):
        reps = [l for l in range(n) if l not in (res, k, i) and R[l,i] > eps]
        return (sum(R[l,k] / (R[l,i] + eps) for l in reps) / len(reps)) if reps else 1.0

    total = np.zeros(n)
    for j in range(n):
        f = np.zeros(n)
        for i in range(n):
            if i == j: continue
            f[i] = 1.0 / (1.0 + phi[j,i] +
                          sum(psi(j, k, i) for k in range(n) if k not in (i, j)))
        f[j] = 1.0 - f.sum()
        total += f

    shares = total / n
    return shares / shares.sum()

# ---------- Helpers ----------
def truthful_matrix(sv):            # sv: 1×n share vector
    return np.tile(sv, (len(sv), 1))

FLOOR        = 0.01
CLAIM_SCALE  = 100.0
ESTATE       = 20.0
TOL          = 1e-9                 # comparison tolerance

def award_if_row(sv, manip, row):
    """Given a hypothetical row, compute manipulator's CEA award at E=40."""
    R = truthful_matrix(sv); R[manip] = row
    shares  = impartial_division(R)
    awards  = cea_allocation(shares * CLAIM_SCALE, ESTATE)
    return awards[manip]

def predict_n_floor(sv, manip, floor=FLOOR):
    """
    Return (k, best_award):
      k ∈ {1,2}  – optimal row contains how many entries equal to floor (=0.01)
    Strategy space reduced to:
      * double‑floor: [self, f, f, 1‑self‑2f]  (who gets the big chunk varies)
      * single‑floor: [self, f, α, β] (one floor, rest proportional to truthful)
    """
    sv = np.asarray(sv, float); n = len(sv)
    others = [i for i in range(n) if i != manip]
    best_aw, best_k = -1.0, None
    # --- double‑floor (2×0.01) ---
    for big in others:
        row = np.full(n, floor)
        row[manip] = sv[manip]
        remain = 1.0 - row.sum()
        if remain <= 0:      # should not happen, guard anyway
            continue
        row[big] += remain

        aw = award_if_row(sv, manip, row)
        if aw > best_aw + TOL:
            best_aw, best_k = aw, 2
    # --- single‑floor (1×0.01) ---
    for floored in others:
        keepers = [i for i in others if i != floored]

        row = np.zeros(n)
        row[manip]   = sv[manip]
        row[floored] = floor

        remain = 1.0 - row[manip] - row[floored]
        weights = np.array([sv[i] for i in keepers])
        row[keepers] = remain / len(keepers) if weights.sum() == 0 \
                       else remain * weights / weights.sum()

        aw = award_if_row(sv, manip, row)
        if aw > best_aw + TOL:
            best_aw, best_k = aw, 1

    return best_k, best_aw

# ---------- Your test vectors ----------
vectors = [
    [0.08, 0.19, 0.07, 0.66],
    [0.48, 0.40, 0.05, 0.07],
    [0.22, 0.24, 0.09, 0.45],
    [0.35, 0.05, 0.24, 0.36],
    [0.15, 0.19, 0.13, 0.53],
    [0.21, 0.23, 0.08, 0.48],
    [0.10, 0.19, 0.09, 0.62],
    [0.18, 0.16, 0.09, 0.57],
    [0.22, 0.20, 0.14, 0.44],
    [0.35, 0.10, 0.24, 0.31],
    [0.10, 0.13, 0.09, 0.68],
    [0.28, 0.42, 0.25, 0.05],
    [0.18, 0.16, 0.19, 0.47],
    [0.25, 0.20, 0.11, 0.44],
    [0.35, 0.04, 0.30, 0.31],
    [0.15, 0.19, 0.06, 0.60],
]

# ---------- Run test (all 4 manipulators per vector) ----------
print("Scenario (rounded) | Manip | #floor=0.01 | Best award\n" + "-"*60)
for sv in vectors:
    sv_round = np.round(sv, 2).tolist()
    for m in range(4):
        k, aw = predict_n_floor(sv, manip=m)
        print(f"{sv_round} |  {m}    |    {k}         |  {aw:.6f}")
