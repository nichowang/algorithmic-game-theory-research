import numpy as np
import itertools


FLOOR        = 0.01
CLAIM_SCALE  = 100.0
ESTATE       = 40.0         
TOL          = 1e-9

# ===== CEA allocator =====
def cea_allocation(claims, estate, tol=1e-12, iters=60):
    c = np.asarray(claims, float)
    lo, hi = 0.0, c.max()
    for _ in range(iters):
        lam = (lo + hi) / 2.0
        if np.minimum(c, lam).sum() > estate:
            hi = lam
        else:
            lo = lam
    return np.minimum(c, hi)


def impartial_division(R, eps=1e-9):
    R = np.asarray(R, float); n = R.shape[0]
    phi = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b: continue
            ratios = [R[l,a] / (R[l,b] + eps)
                      for l in range(n) if l not in (a,b) and R[l,b] > eps]
            phi[a,b] = (sum(ratios) / len(ratios)) if ratios else 1.0

    def psi(res, k, i):
        reps = [l for l in range(n) if l not in (res,k,i) and R[l,i] > eps]
        return (sum(R[l,k]/(R[l,i]+eps) for l in reps) / len(reps)) if reps else 1.0

    total = np.zeros(n)
    for j in range(n):
        f = np.zeros(n)
        for i in range(n):
            if i == j: continue
            f[i] = 1.0 / (1.0 + phi[j,i] +
                          sum(psi(j,k,i) for k in range(n) if k not in (i,j)))
        f[j] = 1.0 - f.sum()
        total += f
    shares = total / n
    return shares / shares.sum()


def truthful_matrix(sv):        
    return np.tile(sv, (len(sv), 1))

def award_if_row(sv, manip, row):

    R = truthful_matrix(sv); R[manip] = row
    shares = impartial_division(R)
    awards = cea_allocation(shares * CLAIM_SCALE, ESTATE)
    return awards[manip]

def predict_floor_general(sv, manip, floor=FLOOR):

    sv = np.asarray(sv, float); n = len(sv)
    others = [i for i in range(n) if i != manip]

    best_aw, best_k = -1.0, None

    # k = 1 … n‑2
    for k in range(1, n-1):
        for floored_subset in itertools.combinations(others, k):
            floored_subset = set(floored_subset)
            keepers = [i for i in others if i not in floored_subset]

            row = np.zeros(n)
            row[manip] = sv[manip]
            row[list(floored_subset)] = floor

            remain = 1.0 - row[manip] - k*floor
            if remain < -TOL:
                continue     
            weights = np.array([sv[i] for i in keepers])
            row[keepers] = remain / len(keepers) if weights.sum() == 0 \
                           else remain * weights / weights.sum()

            aw = award_if_row(sv, manip, row)
            if aw > best_aw + TOL:
                best_aw, best_k = aw, k

    return best_k, best_aw
sv5 = [0.15, 0.2, 0.05, 0.24, 0.36]  
for m in range(5):
    k, aw = predict_floor_general(sv5, manip=m)
    print(f"Manipulator {m}: optimal row sets {k} values to 0.01, award = {aw:.4f}")

