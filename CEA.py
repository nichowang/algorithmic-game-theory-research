import itertools, json, math, numpy as np, pandas as pd
from tqdm import tqdm


def cea_allocation(claims, estate, tol=1e-12, iters=60):
    c = np.asarray(claims, float)
    lo, hi = 0.0, c.max()
    for _ in range(iters):
        lam = (lo + hi) / 2
        hi, lo = (lam, lo) if np.minimum(c, lam).sum() > estate else (hi, lam)
    return np.minimum(c, hi)


def truthful_matrix(vec): return np.tile(vec, (len(vec), 1))

def awards_from_matrix(R, estate=35, scale=100):
    return cea_allocation(R.mean(0) * scale, estate)

def impartial_division(R, eps=1e-9):
    R, n = np.asarray(R, float), R.shape[0]
    phi = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b: continue
            ratios = [R[l, a] / (R[l, b] + eps)
                      for l in range(n) if l not in (a, b) and R[l, b] > eps]
            phi[a, b] = sum(ratios)/len(ratios) if ratios else 1.0
    def psi(res, k, i):
        reps = [l for l in range(n) if l not in (res, k, i) and R[l, i] > eps]
        return sum(R[l, k]/(R[l, i]+eps) for l in reps)/len(reps) if reps else 1.0
    total = np.zeros(n)
    for j in range(n):
        f = np.zeros(n)
        for i in range(n):
            if i == j: continue
            f[i] = 1.0 / (1.0 + phi[j, i] +
                          sum(psi(j, k, i) for k in range(n) if k not in (i, j)))
        f[j] = 1.0 - f.sum()
        total += f
    shares = total / n
    return shares / shares.sum()


def all_opt_rows_restricted(R0, m, estate=35, scale=100, tol=1e-9):
    n = R0.shape[0]
    fixed  = float(R0[m, m])
    remain = int(round((1 - fixed) * scale))
    others = [i for i in range(n) if i != m]

    best_val, best_rows, best_shares = -1.0, [], []
    for cuts in itertools.combinations(range(1, remain), len(others) - 1):
        parts = np.diff((0, *cuts, remain))
        row   = np.zeros(n); row[m] = fixed
        for idx, p in zip(others, parts):
            row[idx] = p / scale
        R = R0.copy(); R[m] = row

        shares = impartial_division(R)
        val    = cea_allocation(shares * scale, estate)[m]

        if val > best_val + tol:
            best_val, best_rows, best_shares = val, \
                 [np.round(row, 3).tolist()], [np.round(shares, 3).tolist()]
        elif abs(val - best_val) < tol:
            best_rows.append(np.round(row, 3).tolist())
            best_shares.append(np.round(shares, 3).tolist())

    return round(best_val, 3), best_rows, best_shares

def all_opt_rows_unbounded(R0, m, estate=35, scale=100, tol=1e-9):
    n = R0.shape[0]
    fixed  = float(R0[0, m])
    remain = int(round((1 - fixed) * scale))
    others = [i for i in range(n) if i != m]

    best_val, best_rows = -1.0, []
    for cuts in itertools.combinations(range(1, remain), len(others) - 1):
        parts = np.diff((0, *cuts, remain))
        row   = np.zeros(n); row[m] = fixed
        for idx, p in zip(others, parts):
            row[idx] = p / scale
        val = cea_allocation(row * scale, estate)[m]

        if val > best_val + tol:
            best_val, best_rows = val, [np.round(row, 3).tolist()]
        elif abs(val - best_val) < tol:
            best_rows.append(np.round(row, 3).tolist())

    return round(best_val, 3), best_rows

# ---------------- 4.  scenarios --------------------- #
share_vectors = [
    np.array([0.08, 0.19, 0.07, 0.66]),
    np.array([0.48, 0.40, 0.05, 0.07]),
    np.array([0.22, 0.24, 0.09, 0.45]),
    np.array([0.35, 0.05, 0.24, 0.36]),
    np.array([0.15, 0.19, 0.13, 0.53]),
    np.array([0.21, 0.23, 0.08, 0.48]),
    np.array([0.10, 0.19, 0.09, 0.62]),
    np.array([0.18, 0.16, 0.09, 0.57]),
    np.array([0.22, 0.20, 0.14, 0.44]),
    np.array([0.35, 0.10, 0.24, 0.31]),
    np.array([0.10, 0.13, 0.09, 0.68]),
    np.array([0.28, 0.42, 0.25, 0.05]),
    np.array([0.18, 0.16, 0.19, 0.47]),
    np.array([0.25, 0.20, 0.11, 0.44]),
    np.array([0.35, 0.04, 0.30, 0.31]),
    np.array([0.15, 0.19, 0.06, 0.60]),
]

records = []
E = 35   

for sv in tqdm(share_vectors, desc='CEA scenarios'):
    R_truth = truthful_matrix(sv)
    orig_aw = awards_from_matrix(R_truth, estate=E)
    for m in range(len(sv)):
        rest_best, rest_rows, rest_sh = all_opt_rows_restricted(R_truth, m, estate=E)
        unb_best,  unb_rows           = all_opt_rows_unbounded(R_truth, m, estate=E)

        records.append({
            "Scenario":         np.round(sv, 3).tolist(),
            "Manipulator":      m,
            "Orig_Best":        round(orig_aw[m], 3),
            "Rest_Best":        rest_best,
            "Unbd_Best":        unb_best,
            "Optimal_Rows":     json.dumps(rest_rows),
            "Unbd_Rows":        json.dumps(unb_rows),
            "Impartial_Shares": json.dumps(rest_sh)
        })

pd.DataFrame(records).to_excel("CEA_detailed_awards_summar1y.xlsx", index=False)
print("✅ CEA_detailed_awards_summary1.xlsx")




