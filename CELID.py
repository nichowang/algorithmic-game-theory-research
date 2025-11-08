# =============  CEL manipulation – save ALL optimal rows (5 agents)  =============
import itertools, json, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm


def cel_allocation(claims, estate, tol=1e-12):
    c = np.asarray(claims, float)
    lo, hi = 0.0, c.max()
    for _ in range(60):
        lam  = (lo + hi) / 2
        total = np.maximum(0.0, c - lam).sum()
        lo, hi = (lam, hi) if total > estate else (lo, lam)
    lam     = hi
    awards  = np.maximum(0.0, c - lam)
    factor  = estate / awards.sum()
    if factor < 1 + tol:
        awards *= factor
    return np.minimum(awards, c + tol)

def truthful_matrix(sv):
    return np.tile(sv, (len(sv), 1))

def awards_from_matrix(R, estate=70, claim_scale=100):
    return cel_allocation(R.mean(0) * claim_scale, estate)

def impartial_division(R, eps=1e-9):
    R, n = np.asarray(R, float), R.shape[0]
    phi = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            ratios = [R[l, a] / (R[l, b] + eps)
                      for l in range(n) if l not in (a, b) and R[l, b] > eps]
            phi[a, b] = sum(ratios) / len(ratios) if ratios else 1.0

    def psi(res, k, i):
        reps = [l for l in range(n) if l not in (res, k, i) and R[l, i] > eps]
        return sum(R[l, k] / (R[l, i] + eps) for l in reps) / len(reps) if reps else 1.0

    total = np.zeros(n)
    for j in range(n):
        f = np.zeros(n)
        for i in range(n):
            if i == j:
                continue
            term1 = phi[j, i]
            term2 = sum(psi(j, k, i) for k in range(n) if k not in (i, j))
            f[i]  = 1.0 / (1.0 + term1 + term2)
        f[j]      = 1.0 - f.sum()
        total    += f

    shares = total / n
    shares /= shares.sum()
    return shares

def all_opt_rows(R0, manip, estate=70, claim_scale=100, tol=1e-9, max_keep=None):
    n      = R0.shape[0]
    fixed  = float(R0[manip, manip])
    remain = int(round((1 - fixed) * claim_scale))
    others = [i for i in range(n) if i != manip]
    k      = len(others) - 1

    best_val  = -1.0
    best_rows = []

    combos = itertools.combinations(range(1, remain), k)
    total  = math.comb(remain - 1, k)
    for cuts in tqdm(combos, total=total, leave=False, desc=f"search row{manip}"):
        parts = np.diff((0, *cuts, remain))
        row   = np.zeros(n); row[manip] = fixed
        for idx, p in zip(others, parts):
            row[idx] = p / claim_scale

        R   = R0.copy(); R[manip] = row
        val = cel_allocation(impartial_division(R) * claim_scale, estate)[manip]

        if val > best_val + tol:
            best_val  = val
            best_rows = [row.tolist()]
        elif abs(val - best_val) < tol:
            best_rows.append(row.tolist())

        if max_keep and len(best_rows) >= max_keep:
            break
    return round(best_val, 6), best_rows


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

records, chart_rows = [], []

for sv in tqdm(share_vectors, desc="Scenarios"):
    R_truth = truthful_matrix(sv)
    orig_aw = awards_from_matrix(R_truth)
    n_agents = len(sv)

    for manip in tqdm(range(n_agents), desc="Manipulators", leave=False):
        best_aw, rows = all_opt_rows(R_truth, manip)

        records.append({
            "Scenario": sv.round(2).tolist(),
            "Manipulator": manip,
            "Best_Award": best_aw,
            "Num_OptimalRows": len(rows),
            "Optimal_Rows": json.dumps(rows)
        })

        chart_rows.append({
            "Scenario": f"{sv.round(2)} | ag{manip}",
            "Original": orig_aw[manip],
            "Best": best_aw
        })

df_opt = pd.DataFrame(records)
df_opt.to_excel("CEL_5agents_AllOptimalRows.xlsx", index=False)
print("✅ saved CEL_5agents_AllOptimalRows.xlsx")

df_chart = pd.DataFrame(chart_rows)
x = np.arange(len(df_chart)); w = 0.30
plt.figure(figsize=(10, 6))
plt.bar(x - w/2, df_chart["Original"], w, label="Original truthful award")
plt.bar(x + w/2, df_chart["Best"],     w, label="Restricted best")
plt.xticks(x, df_chart["Scenario"], rotation=45, ha="right")
plt.ylabel("CEL award (E = 70)")
plt.title("CEL – truthful vs best restricted (5 agents)")
plt.legend(); plt.tight_layout(); plt.show()