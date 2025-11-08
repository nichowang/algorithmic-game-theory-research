import itertools, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------ 1. CEA allocator ------------------ #
def cea_allocation(claims, estate, tol=1e-12, iters=60):
    c = np.asarray(claims, float)
    lo, hi = 0.0, c.max()
    for _ in range(iters):
        lam = (lo + hi) / 2
        hi, lo = (lam, lo) if np.minimum(c, lam).sum() > estate else (hi, lam)
    return np.minimum(c, hi)

# ------------------ 2. Helper builders ---------------- #
def truthful_matrix(share_vec):
    return np.tile(share_vec, (len(share_vec), 1))

def awards_from_matrix(R, estate=35.0, claim_scale=100.0):
    claims = R.mean(axis=0) * claim_scale
    return cea_allocation(claims, estate)

def impartial_division(R, eps=1e-9):
    R, n = np.asarray(R, float), R.shape[0]
    phi = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b: continue
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
            if i == j: continue
            term1 = phi[j, i]
            term2 = sum(psi(j, k, i) for k in range(n) if k not in (i, j))
            f[i] = 1.0 / (1.0 + term1 + term2)
        f[j] = 1.0 - f.sum()
        total += f
    shares = total / n
    return shares / shares.sum()

# ------------------ 3. Manipulation routines ---------- #
def restricted_outcome(R0, manip, estate=35, claim_scale=100):
    n = R0.shape[0]
    fixed  = float(R0[manip, manip])
    remain = int(round((1 - fixed) * claim_scale))
    others = [i for i in range(n) if i != manip]
    best_aw, best_val, best_mat = None, -1.0, None

    total_combo = math.comb(remain - 1, len(others) - 1)
    for cuts in tqdm(itertools.combinations(range(1, remain), len(others) - 1),
                     total=total_combo, leave=False,
                     desc=f"restricted | ag{manip}"):
        parts = np.diff((0, *cuts, remain))
        row = np.zeros(n); row[manip] = fixed
        for idx, p in zip(others, parts):
            row[idx] = p / claim_scale
        R = R0.copy(); R[manip] = row
        aw = cea_allocation(impartial_division(R) * claim_scale, estate)
        if aw[manip] > best_val + 1e-9:
            best_val, best_aw, best_mat = aw[manip], aw, R.copy()
    return best_aw, best_mat

def unbounded_outcome(R0, manip, estate=35, claim_scale=100):
    n      = R0.shape[0]
    row0   = 0
    fixed  = float(R0[row0, manip])
    remain = int(round((1 - fixed) * claim_scale))
    others = [i for i in range(n) if i != manip]

    best_val, best_awards, best_row = -1.0, None, None
    total_combo = math.comb(remain - 1, len(others) - 1)
    for cuts in tqdm(itertools.combinations(range(1, remain), len(others) - 1),
                     total=total_combo, leave=False,
                     desc=f"unbounded | ag{manip}"):
        parts = np.diff((0, *cuts, remain))
        row = np.zeros(n)
        row[manip] = fixed
        for idx, p in zip(others, parts):
            row[idx] = p / claim_scale
        awards = cea_allocation(row * claim_scale, estate)
        if awards[manip] > best_val + 1e-9:
            best_val, best_awards, best_row = awards[manip], awards, row.copy()

    best_matrix = np.tile(best_row, (n, 1))
    return best_awards, best_matrix

# ------------------ 4. Scenarios ---------------------- #
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

tables, chart_rows, matrix_rows = [], [], []

for sv in tqdm(share_vectors, desc="Scenarios"):
    R_truth = truthful_matrix(sv)
    orig_aw = awards_from_matrix(R_truth)
    for manip in tqdm(range(len(sv)), desc="Manipulators", leave=False):
        rest_aw, rest_mat = restricted_outcome(R_truth, manip)
        unb_aw,  unb_mat  = unbounded_outcome(R_truth, manip)

        tables.append({
            "Scenario": sv.round(2).tolist(), "Manipulator": manip,
            **{f"Orig_Ag{i}": orig_aw[i] for i in range(4)},
            **{f"Rest_Ag{i}": rest_aw[i] for i in range(4)},
            **{f"Unbd_Ag{i}": unb_aw[i]  for i in range(4)},
        })
        chart_rows.append({
            "Scenario": f"{sv.round(2)} | agent {manip}",
            "Original":   orig_aw[manip],
            "Restricted": rest_aw[manip],
            "Unbounded":  unb_aw[manip],
        })
        matrix_rows.append({
            "Scenario": sv.round(2).tolist(),
            "Manipulator": manip,
            "Restricted_Matrix": np.round(rest_mat, 4).tolist(),
            "Unbounded_Matrix":  np.round(unb_mat, 4).tolist(),
        })

# ------------------ 5. Results & plots ---------------- #
df_chart    = pd.DataFrame(chart_rows)
df_full     = pd.DataFrame(tables)
df_matrices = pd.DataFrame(matrix_rows)
df_matrices["Restricted_Matrix"] = df_matrices["Restricted_Matrix"].apply(str)
df_matrices["Unbounded_Matrix"]  = df_matrices["Unbounded_Matrix"].apply(str)

print("\nSummary table (manipulator awards only):")
print(df_chart.to_string(index=False,
                         float_format=lambda x: f"{x:6.2f}"))

x = np.arange(len(df_chart)); w = 0.25
plt.figure(figsize=(12, 6))
plt.bar(x - w, df_chart["Original"],   w, label="Original")
plt.bar(x,      df_chart["Restricted"], w, label="Restricted")
plt.bar(x + w, df_chart["Unbounded"],  w, label="Unbounded")
plt.xticks(x, df_chart["Scenario"], rotation=45, ha="right")
plt.ylabel("CEA Award to Manipulator (E = 40)")
plt.title("Manipulation Power under CEA (single manipulator)")
plt.legend(); plt.tight_layout(); plt.show()

print("\nFull CEA outcomes (all agents, all manipulators):\n")
print(df_full.to_string(index=False,
                        float_format=lambda x: f"{x:8.4f}"))
