# =================== CEL: manipulator payoff vs n (use YOUR helpers, n=4,5) ===================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

# ---------- 1) === paste your helpers here (完全使用你给的定义，不做改动) ----------
import itertools, json, math, numpy as np, pandas as pd
from tqdm import tqdm

def cel_allocation(claims, estate, tol=1e-12):
    c = np.asarray(claims, float)
    lo, hi = 0.0, c.max()
    for _ in range(60):
        lam = (lo + hi) / 2
        total = np.maximum(0.0, c - lam).sum()
        lo, hi = (lam, hi) if total > estate else (lo, lam)
    lam = hi
    a = np.maximum(0, c - lam)
    a *= estate / a.sum()
    return a

def truthful_matrix(v): return np.tile(v, (len(v), 1))

def awards_from_matrix(R, estate=70, scale=100):
    return cel_allocation(R.mean(0) * scale, estate)

def impartial_division(R, eps=1e-9):
    R, n = np.asarray(R, float), R.shape[0]
    phi  = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b: continue
            ratios = [R[l,a]/(R[l,b]+eps)
                      for l in range(n) if l not in (a,b) and R[l,b] > eps]
            phi[a,b] = sum(ratios)/len(ratios) if ratios else 1.0
    def psi(res,k,i):
        reps = [l for l in range(n) if l not in (res,k,i) and R[l,i] > eps]
        return sum(R[l,k]/(R[l,i]+eps) for l in reps)/len(reps) if reps else 1.0
    total = np.zeros(n)
    for j in range(n):
        f = np.zeros(n)
        for i in range(n):
            if i == j: continue
            f[i] = 1.0 / (1.0 + phi[j,i] +
                          sum(psi(j,k,i) for k in range(n) if k not in (i,j)))
        f[j] = 1.0 - f.sum()
        total += f
    return (total / n) / (total / n).sum()

def all_opt_rows_restricted(R0, m, estate=70, scale=100, tol=1e-9):
    n       = R0.shape[0]
    fixed   = float(R0[m, m])
    remain  = int(round((1-fixed)*scale))
    others  = [i for i in range(n) if i != m]

    best_val, best_rows, best_shares = -1.0, [], []
    for cuts in itertools.combinations(range(1, remain), len(others)-1):
        parts = np.diff((0, *cuts, remain))
        row   = np.zeros(n); row[m] = fixed
        for idx, part in zip(others, parts):
            row[idx] = part/scale

        R      = R0.copy();  R[m] = row
        shares = impartial_division(R)
        val    = cel_allocation(shares*scale, estate)[m]

        if val > best_val + tol:
            best_val      = val
            best_rows     = [np.round(row, 3).tolist()]
            best_shares   = [np.round(shares, 3).tolist()]
        elif abs(val - best_val) < tol:
            best_rows.append(np.round(row, 3).tolist())
            best_shares.append(np.round(shares, 3).tolist())

    return round(best_val, 3), best_rows, best_shares

def all_opt_rows_unbounded(R0, m, estate=70, scale=100, tol=1e-9):
    n       = R0.shape[0]
    fixed   = float(R0[0, m])        # 保持与你给的一致
    remain  = int(round((1-fixed)*scale))
    others  = [i for i in range(n) if i != m]

    best_val, best_rows = -1.0, []
    for cuts in itertools.combinations(range(1, remain), len(others)-1):
        parts = np.diff((0, *cuts, remain))
        row   = np.zeros(n);  row[m] = fixed
        for idx, part in zip(others, parts):
            row[idx] = part/scale
        val = cel_allocation(row*scale, estate)[m]

        if val > best_val + tol:
            best_val, best_rows = val, [np.round(row, 3).tolist()]
        elif abs(val - best_val) < tol:
            best_rows.append(np.round(row, 3).tolist())

    return round(best_val, 3), best_rows

# ---------- 2) 实验参数 ----------
E_FIXED   = 70        # Estate
D_BIG     = 60.0      # 大申报（操作者的声明）
SCALE     = 100
NS        = [4, 5]    # 只跑 n=4,5；你可以改成 [4,5,6,7]

# ---------- 3) 主流程 ----------
ns, orig_line, rest_line, unbd_line = [], [], [], []

print("n | Original / Restricted / Unrestricted")
print("-"*50)

for n in tqdm(NS, desc="n-sweep"):
    # claims: [E/n,...,E/n, D]
    claims = [E_FIXED / n] * (n - 1) + [D_BIG]
    vec = np.array(claims, float) / SCALE   # 与你的脚本一致：把 claims 缩到 share 尺度
    R_truth = truthful_matrix(vec)
    m = n - 1                                # 把最后一个作为操作者

    # Original
    awards_orig = awards_from_matrix(R_truth, estate=E_FIXED, scale=SCALE)
    manip_orig  = float(awards_orig[m])

    # Restricted / Unrestricted（与你脚本一致的搜索）
    rest_best, _rest_rows, _rest_sh = all_opt_rows_restricted(
        R_truth, m, estate=E_FIXED, scale=SCALE
    )
    unbd_best, _unbd_rows = all_opt_rows_unbounded(
        R_truth, m, estate=E_FIXED, scale=SCALE
    )

    ns.append(n)
    orig_line.append(manip_orig)
    rest_line.append(float(rest_best))
    unbd_line.append(float(unbd_best))

    print(f"{n} | {manip_orig:.6f} / {float(rest_best):.6f} / {float(unbd_best):.6f}")

# ---------- 4) 画图 ----------
import matplotlib.pyplot as plt
plt.figure(figsize=(6.6, 4.2))
plt.plot(ns, orig_line, marker="o",  linestyle="-",  linewidth=2, label="Original")
plt.plot(ns, rest_line, marker="s",  linestyle="--", linewidth=2, label="Restricted")
plt.plot(ns, unbd_line, marker="^",  linestyle=":",  linewidth=2, label="Unrestricted")
plt.xlabel("Number of agents (n)")
plt.ylabel(f"Manipulator payoff (CEL, E={E_FIXED})")
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
plt.title("CEL – manipulator payoff vs n  (test case: [E/n,...,E/n, D_BIG])")
plt.legend()
plt.tight_layout()
plt.show()
