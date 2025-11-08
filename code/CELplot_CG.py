# =================== CG-CEL: run + one-figure plot (3 lines, per-n tqdm progress) ===================
import math, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

# -------------------- 1) Core methods --------------------
def cel_allocation(claims, estate):
    """CEL allocation using exact water-filling algorithm"""
    c = np.asarray(claims, dtype=np.float64)
    D = float(c.sum())
    if estate <= 0:
        return np.zeros_like(c)
    if estate >= D:
        return c.copy()

    idx = np.argsort(-c)              # descending
    cs  = c[idx]
    ps  = np.cumsum(cs)
    n   = len(cs)

    lam = None
    for k in range(1, n + 1):
        lam_k = (ps[k - 1] - estate) / k
        if cs[k - 1] > lam_k and (k == n or lam_k >= cs[k]):
            lam = lam_k; break
    if lam is None:
        lam = (ps[-1] - estate) / n

    as_ = np.maximum(0.0, cs - lam)
    a   = np.zeros_like(as_)
    a[idx] = as_
    s = a.sum()
    if s > 0 and abs(s - estate) > 1e-12:
        a *= (estate / s)
    return a

def truthful_matrix(vec):
    vec = np.asarray(vec, dtype=np.float64)
    return np.tile(vec, (len(vec), 1))

def awards_from_matrix(R, estate=35, scale=100):
    return cg_cel_allocation(R.mean(axis=0) * scale, estate)

# -------- CG-CEL: Conditional Grant version of CEL --------
def cg_cel_allocation(claims, estate):
    """
    CG-CEL: First run CEL on half the estate, then run CEL on the rest with remaining claims.
    """
    c = np.asarray(claims, dtype=np.float64)

    # Step 1: Run CEL on D/2
    half_estate = estate / 2.0
    first_allocation = cel_allocation(c, half_estate)

    # Step 2: Compute remaining claims
    remaining_claims = c - first_allocation
    remaining_claims = np.maximum(remaining_claims, 0)

    # Step 3: Run CEL on remaining estate (D/2) with remaining claims
    second_allocation = cel_allocation(remaining_claims, half_estate)

    # Total allocation
    return first_allocation + second_allocation

# -------- 保持原版的 impartial_division（零改动） --------
def impartial_division(R, eps=1e-9):
    R, n = np.asarray(R, float), R.shape[0]
    phi = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
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
            if i == j:
                continue
            f[i] = 1.0 / (1.0 + phi[j, i] + sum(psi(j, k, i) for k in range(n) if k not in (i, j)))
        f[j] = 1.0 - f.sum()
        total += f
    shares = total / n
    return shares / shares.sum()

# -------------------- 组合生成器（等价但更省内存） --------------------
def compositions_nonneg_iter(total, parts, out_arr=None):
    """
    生成所有非负整数组合 x[0..parts-1] 使 sum(x)=total。
    与 stars-and-bars 完全同一集合（数量 C(total+parts-1, parts-1)）。
    """
    if parts <= 0:
        return
    if out_arr is None:
        x = np.zeros(parts, dtype=np.int32)
    else:
        x = out_arr
        x.fill(0)

    def rec(pos, rem):
        if pos == parts - 1:
            x[pos] = rem
            yield x
        else:
            for v in range(rem + 1):
                x[pos] = v
                yield from rec(pos + 1, rem - v)
    yield from rec(0, total)

def n_compositions(free_units, k):
    return math.comb(free_units + k - 1, k - 1)

# -------------------- 2) 两种操纵搜索（CG-CEL version） --------------------
def all_opt_rows_restricted(R0, m, estate=20, scale=100, tol=1e-9, alpha=0.01, pbar=None):
    n = R0.shape[0]
    fixed  = float(R0[m, m])
    others = [i for i in range(n) if i != m]
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        if pbar: pbar.close()
            # 无可行候选
        return float('-inf'), [], []

    free_units = scale - min_units * k
    best_val, best_rows, best_shares = -1.0, [], []

    # 预分配与复用
    row = np.zeros(n, dtype=np.float64)
    row[m] = fixed
    R = R0.copy()
    buf = np.zeros(k, dtype=np.int32)
    rem = 1.0 - fixed
    inv_scale = 1.0 / scale

    for add_units in compositions_nonneg_iter(free_units, k, out_arr=buf):
        if pbar: pbar.update(1)

        # row[others] = ((min_units + add_units)/scale) * rem
        row.fill(0.0); row[m] = fixed
        row[others] = (add_units + min_units) * inv_scale * rem

        R[m, :] = row
        shares = impartial_division(R)                          # ← 保持原版 ID
        val    = cg_cel_allocation(shares * scale, estate)[m]   # ← 用 CG-CEL 计算 payoff

        if val > best_val + tol:
            best_val = val
            best_rows   = [np.round(row, 3).tolist()]
            best_shares = [np.round(shares, 3).tolist()]
        elif abs(val - best_val) < tol:
            best_rows.append(np.round(row, 3).tolist())
            best_shares.append(np.round(shares, 3).tolist())

    if pbar: pbar.close()
    if best_val < 0:
        return float('-inf'), [], []
    return float(best_val), best_rows, best_shares

def all_opt_rows_unbounded(R0, m, estate=20, scale=100, tol=1e-9, alpha=0.01, pbar=None):
    n = R0.shape[0]
    fixed  = float(R0[m, m])
    others = [i for i in range(n) if i != m]
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        if pbar: pbar.close()
        return float('-inf'), []

    free_units = scale - min_units * k
    best_val, best_rows = -1.0, []

    row = np.zeros(n, dtype=np.float64)
    row[m] = fixed
    buf = np.zeros(k, dtype=np.int32)
    rem = 1.0 - fixed
    inv_scale = 1.0 / scale

    for add_units in compositions_nonneg_iter(free_units, k, out_arr=buf):
        if pbar: pbar.update(1)

        row.fill(0.0); row[m] = fixed
        row[others] = (add_units + min_units) * inv_scale * rem

        val = cg_cel_allocation(row * scale, estate)[m]         # ← 用 CG-CEL 计算 payoff

        if val > best_val + tol:
            best_val, best_rows = val, [np.round(row, 3).tolist()]
        elif abs(val - best_val) < tol:
            best_rows.append(np.round(row, 3).tolist())

    if pbar: pbar.close()
    if best_val < 0:
        return float('-inf'), []
    return float(best_val), best_rows

# -------------------- 3) Test case: CG-CEL setting (D-E, E-(n-1)*0.01, 0.01, ..., 0.01) --------------------
E_FIXED   = 60
SCALE     = 100
ALPHA     = 0.01
# CG-CEL setting: (D-E, E-(n-1)*0.01, 0.01, ..., 0.01)
# After allocating D/2 and halving: ((D-E)/2, (E-(n-1)*0.01)/2, 0.005, ..., 0.005) with E' = E - D/2

BASE_NS    = [4, 5, 6, 7]
MUTATE_NS  = []

_ns_set = set(BASE_NS)
for d in MUTATE_NS:
    if d > 0: _ns_set.add(d)
    elif d < 0: _ns_set.discard(abs(d))
NS_TRACK = sorted(_ns_set)

ns, orig_line, rest_line, unbd_line = [], [], [], []

print(f"Tracking ns = {NS_TRACK}")
print("n | manip awards (Original / Restricted / Unrestricted)")
print("-"*78)

for n in NS_TRACK:
    # CG-CEL setting: (D-E, E-(n-1)*0.01, 0.01, ..., 0.01)
    # First agent (manipulator): D - E
    # Second agent: E - (n-1)*0.01
    # Others: 0.01 each
    d_manip = SCALE - E_FIXED
    d_second = E_FIXED - (n - 1) * 0.01
    claims = [d_manip, d_second] + [0.01] * (n - 2)

    m = 0  # manipulator is first agent
    vec = np.array(claims, dtype=np.float64) / SCALE
    R_truth = truthful_matrix(vec)

    awards_orig = awards_from_matrix(R_truth, estate=E_FIXED, scale=SCALE)
    manip_orig  = float(awards_orig[m])

    k = n - 1
    min_units = int(round(ALPHA * SCALE))
    free_units = max(0, SCALE - min_units * k)
    total_candidates = n_compositions(free_units, k)

    pbar_r = tqdm(total=total_candidates, desc=f"[n={n}] restricted", unit="cand", leave=True)
    rest_best, _, _ = all_opt_rows_restricted(
        R_truth, m, estate=E_FIXED, scale=SCALE, alpha=ALPHA, pbar=pbar_r
    )

    pbar_u = tqdm(total=total_candidates, desc=f"[n={n}] unbounded", unit="cand", leave=True)
    unbd_best, _ = all_opt_rows_unbounded(
        R_truth, m, estate=E_FIXED, scale=SCALE, alpha=ALPHA, pbar=pbar_u
    )

    ns.append(n)
    orig_line.append(manip_orig)
    rest_line.append(float(rest_best))
    unbd_line.append(float(unbd_best))

    print(f"{n} | {manip_orig:.6f} / {float(rest_best):.6f} / {float(unbd_best):.6f}")
    print(f"    Difference (Restricted - Unrestricted): {float(rest_best) - float(unbd_best):.10f}")
    print(f"    Unrestricted >= Restricted? {float(unbd_best) >= float(rest_best) - 1e-6}")

# -------------------- 4) Plot --------------------
plt.figure(figsize=(6.6, 4.2))
plt.plot(ns, orig_line, marker="o",  linestyle="-",  linewidth=2, label="Original")
plt.plot(ns, rest_line, marker="s",  linestyle="--", linewidth=2, label="Restricted")
plt.plot(ns, unbd_line, marker="^",  linestyle=":",  linewidth=2, label="Unrestricted")
plt.xlabel("Number of agents (n)")
plt.ylabel(f"Manipulator payoff (CG-CEL, E={E_FIXED})")
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
plt.title("CG-CEL – manipulator payoff vs n  (setting: [D-E, E-(n-1)*0.01, 0.01, ...])")
plt.legend()
plt.tight_layout()

ns_tag = "-".join(map(str, ns))
out_name = f"cg_cel_E{int(E_FIXED)}_ns{ns_tag}.png"
plt.savefig(out_name, dpi=240)
plt.close()
print(f"✅ Saved: {out_name}")
