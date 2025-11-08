# =================== CEA: run + one-figure plot (3 lines, per-n tqdm progress) ===================
import math, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
from functools import lru_cache
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

# -------------------- 1) Core methods --------------------
def cea_allocation(claims, estate, iters=60):
    c = np.asarray(claims, dtype=np.float64)
    lo, hi = 0.0, float(c.max())
    if c.sum() <= estate:
        return c.copy()
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        if np.minimum(c, lam).sum() > estate:
            hi = lam
        else:
            lo = lam
    return np.minimum(c, hi)

def truthful_matrix(vec):
    vec = np.asarray(vec, dtype=np.float64)
    return np.tile(vec, (len(vec), 1))

def awards_from_matrix(R, estate=35, scale=100):
    return cea_allocation(R.mean(axis=0) * scale, estate)

# -------- Optimized impartial_division with numba --------
@njit(cache=True)
def _impartial_division_core(R, eps=1e-9):
    """Numba-optimized core computation"""
    n = R.shape[0]
    phi = np.ones((n, n), dtype=np.float64)

    # Compute phi
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            s = 0.0
            cnt = 0
            for l in range(n):
                if l == a or l == b:
                    continue
                if R[l, b] > eps:
                    s += R[l, a] / (R[l, b] + eps)
                    cnt += 1
            if cnt > 0:
                phi[a, b] = s / cnt

    # Compute shares
    total = np.zeros(n, dtype=np.float64)
    for j in range(n):
        f = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == j:
                continue
            s = 1.0 + phi[j, i]
            # Add psi terms
            for k in range(n):
                if k == i or k == j:
                    continue
                psi_s = 0.0
                psi_cnt = 0
                for l in range(n):
                    if l == j or l == k or l == i:
                        continue
                    if R[l, i] > eps:
                        psi_s += R[l, k] / (R[l, i] + eps)
                        psi_cnt += 1
                if psi_cnt > 0:
                    s += psi_s / psi_cnt
                else:
                    s += 1.0
            f[i] = 1.0 / s
        f[j] = 1.0 - f.sum()
        total += f

    shares = total / n
    s = shares.sum()
    if s > 0:
        shares = shares / s
    return shares

@lru_cache(maxsize=8192)
def impartial_division_cached(R_tuple, eps=1e-9):
    """Cached version - takes tuple instead of array"""
    n = int(np.sqrt(len(R_tuple)))
    R = np.array(R_tuple, dtype=np.float64).reshape(n, n)
    return tuple(_impartial_division_core(R, eps))

def impartial_division(R, eps=1e-9):
    """Wrapper that converts array to tuple for caching"""
    return np.array(impartial_division_cached(tuple(R.flatten()), eps))

# -------------------- 组合生成器（等价但更省内存） --------------------
def compositions_nonneg_iter(total, parts, out_arr=None):
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

# -------------------- 2) 两种操纵搜索 --------------------
def all_opt_rows_restricted(R0, m, estate=20, scale=100, alpha=0.01, pbar=None):
    n = R0.shape[0]
    fixed = float(R0[m, m])
    others = np.array([i for i in range(n) if i != m])
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        if pbar: pbar.close()
        return float('-inf'), [], []

    free_units = scale - min_units * k
    best_val = -1.0
    rem = 1.0 - fixed
    inv_scale = 1.0 / scale

    row = np.empty(n, dtype=np.float64)
    R = R0.copy()
    buf = np.zeros(k, dtype=np.int32)

    for add_units in compositions_nonneg_iter(free_units, k, out_arr=buf):
        row[m] = fixed
        row[others] = (add_units + min_units) * inv_scale * rem
        R[m, :] = row

        shares = impartial_division(R)
        val = cea_allocation(shares * scale, estate)[m]

        if pbar: pbar.update(1)
        if val > best_val:
            best_val = val

    if pbar: pbar.close()
    return float(best_val), [], []

def all_opt_rows_unbounded(R0, m, estate=20, scale=100, alpha=0.01, pbar=None):
    n = R0.shape[0]
    fixed = float(R0[m, m])
    others = np.array([i for i in range(n) if i != m])
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        if pbar: pbar.close()
        return float('-inf'), []

    free_units = scale - min_units * k
    best_val = -1.0
    rem = 1.0 - fixed
    inv_scale = 1.0 / scale

    row = np.empty(n, dtype=np.float64)
    buf = np.zeros(k, dtype=np.int32)

    for add_units in compositions_nonneg_iter(free_units, k, out_arr=buf):
        row[m] = fixed
        row[others] = (add_units + min_units) * inv_scale * rem

        val = cea_allocation(row * scale, estate)[m]

        if pbar: pbar.update(1)
        if val > best_val:
            best_val = val

    if pbar: pbar.close()
    return float(best_val), []

# -------------------- 3) Test case --------------------
E_FIXED = 20
SCALE = 100
ALPHA = 0.01
D_BIG = 60.0

BASE_NS = [4, 5, 6, 7]
MUTATE_NS = []

_ns_set = set(BASE_NS)
for d in MUTATE_NS:
    if d > 0:
        _ns_set.add(d)
    elif d < 0:
        _ns_set.discard(abs(d))
NS_TRACK = sorted(_ns_set)

if __name__ == "__main__":
    ns, orig_line, rest_line, unbd_line = [], [], [], []

    print(f"Tracking ns = {NS_TRACK}")
    print("n | manip awards (Original / Restricted / Unrestricted)")
    print("-" * 78)

    for n in NS_TRACK:
        claims = [E_FIXED / n] * (n - 1) + [D_BIG]
        m = n - 1
        vec = np.array(claims, dtype=np.float64) / SCALE
        R_truth = truthful_matrix(vec)

        awards_orig = awards_from_matrix(R_truth, estate=E_FIXED, scale=SCALE)
        manip_orig = float(awards_orig[m])

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

    # Save plot data
    print("\n=== SAVED DATA POINTS (n, Original, Restricted, Unrestricted) ===")
    for i in range(len(ns)):
        print(f"({ns[i]}, {orig_line[i]:.6f}, {rest_line[i]:.6f}, {unbd_line[i]:.6f})")

    # -------------------- 4) Plot --------------------
    plt.figure(figsize=(6.6, 4.2))
    plt.plot(ns, orig_line, marker="o", linestyle="-", linewidth=2, label="Original")
    plt.plot(ns, rest_line, marker="s", linestyle="--", linewidth=2, label="Restricted")
    plt.plot(ns, unbd_line, marker="^", linestyle=":", linewidth=2, label="Unrestricted")
    plt.xlabel("Number of agents (n)")
    plt.ylabel(f"Manipulator payoff (CEA, E={E_FIXED})")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.title("CEA – manipulator payoff vs n  (test case: [E/n,...,E/n, d_big])")
    plt.legend()
    plt.tight_layout()

    ns_tag = "-".join(map(str, ns))
    out_name = f"cea_E{int(E_FIXED)}_dbig{int(D_BIG)}_ns{ns_tag}.png"
    plt.savefig(out_name, dpi=240)
    plt.close()
    print(f"✅ Saved: {out_name}")
