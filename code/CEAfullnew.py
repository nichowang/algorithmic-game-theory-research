# =================== CEA sweep (unique combos, weighted average) ===================
import numpy as np
import itertools
from tqdm import tqdm
from functools import lru_cache
import matplotlib.pyplot as plt

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

# ======= 全局开关：是否按多重度加权（推荐 True） =======
WEIGHT_BY_MULTIPLICITY = True

# ---------- Core: CEA ----------
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

def truthful_matrix(vec):
    v = np.asarray(vec, float)
    return np.tile(v, (len(v), 1))

def awards_from_matrix(R, estate=20, scale=100):
    return cea_allocation(R.mean(0) * scale, estate)

# ---------- Optimized impartial division with numba ----------
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

# ---------- Precompute compositions ----------
_COMPOSITIONS_CACHE = {}

def get_compositions_cached(free_units, k):
    """Cache compositions as numpy array for faster iteration"""
    key = (free_units, k)
    if key in _COMPOSITIONS_CACHE:
        return _COMPOSITIONS_CACHE[key]

    arr = []
    for cuts in itertools.combinations(range(free_units + k - 1), k - 1):
        add = np.diff((-1, *cuts, free_units + k - 1)) - 1
        arr.append(add)
    comp = np.asarray(arr, dtype=np.int32)
    _COMPOSITIONS_CACHE[key] = comp
    return comp


# ---------- Optimized restricted with minimal copying ----------
def restricted_best(R0, m, estate=20, scale=100, alpha=0.01):
    n = R0.shape[0]
    fixed = R0[m, m]
    others = np.array([i for i in range(n) if i != m], dtype=np.int32)
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        return -np.inf

    free_units = scale - min_units * k
    comp = get_compositions_cached(free_units, k)
    best_val = -1.0
    rem = 1.0 - fixed

    # Reuse arrays
    row = np.empty(n, dtype=np.float64)
    R = R0.copy()
    row[m] = fixed

    for add in comp:
        p_units_sum = (add + min_units).sum()
        row[others] = (add + min_units) * (rem / p_units_sum)
        R[m] = row

        shares = impartial_division(R)
        val = cea_allocation(shares * scale, estate)[m]
        if val > best_val:
            best_val = val
    return best_val


# ---------- Optimized unrestricted with minimal copying ----------
def unrestricted_best(R0, m, estate=20, scale=100, alpha=0.01):
    n = R0.shape[0]
    fixed = R0[m, m]
    others = np.array([i for i in range(n) if i != m], dtype=np.int32)
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        return -np.inf

    free_units = scale - min_units * k
    comp = get_compositions_cached(free_units, k)
    best_val = -1.0
    rem = 1.0 - fixed

    # Reuse array
    row = np.empty(n, dtype=np.float64)
    row[m] = fixed

    for add in comp:
        p_units_sum = (add + min_units).sum()
        row[others] = (add + min_units) * (rem / p_units_sum)

        val = cea_allocation(row * scale, estate)[m]
        if val > best_val:
            best_val = val
    return best_val


# ---------- 计算 (a1,a2,a3) 的多重度：等价排列个数（n=4 时是 1/3/6） ----------
def multiplicity_three(a1, a2, a3):
    if a1 == a2 == a3:
        return 1
    elif a1 == a2 or a2 == a3 or a1 == a3:
        return 3
    else:
        return 6

# ---------- 统计唯一三元组总数（用于设置 tqdm total） ----------
def count_unique_triples(rest_units, min_units=1):
    cnt = 0
    for a1 in range(min_units, rest_units - 2*min_units + 1):
        for a2 in range(a1, rest_units - a1 - min_units + 1):
            a3 = rest_units - a1 - a2
            if a3 >= a2 and a3 >= min_units:
                cnt += 1
    return cnt

# ---------- 主流程：对 a0=0.01..0.97 扫描 (single-threaded, optimized) ----------
def sweep_unique_combinations_with_progress(E=40, D=100, alpha=0.01, manip_index=0):
    n = 4
    min_units = int(round(alpha * D))
    rows = []

    outer_total = D - 3 * min_units
    outer = tqdm(
        range(1, D - 3 * min_units + 1),
        desc="[a0] 0.01→0.97",
        total=outer_total,
        position=0,
        leave=True,
        ncols=100,
    )

    for a0_units in outer:
        a0 = a0_units / D
        rest_units = D - a0_units

        orig_sum = r_sum = u_sum = weight_sum = 0.0

        # Count total triples for this a0
        total_triples = count_unique_triples(rest_units, min_units)

        # Inner progress bar for triples
        inner_pbar = tqdm(total=total_triples, desc=f"[a0={a0:0.2f}]", position=1, leave=False, ncols=100)

        for a1_units in range(min_units, rest_units - 2 * min_units + 1):
            for a2_units in range(a1_units, rest_units - a1_units - min_units + 1):
                a3_units = rest_units - a1_units - a2_units
                if a3_units >= a2_units and a3_units >= min_units:
                    v = np.array([a0_units, a1_units, a2_units, a3_units], dtype=np.float64) / D
                    R0 = truthful_matrix(v)

                    orig = awards_from_matrix(R0, estate=E, scale=D)[manip_index]
                    r_best = restricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha)
                    u_best = unrestricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha)

                    w = multiplicity_three(a1_units, a2_units, a3_units) if WEIGHT_BY_MULTIPLICITY else 1
                    orig_sum += orig * w
                    r_sum += r_best * w
                    u_sum += u_best * w
                    weight_sum += w

                    inner_pbar.update(1)

        inner_pbar.close()

        rows.append(
            {
                "a0": round(a0, 2),
                "original_avg": round(orig_sum / weight_sum, 6),
                "restricted_avg": round(r_sum / weight_sum, 6),
                "unrestricted_avg": round(u_sum / weight_sum, 6),
            }
        )

    import pandas as pd

    return pd.DataFrame(rows)


# ---------- 运行 & 画图 ----------
if __name__ == "__main__":
    E, D, alpha = 20, 100, 0.01
    df = sweep_unique_combinations_with_progress(E=E, D=D, alpha=alpha, manip_index=0)

    print(df.head())
    print(df.tail())

    # Extract and print specific a0 values [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    target_a0_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print("\n=== SAVED DATA POINTS at a0 = [0, 0.2, 0.4, 0.6, 0.8, 1.0] ===")
    for target_a0 in target_a0_values:
        # Find closest match in dataframe
        closest_row = df.iloc[(df["a0"] - target_a0).abs().argsort()[:1]]
        if len(closest_row) > 0:
            row = closest_row.iloc[0]
            print(
                f"a0={row['a0']:.2f}: Original={row['original_avg']:.6f}, Restricted={row['restricted_avg']:.6f}, Unrestricted={row['unrestricted_avg']:.6f}"
            )

    plt.figure(figsize=(8, 5))
    plt.plot(df["a0"], df["original_avg"], label="Original", linewidth=2)
    plt.plot(df["a0"], df["restricted_avg"], label="Restricted", linewidth=2)
    plt.plot(df["a0"], df["unrestricted_avg"], label="Unrestricted", linewidth=2)
    plt.xlabel("Manipulator initial claim a0")
    plt.ylabel("Final reward of manipulator (CEA)")
    plt.title(
        f"CEA – Manipulator Reward vs a0 (unique combos; E={E}, D={D}, n=4)\n"
        f"weighted_by_multiplicity={WEIGHT_BY_MULTIPLICITY}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cea_sweep_a0_unique_weighted.png", dpi=200)
    plt.show()
