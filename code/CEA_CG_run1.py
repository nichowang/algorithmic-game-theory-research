# =================== CG-CEA sweep for E=[5,10,15] - CSV + Ratio + PDF ===================
import numpy as np
import itertools
from tqdm import tqdm
from functools import lru_cache
import matplotlib.pyplot as plt
import pandas as pd

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

WEIGHT_BY_MULTIPLICITY = True
E_VALUES = [5, 10, 15]  
D = 100  
ALPHA = 0.01


@njit(cache=True)
def cg_cea_allocation_numba(claims, estate, iters=60):
    """Contested Garment CEA allocation - Allocates at most claims[i]/2 to each agent i"""
    n = len(claims)
    half_claims = np.empty(n, dtype=np.float64)
    for i in range(n):
        half_claims[i] = claims[i] * 0.5

    lo, hi = 0.0, 0.0
    for i in range(n):
        if half_claims[i] > hi:
            hi = half_claims[i]

    for _ in range(iters):
        lam = (lo + hi) * 0.5
        total = 0.0
        for i in range(n):
            if half_claims[i] < lam:
                total += half_claims[i]
            else:
                total += lam
        if total > estate:
            hi = lam
        else:
            lo = lam

    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        if half_claims[i] < hi:
            result[i] = half_claims[i]
        else:
            result[i] = hi
    return result

def cea_allocation(claims, estate, tol=1e-12, iters=60):
    """CG-CEA allocation wrapper"""
    c = np.asarray(claims, dtype=np.float64)
    return cg_cea_allocation_numba(c, estate, iters)

@njit(cache=True)
def truthful_matrix_numba(vec):
    """Numba-optimized truthful matrix"""
    n = len(vec)
    result = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            result[i, j] = vec[j]
    return result

def truthful_matrix(vec):
    v = np.asarray(vec, dtype=np.float64)
    return truthful_matrix_numba(v)

def awards_from_matrix(R, estate=20, scale=100):
    return cea_allocation(R.mean(0) * scale, estate)


@njit(cache=True)
def _impartial_division_core(R, eps=1e-9):
    """Numba-optimized core computation"""
    n = R.shape[0]
    phi = np.ones((n, n), dtype=np.float64)


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

@lru_cache(maxsize=16384)
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

    row = np.empty(n, dtype=np.float64)
    row[m] = fixed

    for add in comp:
        p_units_sum = (add + min_units).sum()
        row[others] = (add + min_units) * (rem / p_units_sum)

        val = cea_allocation(row * scale, estate)[m]
        if val > best_val:
            best_val = val
    return best_val

# ---------- Multiplicity computation ----------
@njit(cache=True)
def multiplicity_three_numba(a1, a2, a3):
    if a1 == a2 == a3:
        return 1
    elif a1 == a2 or a2 == a3 or a1 == a3:
        return 3
    else:
        return 6

# ---------- Count unique triples ----------
def count_unique_triples(rest_units, min_units=1):
    cnt = 0
    for a1 in range(min_units, rest_units - 2*min_units + 1):
        for a2 in range(a1, rest_units - a1 - min_units + 1):
            a3 = rest_units - a1 - a2
            if a3 >= a2 and a3 >= min_units:
                cnt += 1
    return cnt

# ---------- Main sweep for single E value ----------
def sweep_for_single_E(E, D=100, alpha=0.01, manip_index=0):
    n = 4
    min_units = int(round(alpha * D))
    rows = []

    outer_total = D - 3 * min_units
    outer = tqdm(
        range(1, D - 3 * min_units + 1),
        desc=f"[E={E}] a0 sweep",
        total=outer_total,
        position=0,
        leave=True,
        ncols=100,
    )

    for a0_units in outer:
        a0 = a0_units / D
        rest_units = D - a0_units

        orig_sum = r_sum = u_sum = weight_sum = 0.0
        total_triples = count_unique_triples(rest_units, min_units)

        for a1_units in range(min_units, rest_units - 2 * min_units + 1):
            for a2_units in range(a1_units, rest_units - a1_units - min_units + 1):
                a3_units = rest_units - a1_units - a2_units
                if a3_units >= a2_units and a3_units >= min_units:
                    v = np.array([a0_units, a1_units, a2_units, a3_units], dtype=np.float64) / D
                    R0 = truthful_matrix(v)

                    orig = awards_from_matrix(R0, estate=E, scale=D)[manip_index]
                    r_best = restricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha)
                    u_best = unrestricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha)

                    w = multiplicity_three_numba(a1_units, a2_units, a3_units) if WEIGHT_BY_MULTIPLICITY else 1
                    orig_sum += orig * w
                    r_sum += r_best * w
                    u_sum += u_best * w
                    weight_sum += w

        rows.append({
            "a0": round(a0, 2),
            "original": round(orig_sum / weight_sum, 6),
            "restricted": round(r_sum / weight_sum, 6),
            "unrestricted": round(u_sum / weight_sum, 6),
        })

    return pd.DataFrame(rows)

# ---------- Main execution ----------
if __name__ == "__main__":
    print(" Warming up numba JIT compilation...")
    dummy_claims = np.array([1.0, 2.0, 3.0, 4.0])
    _ = cg_cea_allocation_numba(dummy_claims, 20.0)
    dummy_vec = np.array([0.25, 0.25, 0.25, 0.25])
    _ = truthful_matrix_numba(dummy_vec)
    dummy_R = np.random.rand(4, 4)
    _ = _impartial_division_core(dummy_R)
    _ = multiplicity_three_numba(1, 2, 3)
    print(" Numba compilation complete!\n")

    print(" Precomputing composition cache...")
    for rest in range(3, 100):
        _ = get_compositions_cached(rest - 3, 3)
    print(" Composition cache ready!\n")

    all_dfs = {}

    for E in E_VALUES:
        print(f"\n{'='*60}")
        print(f" Running for E={E}")
        print(f"{'='*60}")
        df = sweep_for_single_E(E=E, D=D, alpha=ALPHA)
        all_dfs[E] = df

        # Save CSV with absolute values
        csv_filename = f"CG_CEA_E{E}_absolute.csv"
        df.to_csv(csv_filename, index=False)
        print(f" Saved {csv_filename}")

    # Compute and save ratios
    print(f"\n{'='*60}")
    print(" Computing ratios...")
    print(f"{'='*60}")

    for E in E_VALUES:
        df = all_dfs[E]
        df_ratio = df.copy()
        df_ratio['restricted/original'] = df['restricted'] / df['original']
        df_ratio['unrestricted/original'] = df['unrestricted'] / df['original']

        csv_ratio_filename = f"CG_CEA_E{E}_ratio.csv"
        df_ratio[['a0', 'restricted/original', 'unrestricted/original']].to_csv(csv_ratio_filename, index=False)
        print(f"Saved {csv_ratio_filename}")

    # Create separate plots for each E value
    print(f"\n{'='*60}")
    print("Creating plots...")
    print(f"{'='*60}")

    pdf_files = []
    for E in E_VALUES:
        df = all_dfs[E]

        plt.figure(figsize=(8, 6))
        plt.plot(df["a0"], df["original"], label="Original", linewidth=2, marker='o', markersize=4)
        plt.plot(df["a0"], df["restricted"], label="Restricted", linewidth=2, marker='s', markersize=4)
        plt.plot(df["a0"], df["unrestricted"], label="Unrestricted", linewidth=2, marker='^', markersize=4)

        plt.xlabel("Manipulator initial claim a0", fontsize=12)
        plt.ylabel("Final reward of manipulator", fontsize=12)
        plt.title(f"CG-CEA Manipulator Reward vs a0 (E={E}, D={D}, n=4)", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()

        pdf_filename = f"CG_CEA_E{E}.pdf"
        plt.savefig(pdf_filename, dpi=300, bbox_inches='tight')
        pdf_files.append(pdf_filename)
        print(f" Saved {pdf_filename}")
        plt.close()

    print(f"\n{'='*60}")
    print(" ALL TASKS COMPLETED!")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    for E in E_VALUES:
        print(f"  - CG_CEA_E{E}_absolute.csv")
        print(f"  - CG_CEA_E{E}_ratio.csv")
        print(f"  - CG_CEA_E{E}.pdf")
