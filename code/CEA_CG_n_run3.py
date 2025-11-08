# =================== CG-CEA: run + plot for E=[35,40,45,50], n=4,5,6,7 - CSV + Ratio + PDF ===================
import math, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
from functools import lru_cache
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

# -------------------- 1) Core methods with CG-CEA --------------------
@njit(cache=True)
def cg_cea_allocation_numba(claims, estate, iters=60):
    """Contested Garment CEA - Allocates at most claims[i]/2 to each agent"""
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

def cea_allocation(claims, estate, iters=60):
    """CG-CEA allocation wrapper"""
    c = np.asarray(claims, dtype=np.float64)
    return cg_cea_allocation_numba(c, estate, iters)

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

@lru_cache(maxsize=8192)
def impartial_division_cached(R_tuple, eps=1e-9):
    """Cached version - takes tuple instead of array"""
    n = int(np.sqrt(len(R_tuple)))
    R = np.array(R_tuple, dtype=np.float64).reshape(n, n)
    return tuple(_impartial_division_core(R, eps))

def impartial_division(R, eps=1e-9):
    """Wrapper that converts array to tuple for caching"""
    return np.array(impartial_division_cached(tuple(R.flatten()), eps))

# -------------------- Composition generator --------------------
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

# -------------------- 2) Manipulation search --------------------
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

# -------------------- 3) Main computation --------------------
SCALE = 100
ALPHA = 0.01
D_BIG = 60.0
NS_TRACK = [4, 5, 6, 7]
E_VALUES = [35, 40, 45, 50]

if __name__ == "__main__":
    # Warmup numba
    print("🔥 Warming up numba...")
    dummy_claims = np.array([1.0, 2.0, 3.0, 4.0])
    _ = cg_cea_allocation_numba(dummy_claims, 20.0)
    dummy_R = np.random.rand(4, 4)
    _ = _impartial_division_core(dummy_R)
    print("✅ Numba ready!\n")

    all_results = {}

    for E_FIXED in E_VALUES:
        print(f"\n{'='*70}")
        print(f"Running for E={E_FIXED}")
        print(f"{'='*70}")

        ns, orig_line, rest_line, unbd_line = [], [], [], []

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

            pbar_r = tqdm(total=total_candidates, desc=f"[E={E_FIXED}, n={n}] restricted", unit="cand", leave=True)
            rest_best, _, _ = all_opt_rows_restricted(
                R_truth, m, estate=E_FIXED, scale=SCALE, alpha=ALPHA, pbar=pbar_r
            )

            pbar_u = tqdm(total=total_candidates, desc=f"[E={E_FIXED}, n={n}] unbounded", unit="cand", leave=True)
            unbd_best, _ = all_opt_rows_unbounded(
                R_truth, m, estate=E_FIXED, scale=SCALE, alpha=ALPHA, pbar=pbar_u
            )

            ns.append(n)
            orig_line.append(manip_orig)
            rest_line.append(float(rest_best))
            unbd_line.append(float(unbd_best))

            print(f"n={n} | Original: {manip_orig:.6f}, Restricted: {float(rest_best):.6f}, Unrestricted: {float(unbd_best):.6f}")

        all_results[E_FIXED] = {
            'n': ns,
            'original': orig_line,
            'restricted': rest_line,
            'unrestricted': unbd_line
        }

    # Save CSV files (combined absolute + ratio)
    print(f"\n{'='*70}")
    print("💾 Saving CSV files...")
    print(f"{'='*70}")

    for E in E_VALUES:
        data = all_results[E]

        # Combined CSV with both absolute values and ratios
        df = pd.DataFrame({
            'n': data['n'],
            'original': data['original'],
            'restricted': data['restricted'],
            'unrestricted': data['unrestricted'],
            'restricted/original': np.array(data['restricted']) / np.array(data['original']),
            'unrestricted/original': np.array(data['unrestricted']) / np.array(data['original'])
        })
        csv_file = f"CG_CEA_n_E{E}.csv"
        df.to_csv(csv_file, index=False)
        print(f"✅ Saved {csv_file}")

    # Generate PDF plots
    print(f"\n{'='*70}")
    print("📊 Generating PDF plots...")
    print(f"{'='*70}")

    for E in E_VALUES:
        data = all_results[E]

        plt.figure(figsize=(8, 6))
        plt.plot(data['n'], data['original'], marker="o", linestyle="-", linewidth=2, markersize=8, label="Original")
        plt.plot(data['n'], data['restricted'], marker="s", linestyle="--", linewidth=2, markersize=8, label="Restricted")
        plt.plot(data['n'], data['unrestricted'], marker="^", linestyle=":", linewidth=2, markersize=8, label="Unrestricted")

        plt.xlabel("Number of agents (n)", fontsize=12)
        plt.ylabel(f"Manipulator payoff", fontsize=12)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        plt.title(f"CG-CEA – Manipulator payoff vs n (E={E})", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        pdf_file = f"CG_CEA_n_E{E}.pdf"
        plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved {pdf_file}")
        plt.close()

    print(f"\n{'='*70}")
    print("✅ ALL TASKS COMPLETED!")
    print(f"{'='*70}")
    print(f"\nGenerated files for E={E_VALUES}:")
    for E in E_VALUES:
        print(f"  - CG_CEA_n_E{E}.csv (absolute + ratio)")
        print(f"  - CG_CEA_n_E{E}.pdf")
