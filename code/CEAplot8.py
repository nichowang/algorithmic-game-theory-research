# =================== CEA: Extended to 8 agents with optimizations (E=30) ===================
import math, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
from numba import njit
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# -------------------- 1) Core methods (Numba optimized) --------------------
@njit(cache=True, fastmath=True)
def cea_allocation_fast(claims, estate, iters=60):
    c = claims.copy()
    lo, hi = 0.0, c.max()
    if c.sum() <= estate:
        return c
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        total = np.minimum(c, lam).sum()
        if total > estate:
            hi = lam
        else:
            lo = lam
    return np.minimum(c, hi)

def cea_allocation(claims, estate, iters=60):
    c = np.asarray(claims, dtype=np.float64)
    return cea_allocation_fast(c, estate, iters)

def truthful_matrix(vec):
    vec = np.asarray(vec, dtype=np.float64)
    return np.tile(vec, (len(vec), 1))

def awards_from_matrix(R, estate=30, scale=100):
    return cea_allocation(R.mean(axis=0) * scale, estate)

# -------- Optimized impartial_division (Numba) --------
@njit(cache=True, fastmath=True)
def impartial_division_fast(R, eps=1e-9):
    n = R.shape[0]
    phi = np.zeros((n, n))

    # Compute phi
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            sum_ratios = 0.0
            count = 0
            for l in range(n):
                if l != a and l != b and R[l, b] > eps:
                    sum_ratios += R[l, a] / (R[l, b] + eps)
                    count += 1
            phi[a, b] = sum_ratios / count if count > 0 else 1.0

    total = np.zeros(n)
    for j in range(n):
        f = np.zeros(n)
        for i in range(n):
            if i == j:
                continue

            # Compute psi sum inline
            psi_sum = 0.0
            for k in range(n):
                if k != i and k != j:
                    sum_ratios = 0.0
                    count = 0
                    for l in range(n):
                        if l != j and l != k and l != i and R[l, i] > eps:
                            sum_ratios += R[l, k] / (R[l, i] + eps)
                            count += 1
                    psi_sum += sum_ratios / count if count > 0 else 1.0

            f[i] = 1.0 / (1.0 + phi[j, i] + psi_sum)

        sum_f = 0.0
        for i in range(n):
            if i != j:
                sum_f += f[i]
        f[j] = 1.0 - sum_f

        for i in range(n):
            total[i] += f[i]

    shares = total / n
    sum_shares = shares.sum()
    return shares / sum_shares

def impartial_division(R, eps=1e-9):
    R = np.asarray(R, dtype=np.float64)
    return impartial_division_fast(R, eps)

# -------------------- Composition generation --------------------
def compositions_nonneg_iter(total, parts):
    """Generate all non-negative integer compositions"""
    if parts == 1:
        yield [total]
        return
    for i in range(total + 1):
        for rest in compositions_nonneg_iter(total - i, parts - 1):
            yield [i] + rest

def n_compositions(free_units, k):
    return math.comb(free_units + k - 1, k - 1)

# -------------------- Optimization functions (no nested functions for Numba) --------------------
@njit(cache=True, fastmath=True)
def evaluate_restricted_single(add_units, R0, m, estate, scale, min_units):
    """Evaluate a single composition for restricted case"""
    n = R0.shape[0]
    fixed = R0[m, m]
    k = len(add_units)

    row = np.zeros(n)
    row[m] = fixed
    R = R0.copy()
    rem = 1.0 - fixed
    inv_scale = 1.0 / scale

    # Fill row for others
    idx = 0
    for i in range(n):
        if i != m:
            row[i] = (add_units[idx] + min_units) * inv_scale * rem
            idx += 1

    R[m, :] = row
    shares = impartial_division_fast(R, 1e-9)
    val = cea_allocation_fast(shares * scale, estate, 60)[m]
    return val

@njit(cache=True, fastmath=True)
def evaluate_unbounded_single(add_units, R0, m, estate, scale, min_units):
    """Evaluate a single composition for unbounded case"""
    n = R0.shape[0]
    fixed = R0[m, m]
    k = len(add_units)

    row = np.zeros(n)
    row[m] = fixed
    rem = 1.0 - fixed
    inv_scale = 1.0 / scale

    # Fill row for others
    idx = 0
    for i in range(n):
        if i != m:
            row[i] = (add_units[idx] + min_units) * inv_scale * rem
            idx += 1

    val = cea_allocation_fast(row * scale, estate, 60)[m]
    return val

def optimize_restricted(R0, m, estate, scale, alpha, pbar=None):
    """Optimize restricted case"""
    n = R0.shape[0]
    k = n - 1
    min_units = int(round(alpha * scale))

    if min_units * k > scale:
        return float('-inf')

    free_units = scale - min_units * k
    best_val = -1.0

    for comp in compositions_nonneg_iter(free_units, k):
        add_units = np.array(comp, dtype=np.float64)
        val = evaluate_restricted_single(add_units, R0, m, estate, scale, min_units)
        if val > best_val:
            best_val = val
        if pbar:
            pbar.update(1)

    return best_val

def optimize_unbounded(R0, m, estate, scale, alpha, pbar=None):
    """Optimize unbounded case"""
    n = R0.shape[0]
    k = n - 1
    min_units = int(round(alpha * scale))

    if min_units * k > scale:
        return float('-inf')

    free_units = scale - min_units * k
    best_val = -1.0

    for comp in compositions_nonneg_iter(free_units, k):
        add_units = np.array(comp, dtype=np.float64)
        val = evaluate_unbounded_single(add_units, R0, m, estate, scale, min_units)
        if val > best_val:
            best_val = val
        if pbar:
            pbar.update(1)

    return best_val

# -------------------- Sequential processing (more reliable for large n) --------------------
def process_single_n(n, E_FIXED, SCALE, ALPHA, D_BIG):
    """Process a single n value"""
    claims = [E_FIXED / n] * (n - 1) + [D_BIG]
    m = n - 1
    vec = np.array(claims, dtype=np.float64) / SCALE
    R_truth = truthful_matrix(vec)

    # Original
    awards_orig = awards_from_matrix(R_truth, estate=E_FIXED, scale=SCALE)
    manip_orig = float(awards_orig[m])

    # Setup for optimization
    k = n - 1
    min_units = int(round(ALPHA * SCALE))
    free_units = max(0, SCALE - min_units * k)
    total_candidates = n_compositions(free_units, k)

    # Restricted optimization with progress bar
    pbar_r = tqdm(total=total_candidates, desc=f"[n={n}] restricted", unit="cand", leave=True)
    rest_best = optimize_restricted(R_truth, m, E_FIXED, SCALE, ALPHA, pbar_r)
    pbar_r.close()

    # Unbounded optimization with progress bar
    pbar_u = tqdm(total=total_candidates, desc=f"[n={n}] unbounded", unit="cand", leave=True)
    unbd_best = optimize_unbounded(R_truth, m, E_FIXED, SCALE, ALPHA, pbar_u)
    pbar_u.close()

    return n, manip_orig, rest_best, unbd_best

# -------------------- Test parameters --------------------
if __name__ == '__main__':
    E_FIXED = 30  # Changed from 20 to 30
    SCALE = 100
    ALPHA = 0.01
    D_BIG = 60.0

    BASE_NS = [4, 5, 6, 7, 8]  # Extended to include 8
    MUTATE_NS = []

    _ns_set = set(BASE_NS)
    for d in MUTATE_NS:
        if d > 0:
            _ns_set.add(d)
        elif d < 0:
            _ns_set.discard(abs(d))
    NS_TRACK = sorted(_ns_set)

    print(f"Tracking ns = {NS_TRACK}")
    print("n | manip awards (Original / Restricted / Unrestricted)")
    print("-"*78)

    # Process each n value sequentially
    results = []
    for n in NS_TRACK:
        result = process_single_n(n, E_FIXED, SCALE, ALPHA, D_BIG)
        results.append(result)
        n_val, orig, rest, unbd = result
        print(f"{n_val} | {orig:.6f} / {rest:.6f} / {unbd:.6f}")

    # Extract results for plotting
    ns, orig_line, rest_line, unbd_line = [], [], [], []
    for n, orig, rest, unbd in results:
        ns.append(n)
        orig_line.append(orig)
        rest_line.append(float(rest))
        unbd_line.append(float(unbd))

    # -------------------- Plot --------------------
    plt.figure(figsize=(6.6, 4.2))
    plt.plot(ns, orig_line, marker="o", linestyle="-", linewidth=2, label="Original")
    plt.plot(ns, rest_line, marker="s", linestyle="--", linewidth=2, label="Restricted")
    plt.plot(ns, unbd_line, marker="^", linestyle=":", linewidth=2, label="Unrestricted")
    plt.xlabel("Number of agents (n)")
    plt.ylabel(f"Manipulator payoff (CEA, E={E_FIXED})")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.title("CEA – manipulator payoff vs n (test case: [E/n,...,E/n, d_big])")
    plt.legend()
    plt.tight_layout()

    ns_tag = "-".join(map(str, ns))
    out_name = f"cea_E{int(E_FIXED)}_dbig{int(D_BIG)}_ns{ns_tag}.png"
    plt.savefig(out_name, dpi=240)
    plt.close()
    print(f"✅ Saved: {out_name}")

    # Print computation statistics
    print("\nComputation statistics:")
    for n in NS_TRACK:
        k = n - 1
        min_units = int(round(ALPHA * SCALE))
        free_units = max(0, SCALE - min_units * k)
        total_comps = n_compositions(free_units, k)
        print(f"n={n}: {total_comps:,} compositions per optimization")