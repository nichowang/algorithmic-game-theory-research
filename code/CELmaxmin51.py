# =================== CEL sweep (max-min differences) - Ultra Optimized for E=51 ===================
import numpy as np
import itertools
from math import comb
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import jit, njit, prange, types
from numba.typed import Dict
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# ---------- Core: CEL (Numba optimized) ----------
@njit(cache=True, fastmath=True, nogil=True)
def cel_allocation_fast(claims, estate, iters=60):
    c = claims
    lo, hi = 0.0, c.max()
    for _ in range(iters):
        lam = (lo + hi) * 0.5
        total = 0.0
        for i in range(len(c)):
            val = c[i] - lam
            if val > 0:
                total += val
        if total > estate:
            lo = lam
        else:
            hi = lam
    lam = hi
    a = np.empty_like(c)
    sum_a = 0.0
    for i in range(len(c)):
        val = c[i] - lam
        a[i] = val if val > 0 else 0.0
        sum_a += a[i]
    if sum_a > 0:
        factor = estate / sum_a
        for i in range(len(a)):
            a[i] *= factor
    return a

# ---------- Impartial Division (Ultra optimized) ----------
@njit(cache=True, fastmath=True, nogil=True, parallel=False)
def impartial_division_ultra(R, eps=1e-9):
    n = R.shape[0]
    phi = np.zeros((n, n))

    # Pre-compute phi - unrolled for n=4
    for a in prange(n):
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

    # Main computation - optimized for n=4
    for j in range(n):
        f = np.zeros(n)

        for i in range(n):
            if i == j:
                continue

            # Inline psi computation
            psi_sum = 0.0
            for k in range(n):
                if k == i or k == j:
                    continue
                sum_ratios = 0.0
                count = 0
                for l in range(n):
                    if l != j and l != k and l != i and R[l, i] > eps:
                        sum_ratios += R[l, k] / (R[l, i] + eps)
                        count += 1
                if count > 0:
                    psi_sum += sum_ratios / count
                else:
                    psi_sum += 1.0

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
    for i in range(len(shares)):
        shares[i] /= sum_shares
    return shares

# ---------- Complete optimization function ----------
@njit(cache=True, fastmath=True, nogil=True)
def compute_all_values_for_a0(a0_units, D, E, min_units, manip_index):
    """Compute all values for a single a0 configuration"""
    n = 4
    rest_units = D - a0_units

    # Count total combinations
    count = 0
    for a1 in range(min_units, rest_units - 2*min_units + 1):
        for a2 in range(a1, rest_units - a1 - min_units + 1):
            a3 = rest_units - a1 - a2
            if a3 >= a2 and a3 >= min_units:
                count += 1

    if count == 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0]), 0

    # Allocate result arrays
    orig_vals = np.empty(count)
    r_vals = np.empty(count)
    u_vals = np.empty(count)

    # Pre-allocate working arrays
    v = np.zeros(4)
    v[0] = a0_units / D
    R0 = np.zeros((4, 4))
    row = np.zeros(4)
    R_work = np.zeros((4, 4))

    idx = 0
    free_units = D - min_units * 3

    for a1_units in range(min_units, rest_units - 2*min_units + 1):
        for a2_units in range(a1_units, rest_units - a1_units - min_units + 1):
            a3_units = rest_units - a1_units - a2_units
            if a3_units < a2_units or a3_units < min_units:
                continue

            # Update v
            v[1] = a1_units / D
            v[2] = a2_units / D
            v[3] = a3_units / D

            # Create truthful matrix (optimized)
            for i in range(4):
                for j in range(4):
                    R0[i, j] = v[j]

            # Original value
            mean_claims = np.zeros(4)
            for j in range(4):
                sum_val = 0.0
                for i in range(4):
                    sum_val += R0[i, j]
                mean_claims[j] = sum_val / 4.0 * D
            orig_vals[idx] = cel_allocation_fast(mean_claims, E, 60)[manip_index]

            # Restricted best - inline optimization
            best_r = -1.0
            fixed = R0[manip_index, manip_index]
            rem = 1.0 - fixed
            row[manip_index] = fixed

            for b1 in range(free_units + 1):
                for b2 in range(free_units - b1 + 1):
                    b3 = free_units - b1 - b2

                    p_sum = (b1 + b2 + b3 + 3 * min_units) / D
                    p1 = ((b1 + min_units) / D) / p_sum * rem
                    p2 = ((b2 + min_units) / D) / p_sum * rem
                    p3 = ((b3 + min_units) / D) / p_sum * rem

                    # Fill row
                    idx_other = 0
                    for i in range(4):
                        if i != manip_index:
                            if idx_other == 0:
                                row[i] = p1
                            elif idx_other == 1:
                                row[i] = p2
                            else:
                                row[i] = p3
                            idx_other += 1

                    # Copy R0 to R_work
                    for i in range(4):
                        for j in range(4):
                            R_work[i, j] = R0[i, j]
                    R_work[manip_index] = row

                    shares = impartial_division_ultra(R_work, 1e-9)
                    val = cel_allocation_fast(shares * D, E, 60)[manip_index]

                    if val > best_r:
                        best_r = val

            r_vals[idx] = best_r

            # Unrestricted best - inline optimization
            best_u = -1.0
            for b1 in range(free_units + 1):
                for b2 in range(free_units - b1 + 1):
                    b3 = free_units - b1 - b2

                    p_sum = (b1 + b2 + b3 + 3 * min_units) / D
                    p1 = ((b1 + min_units) / D) / p_sum * rem
                    p2 = ((b2 + min_units) / D) / p_sum * rem
                    p3 = ((b3 + min_units) / D) / p_sum * rem

                    idx_other = 0
                    for i in range(4):
                        if i != manip_index:
                            if idx_other == 0:
                                row[i] = p1
                            elif idx_other == 1:
                                row[i] = p2
                            else:
                                row[i] = p3
                            idx_other += 1

                    val = cel_allocation_fast(row * D, E, 60)[manip_index]

                    if val > best_u:
                        best_u = val

            u_vals[idx] = best_u
            idx += 1

    return orig_vals, r_vals, u_vals, count

# ---------- Wrapper for multiprocessing ----------
def process_a0_wrapper(args):
    a0_units, D, E, min_units, manip_index = args
    return compute_all_values_for_a0(a0_units, D, E, min_units, manip_index)

# ---------- Main sweep with parallel processing ----------
def sweep_minmax_differences_with_progress(E=51, D=100, alpha=0.01, manip_index=0):
    n = 4
    min_units = int(round(alpha * D))
    rows = []

    a0_range = list(range(1, D - 3*min_units + 1))

    # Prepare arguments for parallel processing
    args_list = [
        (a0_units, D, E, min_units, manip_index)
        for a0_units in a0_range
    ]

    # Use multiprocessing for parallel computation
    num_workers = min(cpu_count() - 1, 8)  # Leave one CPU free, max 8 workers

    print(f"Processing with {num_workers} parallel workers...")

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_a0_wrapper, args_list),
            total=len(args_list),
            desc="[a0] 0.01→0.97",
            ncols=100
        ))

    # Process results
    for a0_units, result in zip(a0_range, results):
        orig_vals, r_vals, u_vals, count = result

        if count > 0:
            a0 = a0_units / D

            # Compute differences
            restricted_orig_diff = r_vals.max() - orig_vals.min()
            unrestricted_orig_diff = u_vals.max() - orig_vals.min()
            unrestricted_restricted_diff = u_vals.max() - r_vals.min()

            rows.append({
                "a0": round(a0, 2),
                "restricted_original": round(restricted_orig_diff, 6),
                "unrestricted_original": round(unrestricted_orig_diff, 6),
                "unrestricted_restricted": round(unrestricted_restricted_diff, 6),
                "unique_cases": count,
            })

    import pandas as pd
    return pd.DataFrame(rows)

# ---------- Run & plot ----------
if __name__ == "__main__":
    E, D, alpha = 51, 100, 0.01
    df = sweep_minmax_differences_with_progress(E=E, D=D, alpha=alpha, manip_index=0)

    print(df.head()); print(df.tail())

    plt.figure(figsize=(10, 6))
    plt.plot(df["a0"], df["restricted_original"], label="Restricted - Original", linewidth=2)
    plt.plot(df["a0"], df["unrestricted_original"], label="Unrestricted - Original", linewidth=2)
    plt.plot(df["a0"], df["unrestricted_restricted"], label="Unrestricted - Restricted", linewidth=2)
    plt.xlabel("Manipulator initial claim a0")
    plt.ylabel("Max-Min Difference in Manipulator Reward")
    plt.title(f"CEL – Max-Min Differences vs a0 (E={E}, D={D}, n=4)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cel_sweep_a0_minmax_differences_E51.png", dpi=200)
    plt.show()