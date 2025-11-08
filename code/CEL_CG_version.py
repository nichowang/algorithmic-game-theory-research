"""
CG-CEL Implementation: Contested Garment with CEL
Based on the paper's description (Section 2.2.2, Page 3)

Key difference from standard CEL in CEL1.py:
- When E > D/2, we first allocate D/2 total budget (each gets d_i/2)
- Then run CEL on the REMAINING effective claims with estate E' = E - D/2
- Unanimous vector should be (D-E, E - (n-1)*0.01, 0.01, ..., 0.01)
"""

import numpy as np
import itertools
from math import comb
from tqdm import tqdm
import matplotlib.pyplot as plt

WEIGHT_BY_MULTIPLICITY = True

# ---------- Core: CEL (unchanged from CEL1.py) ----------
def cel_allocation(claims, estate, tol=1e-12, iters=60):
    """Constrained Equal Losses allocation"""
    c = np.asarray(claims, float)
    lo, hi = 0.0, c.max()
    for _ in range(iters):
        lam = (lo + hi) / 2
        total = np.maximum(0.0, c - lam).sum()
        if total > estate:
            lo = lam
        else:
            hi = lam
    lam = hi
    a = np.maximum(0.0, c - lam)
    # scale to match the exact estate
    if a.sum() > 0:
        a *= estate / a.sum()
    return a

# ---------- CG-CEL: Contested Garment Rule with CEL ----------
def cg_cel_allocation(claims, estate):
    """
    Contested Garment Rule using CEL

    When E > D/2:
    1. First allocate d_i/2 to each agent i (total D/2 allocated)
    2. Run CEL on remaining claims d_i/2 with estate E' = E - D/2

    Args:
        claims: numpy array of claims (should sum to D)
        estate: E (total budget available)

    Returns:
        numpy array of allocations
    """
    c = np.asarray(claims, float)
    D = c.sum()
    E = estate

    if E <= D / 2:
        # CEA regime (not implemented here, focus on CEL)
        raise NotImplementedError("CEA regime (E <= D/2) not needed for this experiment")
    else:
        # CEL regime: E > D/2
        # Step 1: Allocate d_i/2 to each agent
        initial_allocation = c / 2.0

        # Step 2: Run CEL on remaining claims (c/2) with remaining estate (E - D/2)
        remaining_claims = c / 2.0
        remaining_estate = E - D / 2.0

        cel_allocation_result = cel_allocation(remaining_claims, remaining_estate)

        # Total allocation = initial + CEL result
        total_allocation = initial_allocation + cel_allocation_result

        return total_allocation

def truthful_matrix(vec):
    """Create truthful reporting matrix where all rows are identical"""
    v = np.asarray(vec, float)
    return np.tile(v, (len(v), 1))

def awards_from_matrix_cg_cel(R, estate=60, scale=100):
    """
    Get CG-CEL awards from a reporting matrix R
    Uses row-averaged shares as claims
    """
    claims = R.mean(0) * scale  # claims vector
    return cg_cel_allocation(claims, estate)

# ---------- Impartial Division (unchanged) ----------
def impartial_division(R, eps=1e-9):
    R, n = np.asarray(R, float), R.shape[0]
    phi = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            ratios = [R[l, a] / (R[l, b] + eps)
                      for l in range(n) if l not in (a, b) and R[l, b] > eps]
            phi[a, b] = (sum(ratios) / len(ratios)) if ratios else 1.0

    def psi(res, k, i):
        reps = [l for l in range(n) if l not in (res, k, i) and R[l, i] > eps]
        return (sum(R[l, k] / (R[l, i] + eps) for l in reps) / len(reps)) if reps else 1.0

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

# ---------- Enumerate compositions ----------
def compositions_nonneg(free_units, k):
    for cuts in itertools.combinations(range(free_units + k - 1), k - 1):
        add = np.diff((-1, *cuts, free_units + k - 1)) - 1
        yield add

# ---------- Restricted manipulation: ID -> CG-CEL ----------
def restricted_best_cg_cel(R0, m, estate=60, scale=100, alpha=0.01, tol=1e-9):
    """
    Best manipulation via Impartial Division then CG-CEL
    """
    n = R0.shape[0]
    fixed = float(R0[m, m])
    others = [i for i in range(n) if i != m]
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        return float("-inf")

    free_units = scale - min_units * k
    best_val = -1.0

    for add in compositions_nonneg(free_units, k):
        p_units = np.asarray(add) + min_units
        p = p_units / p_units.sum()
        rem = 1.0 - fixed

        row = np.zeros(n, dtype=float)
        row[m] = fixed
        row[others] = p * rem

        R = R0.copy()
        R[m] = row
        shares = impartial_division(R)
        val = cg_cel_allocation(shares * scale, estate)[m]
        if val > best_val + tol:
            best_val = val

    return best_val

# ---------- Unrestricted manipulation: direct CG-CEL ----------
def unrestricted_best_cg_cel(R0, m, estate=60, scale=100, alpha=0.01, tol=1e-9):
    """
    Best manipulation via direct CG-CEL
    """
    n = R0.shape[0]
    fixed = float(R0[m, m])
    others = [i for i in range(n) if i != m]
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        return float("-inf")

    free_units = scale - min_units * k
    best_val = -1.0

    for add in compositions_nonneg(free_units, k):
        p_units = np.asarray(add) + min_units
        p = p_units / p_units.sum()
        rem = 1.0 - fixed

        row = np.zeros(n, dtype=float)
        row[m] = fixed
        row[others] = p * rem

        val = cg_cel_allocation(row * scale, estate)[m]
        if val > best_val + tol:
            best_val = val

    return best_val

# ---------- Multiplicity ----------
def multiplicity_three(a1, a2, a3):
    if a1 == a2 == a3:
        return 1
    elif a1 == a2 or a2 == a3:
        return 3
    else:
        return 6

def count_unique_triples(rest_units, min_units=1):
    cnt = 0
    for a1 in range(min_units, rest_units - 2*min_units + 1):
        for a2 in range(a1, rest_units - a1 - min_units + 1):
            a3 = rest_units - a1 - a2
            if a3 >= a2 and a3 >= min_units:
                cnt += 1
    return cnt

# ---------- Main sweep for n=4 ----------
def sweep_cg_cel_n4(E=60, D=100, alpha=0.01, manip_index=0):
    """
    Sweep over unanimous vectors for n=4

    According to the paper, for CEL (E > D/2), the worst-case unanimous vector is:
    - Manipulator: (D-E)/2
    - One agent: (D+E)/2 - (n-2)*0.01 (essentially the remainder)
    - Other n-2 agents: 0.01 each

    But we sweep over all to see the pattern.
    """
    n = 4
    min_units = int(round(alpha * D))  # =1
    rows = []

    outer_total = (D - 3*min_units) - 1 + 1
    outer = tqdm(range(1, D - 3*min_units + 1),
                 desc="[a0] 0.01→0.97",
                 total=outer_total, position=0, leave=True, ncols=100)

    for a0_units in outer:
        a0 = a0_units / D
        rest_units = D - a0_units

        total_unique = count_unique_triples(rest_units, min_units=min_units)

        orig_sum = r_sum = u_sum = 0.0
        weight_sum = 0

        inner = tqdm(total=total_unique,
                     desc=f"[a0={a0:0.2f}] unique triples",
                     position=1, leave=False, ncols=100,
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        for a1_units in range(min_units, rest_units - 2*min_units + 1):
            for a2_units in range(a1_units, rest_units - a1_units - min_units + 1):
                a3_units = rest_units - a1_units - a2_units
                if a3_units < a2_units or a3_units < min_units:
                    continue

                v_units = np.array([a0_units, a1_units, a2_units, a3_units], dtype=float)
                v = v_units / D
                R0 = truthful_matrix(v)

                # Original: truthful -> CG-CEL
                orig = awards_from_matrix_cg_cel(R0, estate=E, scale=D)[manip_index]
                # Restricted: best manip row -> ID -> CG-CEL
                r_best = restricted_best_cg_cel(R0, m=manip_index, estate=E, scale=D, alpha=alpha)
                # Unrestricted: best manip row -> direct CG-CEL
                u_best = unrestricted_best_cg_cel(R0, m=manip_index, estate=E, scale=D, alpha=alpha)

                w = multiplicity_three(a1_units, a2_units, a3_units) if WEIGHT_BY_MULTIPLICITY else 1
                orig_sum += orig * w
                r_sum    += r_best * w
                u_sum    += u_best * w
                weight_sum += w

                inner.update(1)

        inner.close()

        rows.append({
            "a0": round(a0, 2),
            "original_avg": round(orig_sum / weight_sum, 6),
            "restricted_avg": round(r_sum    / weight_sum, 6),
            "unrestricted_avg": round(u_sum  / weight_sum, 6),
            "unique_cases": total_unique,
            "weight_sum": weight_sum,
        })

    import pandas as pd
    return pd.DataFrame(rows)

# ---------- Experiment for n=5,6,7 ----------
def sweep_cg_cel_general_n(n=5, E=60, D=100, alpha=0.01, manip_index=0):
    """
    For general n, sweep over all possible unanimous vectors
    This is computationally expensive, so we may sample or use specific cases
    """
    # For simplicity, let's test the "worst case" unanimous vector mentioned in paper
    # For CEL: manipulator gets (D-E)/2, one other gets large share, rest get 0.01

    min_units = int(round(alpha * D))  # =1

    # Worst case unanimous vector for CEL (from paper, Figure 2b caption):
    # Manipulator: (D-E)/2
    # One agent: (D+E)/2 - (n-2)*0.01
    # Other n-2 agents: 0.01

    manip_claim = (D - E) / 2.0
    other_large_claim = (D + E) / 2.0 - (n - 2) * alpha * D
    small_claim = alpha * D  # 0.01 * 100 = 1

    # Build unanimous vector
    v_units = np.zeros(n)
    v_units[manip_index] = manip_claim
    v_units[1 if manip_index != 1 else 2] = other_large_claim  # one other agent

    # Fill remaining agents with small_claim
    for i in range(n):
        if i != manip_index and v_units[i] == 0:
            v_units[i] = small_claim

    v = v_units / D
    R0 = truthful_matrix(v)

    # Original
    orig = awards_from_matrix_cg_cel(R0, estate=E, scale=D)[manip_index]

    # Restricted (this will be very slow for large n, so we simplify)
    # For demonstration, we compute a few manipulations
    r_best = restricted_best_cg_cel(R0, m=manip_index, estate=E, scale=D, alpha=alpha)

    # Unrestricted
    u_best = unrestricted_best_cg_cel(R0, m=manip_index, estate=E, scale=D, alpha=alpha)

    return {
        "n": n,
        "E": E,
        "original": orig,
        "restricted": r_best,
        "unrestricted": u_best,
        "unanimous_vector": v.tolist()
    }

# ---------- Run & plot for n=4 ----------
if __name__ == "__main__":
    print("=" * 80)
    print("CG-CEL Experiment: Testing the updated unanimous vector proposal")
    print("=" * 80)

    # Test for n=4 with E=60 (CEL regime)
    print("\n[1] Running sweep for n=4, E=60, D=100 (CEL regime)")
    E, D, alpha = 60, 100, 0.01
    df = sweep_cg_cel_n4(E=E, D=D, alpha=alpha, manip_index=0)

    print("\nFirst few rows:")
    print(df.head())
    print("\nLast few rows:")
    print(df.tail())

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["a0"], df["original_avg"],     label="Original",     linewidth=2, marker='o', markersize=3)
    plt.plot(df["a0"], df["restricted_avg"],   label="Restricted",   linewidth=2, marker='s', markersize=3)
    plt.plot(df["a0"], df["unrestricted_avg"], label="Unrestricted", linewidth=2, marker='^', markersize=3)
    plt.xlabel("Manipulator initial claim a0", fontsize=12)
    plt.ylabel("Final reward of manipulator (CG-CEL)", fontsize=12)
    plt.title(f"CG-CEL – Manipulator Reward vs a0 (n=4, E={E}, D={D})\n"
              f"weighted_by_multiplicity={WEIGHT_BY_MULTIPLICITY}", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("cg_cel_sweep_n4.png", dpi=200)
    print("\nPlot saved as: cg_cel_sweep_n4.png")
    plt.show()

    # Test for n=5,6,7 with specific unanimous vectors
    print("\n[2] Testing worst-case unanimous vectors for n=5,6,7")
    print("=" * 80)

    results = []
    for n in [4, 5, 6, 7]:
        print(f"\nTesting n={n}...")
        result = sweep_cg_cel_general_n(n=n, E=60, D=100, alpha=0.01, manip_index=0)
        results.append(result)
        print(f"  Unanimous vector: {result['unanimous_vector']}")
        print(f"  Original:      {result['original']:.4f}")
        print(f"  Restricted:    {result['restricted']:.4f}")
        print(f"  Unrestricted:  {result['unrestricted']:.4f}")

    # Plot n vs rewards
    plt.figure(figsize=(10, 6))
    ns = [r['n'] for r in results]
    orig_vals = [r['original'] for r in results]
    rest_vals = [r['restricted'] for r in results]
    unre_vals = [r['unrestricted'] for r in results]

    plt.plot(ns, orig_vals, label="Original",     linewidth=2, marker='o', markersize=8)
    plt.plot(ns, rest_vals, label="Restricted",   linewidth=2, marker='s', markersize=8)
    plt.plot(ns, unre_vals, label="Unrestricted", linewidth=2, marker='^', markersize=8)
    plt.xlabel("Number of agents (n)", fontsize=12)
    plt.ylabel("Final reward of manipulator (CG-CEL)", fontsize=12)
    plt.title("CG-CEL Manipulator Reward vs n (E=60, D=100)\n"
              "Worst-case unanimous vector", fontsize=13)
    plt.xticks(ns)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("cg_cel_vs_n.png", dpi=200)
    print("\nPlot saved as: cg_cel_vs_n.png")
    plt.show()

    print("\n" + "=" * 80)
    print("CG-CEL Experiment Complete!")
    print("=" * 80)
