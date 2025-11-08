"""
CG-CEL Implementation: Simple test for n=4,5,6,7
Only test the worst-case unanimous vector mentioned in the paper
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt

# ---------- Core: CEL ----------
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
    if a.sum() > 0:
        a *= estate / a.sum()
    return a

# ---------- CG-CEL: Contested Garment Rule ----------
def cg_cel_allocation(claims, estate):
    """
    Contested Garment Rule using CEL
    When E > D/2:
    1. Allocate d_i/2 to each agent
    2. Run CEL on remaining claims d_i/2 with estate E' = E - D/2
    """
    c = np.asarray(claims, float)
    D = c.sum()
    E = estate

    if E <= D / 2:
        raise NotImplementedError("CEA regime not needed for this test")
    else:
        # CEL regime
        initial_allocation = c / 2.0
        remaining_claims = c / 2.0
        remaining_estate = E - D / 2.0
        cel_result = cel_allocation(remaining_claims, remaining_estate)
        return initial_allocation + cel_result

def truthful_matrix(vec):
    v = np.asarray(vec, float)
    return np.tile(v, (len(v), 1))

# ---------- Impartial Division ----------
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

# ---------- Compositions ----------
def compositions_nonneg(free_units, k):
    for cuts in itertools.combinations(range(free_units + k - 1), k - 1):
        add = np.diff((-1, *cuts, free_units + k - 1)) - 1
        yield add

# ---------- Restricted: ID -> CG-CEL ----------
def restricted_best_cg_cel(R0, m, estate=60, scale=100, alpha=0.01):
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
        if val > best_val:
            best_val = val

    return best_val

# ---------- Unrestricted: direct CG-CEL ----------
def unrestricted_best_cg_cel(R0, m, estate=60, scale=100, alpha=0.01):
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
        if val > best_val:
            best_val = val

    return best_val

# ---------- Test for specific n ----------
def test_cg_cel_for_n(n, E=60, D=100, alpha=0.01, manip_index=0):
    """
    Test CG-CEL for specific n with the worst-case unanimous vector

    Based on paper Figure 2b, the worst-case for CEL:
    - Manipulator: (D-E)/2
    - One agent: (D+E)/2 - (n-2)*alpha*D  (essentially the remainder)
    - Other n-2 agents: alpha*D (0.01 each)
    """
    print(f"\n{'='*60}")
    print(f"Testing n={n}, E={E}, D={D}")
    print(f"{'='*60}")

    # Build unanimous vector
    v_units = np.zeros(n)
    v_units[manip_index] = (D - E) / 2.0

    # One other agent gets large share
    other_large_idx = 1 if manip_index != 1 else 2
    v_units[other_large_idx] = (D + E) / 2.0 - (n - 2) * alpha * D

    # Fill remaining agents with small_claim
    for i in range(n):
        if i != manip_index and i != other_large_idx:
            v_units[i] = alpha * D

    # Normalize
    v = v_units / D

    print(f"Unanimous vector (scaled by D={D}):")
    print(f"  {v_units}")
    print(f"Unanimous vector (normalized):")
    print(f"  {v}")
    print(f"Sum check: {v.sum():.6f} (should be 1.0)")

    # Create truthful matrix
    R0 = truthful_matrix(v)

    # Original: truthful -> CG-CEL
    print(f"\nComputing original (truthful) allocation...")
    shares = R0.mean(0)  # all rows are same, so mean = v
    orig = cg_cel_allocation(shares * D, E)[manip_index]
    print(f"  Original reward: {orig:.4f}")

    # Restricted: ID -> CG-CEL
    print(f"\nComputing restricted best manipulation...")
    r_best = restricted_best_cg_cel(R0, m=manip_index, estate=E, scale=D, alpha=alpha)
    print(f"  Restricted best: {r_best:.4f}")
    print(f"  Gain: {r_best - orig:.4f}")

    # Unrestricted: direct CG-CEL
    print(f"\nComputing unrestricted best manipulation...")
    u_best = unrestricted_best_cg_cel(R0, m=manip_index, estate=E, scale=D, alpha=alpha)
    print(f"  Unrestricted best: {u_best:.4f}")
    print(f"  Gain: {u_best - orig:.4f}")

    return {
        "n": n,
        "E": E,
        "D": D,
        "unanimous_vector": v_units.tolist(),
        "original": orig,
        "restricted": r_best,
        "unrestricted": u_best,
        "restricted_gain": r_best - orig,
        "unrestricted_gain": u_best - orig
    }

# ---------- Main ----------
if __name__ == "__main__":
    print("="*80)
    print("CG-CEL Experiment: Testing n=4,5,6,7")
    print("="*80)

    E, D, alpha = 60, 100, 0.01

    results = []
    for n in [4, 5, 6, 7]:
        result = test_cg_cel_for_n(n=n, E=E, D=D, alpha=alpha, manip_index=0)
        results.append(result)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'n':<5} {'Original':<12} {'Restricted':<12} {'Unrestricted':<12} {'R-Gain':<10} {'U-Gain':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['n']:<5} {r['original']:<12.4f} {r['restricted']:<12.4f} "
              f"{r['unrestricted']:<12.4f} {r['restricted_gain']:<10.4f} {r['unrestricted_gain']:<10.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ns = [r['n'] for r in results]
    orig_vals = [r['original'] for r in results]
    rest_vals = [r['restricted'] for r in results]
    unre_vals = [r['unrestricted'] for r in results]

    # Plot 1: Rewards
    ax1.plot(ns, orig_vals, label="Original", linewidth=2.5, marker='o', markersize=10)
    ax1.plot(ns, rest_vals, label="Restricted", linewidth=2.5, marker='s', markersize=10)
    ax1.plot(ns, unre_vals, label="Unrestricted", linewidth=2.5, marker='^', markersize=10)
    ax1.set_xlabel("Number of agents (n)", fontsize=13)
    ax1.set_ylabel("Final reward of manipulator", fontsize=13)
    ax1.set_title(f"CG-CEL: Manipulator Reward vs n\n(E={E}, D={D})", fontsize=14)
    ax1.set_xticks(ns)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Plot 2: Gains
    rest_gains = [r['restricted_gain'] for r in results]
    unre_gains = [r['unrestricted_gain'] for r in results]

    ax2.plot(ns, rest_gains, label="Restricted Gain", linewidth=2.5, marker='s', markersize=10)
    ax2.plot(ns, unre_gains, label="Unrestricted Gain", linewidth=2.5, marker='^', markersize=10)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel("Number of agents (n)", fontsize=13)
    ax2.set_ylabel("Gain from manipulation", fontsize=13)
    ax2.set_title(f"CG-CEL: Manipulation Gain vs n\n(E={E}, D={D})", fontsize=14)
    ax2.set_xticks(ns)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("cg_cel_results.png", dpi=200)
    print(f"\nPlot saved as: cg_cel_results.png")
    plt.show()

    print("\n" + "="*80)
    print("CG-CEL Experiment Complete!")
    print("="*80)
