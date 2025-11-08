# =================== CG-CEL Experiment ===================
# Setting: ((D-E)/2, (D+E)/2-(N-2)*alpha, alpha, alpha, ...)
# This is the worst-case CEL setting from the CHIPS paper Section 5.2
# D=100, E=55, N=4, alpha=0.01

import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt

# ======= Global switch: weight by multiplicity =======
WEIGHT_BY_MULTIPLICITY = True

# ---------- Core: CG (Contested Garment) ----------
def cg_allocation(claims, estate, tol=1e-12):
    """
    Contested Garment rule (Nucleolus for bankruptcy):
    - If E <= D/2: use CEA on half-claims
    - If E > D/2: give everyone half their claim, then use CEL on the rest
    """
    c = np.asarray(claims, float)
    D = c.sum()

    if estate <= D / 2:
        # CEA regime
        return cea_allocation(c / 2, estate)
    else:
        # CEL regime: give c_i/2 first, then use CEL for the remainder
        half_claims = c / 2
        remaining_estate = estate - D / 2
        cel_part = cel_allocation(c / 2, remaining_estate)
        return half_claims + cel_part

def cea_allocation(claims, estate, tol=1e-12, iters=60):
    """Constrained Equal Awards: min(c_i, lambda) where sum = E"""
    c = np.asarray(claims, float)
    lo, hi = 0.0, estate
    for _ in range(iters):
        lam = (lo + hi) / 2
        total = np.minimum(c, lam).sum()
        if total < estate - tol:
            lo = lam
        else:
            hi = lam
    a = np.minimum(c, hi)
    if a.sum() > 0:
        a *= estate / a.sum()
    return a

def cel_allocation(claims, estate, tol=1e-12, iters=60):
    """Constrained Equal Losses: max(0, c_i - lambda) where sum = E"""
    c = np.asarray(claims, float)
    D = c.sum()

    if D <= estate:
        return c

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

def truthful_matrix(vec):
    """Create truthful report matrix (all agents report the same)"""
    v = np.asarray(vec, float)
    return np.tile(v, (len(v), 1))

def awards_from_matrix(R, estate=55, scale=100):
    """Original: average reports, then apply CG"""
    return cg_allocation(R.mean(0) * scale, estate)

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

# ---------- Enumerate compositions ----------
def compositions_nonneg(free_units, k):
    for cuts in itertools.combinations(range(free_units + k - 1), k - 1):
        add = np.diff((-1, *cuts, free_units + k - 1)) - 1
        yield add

# ---------- Restricted: ID -> CG ----------
def restricted_best(R0, m, estate=55, scale=100, alpha=0.01, tol=1e-9):
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
        val = cg_allocation(shares * scale, estate)[m]
        if val > best_val + tol:
            best_val = val

    return best_val

# ---------- Unrestricted: direct CG ----------
def unrestricted_best(R0, m, estate=55, scale=100, alpha=0.01, tol=1e-9):
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

        val = cg_allocation(row * scale, estate)[m]
        if val > best_val + tol:
            best_val = val

    return best_val

# ---------- Multiplicity ----------
def multiplicity_three(a1, a2, a3):
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

# ---------- Main sweep ----------
def sweep_cg_cel(D=100, E=55, alpha=0.01, manip_index=0):
    """
    Sweep over a0 from 0.01 to 0.97, with (a0, a1, a2, a3) summing to D=100.
    E=55 > D/2=50, so we're in CEL regime.
    """
    n = 4
    min_units = int(round(alpha * D))

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
                     desc=f"[a0={a0:.2f}] unique triples",
                     position=1, leave=False, ncols=100,
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

        for a1_units in range(min_units, rest_units - 2*min_units + 1):
            for a2_units in range(a1_units, rest_units - a1_units - min_units + 1):
                a3_units = rest_units - a1_units - a2_units
                if a3_units < a2_units or a3_units < min_units:
                    continue

                v_units = np.array([a0_units, a1_units, a2_units, a3_units], dtype=float)
                v = v_units / D
                R0 = truthful_matrix(v)

                orig = awards_from_matrix(R0, estate=E, scale=D)[manip_index]
                r_best = restricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha)
                u_best = unrestricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha)

                w = multiplicity_three(a1_units, a2_units, a3_units) if WEIGHT_BY_MULTIPLICITY else 1
                orig_sum += orig * w
                r_sum += r_best * w
                u_sum += u_best * w
                weight_sum += w

                inner.update(1)

        inner.close()

        rows.append({
            "a0": round(a0, 2),
            "original_avg": round(orig_sum / weight_sum, 6),
            "restricted_avg": round(r_sum / weight_sum, 6),
            "unrestricted_avg": round(u_sum / weight_sum, 6),
            "unique_cases": total_unique,
            "weight_sum": weight_sum,
        })

    import pandas as pd
    return pd.DataFrame(rows)

# ---------- Run & plot ----------
if __name__ == "__main__":
    D, E, alpha = 100, 55, 0.01

    print("="*80)
    print("CG (Contested Garment) - CEL Regime Experiment")
    print("="*80)
    print(f"Parameters: D={D}, E={E}, N=4, alpha={alpha}")
    print(f"Since E={E} > D/2={D/2}, we are in the CEL regime")
    print(f"\nPaper's worst-case CEL setting:")
    print(f"  Agent 0 (manipulator): (D-E)/2 = {(D-E)/2}")
    print(f"  Agent 1: (D+E)/2 - (N-2)*alpha = {(D+E)/2 - 2*alpha}")
    print(f"  Agent 2: {alpha}")
    print(f"  Agent 3: {alpha}")
    print(f"  Total: {(D-E)/2 + (D+E)/2 - 2*alpha + 2*alpha} = {D}")
    print("="*80)
    print()

    df = sweep_cg_cel(D=D, E=E, alpha=alpha, manip_index=0)

    print("\n" + "="*80)
    print("Results Preview:")
    print("="*80)
    print(df.head(10))
    print("...")
    print(df.tail(10))

    df.to_csv("cg_cel_sweep_results.csv", index=False)
    print(f"\n✓ Results saved to: cg_cel_sweep_results.csv")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["a0"], df["original_avg"], label="Original", linewidth=2.5, alpha=0.8)
    plt.plot(df["a0"], df["restricted_avg"], label="Restricted", linewidth=2.5, alpha=0.8)
    plt.plot(df["a0"], df["unrestricted_avg"], label="Unrestricted", linewidth=2.5, alpha=0.8)

    # Mark the paper's specific setting
    paper_a0 = (D - E) / 2 / D  # = 22.5/100 = 0.225
    plt.axvline(x=paper_a0, color='red', linestyle='--', linewidth=2, alpha=0.6,
                label=f'Paper setting (a0={paper_a0:.3f})')

    plt.xlabel("Manipulator initial claim a0", fontsize=13)
    plt.ylabel("Final reward of manipulator (CG)", fontsize=13)
    plt.title(f"CG (Contested Garment) – Manipulator Reward vs a0\n"
              f"CEL regime (E={E} > D/2={D/2}), n=4, unique combos\n"
              f"weighted_by_multiplicity={WEIGHT_BY_MULTIPLICITY}", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()

    plt.savefig("cg_cel_sweep_a0.png", dpi=200, bbox_inches='tight')
    print(f"✓ Plot saved to: cg_cel_sweep_a0.png")
    plt.show()
