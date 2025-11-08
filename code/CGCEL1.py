# =================== CG-CEL: n from 4 to 7 with fixed setting ===================
import numpy as np
import itertools
from math import comb
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------- Core: CEL ----------
def cel_allocation(claims, estate, tol=1e-12, iters=60):
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
    # scale to match the exact estate in case of small numeric drift
    if a.sum() > 0:
        a *= estate / a.sum()
    return a

# ---------- CG-CEL: Conditional Grant version ----------
def cg_cel_allocation(claims, estate):
    """
    CG-CEL: First run CEL on half the estate (D/2),
    then run CEL on the rest with remaining claims.
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

def truthful_matrix(vec):
    v = np.asarray(vec, float)
    return np.tile(v, (len(v), 1))

def awards_from_matrix(R, estate=60, scale=100):
    return cg_cel_allocation(R.mean(0) * scale, estate)

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

# ---------- Enumerate all k-part compositions of free_units (nonnegative parts) ----------
def compositions_nonneg(free_units, k):
    for cuts in itertools.combinations(range(free_units + k - 1), k - 1):
        add = np.diff((-1, *cuts, free_units + k - 1)) - 1
        yield add

# ---------- Restricted: manipulator allocates within row (col floor alpha), ID -> CG-CEL ----------
def restricted_best(R0, m, estate=60, scale=100, alpha=0.01, tol=1e-9):
    n = R0.shape[0]
    fixed = float(R0[m, m])
    others = [i for i in range(n) if i != m]
    k = len(others)

    min_units = int(round(alpha * scale))  # =1 when alpha=0.01 & scale=100
    if min_units * k > scale:
        return float("-inf")

    free_units = scale - min_units * k
    best_val = -1.0

    for add in compositions_nonneg(free_units, k):
        p_units = np.asarray(add) + min_units         # at least 1 unit per other column
        p = p_units / p_units.sum()                   # normalize to 1
        rem = 1.0 - fixed

        row = np.zeros(n, dtype=float)
        row[m] = fixed
        row[others] = p * rem

        R = R0.copy(); R[m] = row
        shares = impartial_division(R)
        val = cg_cel_allocation(shares * scale, estate)[m]  # Use CG-CEL
        if val > best_val:
            best_val = val

    return best_val

# ---------- Unrestricted: manipulator's row directly optimized under CG-CEL ----------
def unrestricted_best(R0, m, estate=60, scale=100, alpha=0.01, tol=1e-9):
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

        val = cg_cel_allocation(row * scale, estate)[m]  # Use CG-CEL
        if val > best_val:
            best_val = val

    return best_val

# ---------- Main: test for n=4,5,6,7 with setting (D-E, E-(n-1)*0.01, 0.01, ..., 0.01) ----------
def run_cg_cel_experiments(E=60, D=100, alpha=0.01, manip_index=0):
    ns = [4]  # Test only n=4 first for debugging
    results = []

    print(f"Running CG-CEL experiments for n = {ns}")
    print(f"Setting: (D-E, E-(n-1)*0.01, 0.01, ..., 0.01)")
    print(f"E={E}, D={D}, alpha={alpha}")
    print("-" * 80)

    for n in tqdm(ns, desc="Testing n values"):
        # CG-CEL setting: (D-E, E-(n-1)*0.01, 0.01, ..., 0.01)
        d_manip = D - E
        d_second = E - (n - 1) * 0.01
        claims = [d_manip, d_second] + [0.01] * (n - 2)

        v = np.array(claims, dtype=float) / D
        R0 = truthful_matrix(v)

        # Original: truthful (row-averaged shares) -> CG-CEL
        orig = awards_from_matrix(R0, estate=E, scale=D)[manip_index]

        # Restricted: best manip row under column floors -> ID -> CG-CEL
        r_best = restricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha)

        # Unrestricted: best manip row -> direct CG-CEL
        u_best = unrestricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha)

        results.append({
            "n": n,
            "original": round(orig, 6),
            "restricted": round(r_best, 6),
            "unrestricted": round(u_best, 6),
        })

        print(f"n={n}: Original={orig:.6f}, Restricted={r_best:.6f}, Unrestricted={u_best:.6f}")
        print(f"  Initial claims: {claims}")
        print(f"  Restricted >= Unrestricted? {r_best >= u_best - 1e-6}")

    import pandas as pd
    return pd.DataFrame(results)

# ---------- Run & plot ----------
if __name__ == "__main__":
    E, D, alpha = 60, 100, 0.01
    df = run_cg_cel_experiments(E=E, D=D, alpha=alpha, manip_index=0)

    print("\n" + "="*80)
    print("Results Summary:")
    print(df.to_string(index=False))
    print("="*80)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df["n"], df["original"],     marker='o', label="Original",     linewidth=2)
    plt.plot(df["n"], df["restricted"],   marker='s', label="Restricted",   linewidth=2)
    plt.plot(df["n"], df["unrestricted"], marker='^', label="Unrestricted", linewidth=2)
    plt.xlabel("Number of agents (n)")
    plt.ylabel("Final reward of manipulator (CG-CEL)")
    plt.title(f"CG-CEL – Manipulator Reward vs n (E={E}, D={D})\n"
              f"Setting: (D-E, E-(n-1)*0.01, 0.01, ..., 0.01)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cg_cel_n_4_to_7.png", dpi=200)
    print(f"\n✅ Plot saved as: cg_cel_n_4_to_7.png")
    plt.show()
