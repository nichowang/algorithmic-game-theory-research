# =================== CEL sweep (max-min differences) ===================
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

def truthful_matrix(vec):
    v = np.asarray(vec, float)
    return np.tile(v, (len(v), 1))

def awards_from_matrix(R, estate=80, scale=100):
    return cel_allocation(R.mean(0) * scale, estate)

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

# ---------- Optimized enumeration (memory efficient) ----------
def compositions_nonneg(free_units, k):
    for cuts in itertools.combinations(range(free_units + k - 1), k - 1):
        add = np.diff((-1, *cuts, free_units + k - 1)) - 1
        yield add

# ---------- Restricted optimization ----------
def restricted_best(R0, m, estate=80, scale=100, alpha=0.01, tol=1e-9):
    n = R0.shape[0]
    fixed = float(R0[m, m])
    others = [i for i in range(n) if i != m]
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        return float("-inf")

    free_units = scale - min_units * k
    best_val = -1.0

    row = np.zeros(n, dtype=float)
    row[m] = fixed
    R = R0.copy()

    for add in compositions_nonneg(free_units, k):
        p_units = np.asarray(add) + min_units
        p = p_units / p_units.sum()
        rem = 1.0 - fixed

        row[others] = p * rem
        R[m] = row

        shares = impartial_division(R)
        val = cel_allocation(shares * scale, estate)[m]
        if val > best_val + tol:
            best_val = val

    return best_val

# ---------- Unrestricted optimization ----------
def unrestricted_best(R0, m, estate=80, scale=100, alpha=0.01, tol=1e-9):
    n = R0.shape[0]
    fixed = float(R0[m, m])
    others = [i for i in range(n) if i != m]
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        return float("-inf")

    free_units = scale - min_units * k
    best_val = -1.0

    row = np.zeros(n, dtype=float)
    row[m] = fixed

    for add in compositions_nonneg(free_units, k):
        p_units = np.asarray(add) + min_units
        p = p_units / p_units.sum()
        rem = 1.0 - fixed

        row[others] = p * rem
        val = cel_allocation(row * scale, estate)[m]
        if val > best_val + tol:
            best_val = val

    return best_val

# ---------- Count unique triples (for progress tracking) ----------
def count_unique_triples(rest_units, min_units=1):
    cnt = 0
    for a1 in range(min_units, rest_units - 2*min_units + 1):
        for a2 in range(a1, rest_units - a1 - min_units + 1):
            a3 = rest_units - a1 - a2
            if a3 >= a2 and a3 >= min_units:
                cnt += 1
    return cnt

# ---------- Main sweep: compute max-min differences ----------
def sweep_minmax_differences_with_progress(E=80, D=100, alpha=0.01, manip_index=0):
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

        # Track min/max for each strategy
        orig_values = []
        r_values = []
        u_values = []

        inner = tqdm(total=total_unique,
                     desc=f"[a0={a0:0.2f}] unique triples",
                     position=1, leave=False, ncols=100,
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        # Pre-allocate arrays for memory efficiency
        v_units = np.zeros(4, dtype=float)
        v_units[0] = a0_units

        for a1_units in range(min_units, rest_units - 2*min_units + 1):
            for a2_units in range(a1_units, rest_units - a1_units - min_units + 1):
                a3_units = rest_units - a1_units - a2_units
                if a3_units < a2_units or a3_units < min_units:
                    continue

                # Reuse array
                v_units[1:] = [a1_units, a2_units, a3_units]
                v = v_units / D
                R0 = truthful_matrix(v)

                # Original: truthful -> CEL
                orig = awards_from_matrix(R0, estate=E, scale=D)[manip_index]
                # Restricted: optimal within constraints -> ID -> CEL
                r_best = restricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha)
                # Unrestricted: optimal -> direct CEL
                u_best = unrestricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha)

                orig_values.append(orig)
                r_values.append(r_best)
                u_values.append(u_best)

                inner.update(1)

        inner.close()

        # Convert to numpy arrays for efficient min/max computation
        orig_arr = np.array(orig_values)
        r_arr = np.array(r_values)
        u_arr = np.array(u_values)

        # Compute differences (max - min for each comparison)
        restricted_orig_diff = r_arr.max() - orig_arr.min()
        unrestricted_orig_diff = u_arr.max() - orig_arr.min()
        unrestricted_restricted_diff = u_arr.max() - r_arr.min()

        rows.append({
            "a0": round(a0, 2),
            "restricted_original": round(restricted_orig_diff, 6),
            "unrestricted_original": round(unrestricted_orig_diff, 6),
            "unrestricted_restricted": round(unrestricted_restricted_diff, 6),
            "unique_cases": total_unique,
        })

    import pandas as pd
    return pd.DataFrame(rows)

# ---------- Run & plot ----------
if __name__ == "__main__":
    E, D, alpha = 80, 100, 0.01
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
    plt.savefig("cel_sweep_a0_minmax_differences.png", dpi=200)
    plt.show()