# =================== TALMUD / CEL sweep (E=51 only, fast version) ===================
import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt

# ======= Global switches =======
WEIGHT_BY_MULTIPLICITY = True
USE_TALMUD = True          # True = full Talmud rule; False = pure CEL

# ======= Fast allocators (water-filling; no iterations) =======
def cel_allocation_fast(claims, estate):
    """
    CEL: x_i = max(0, c_i - λ), with sum(x) = estate.
    O(n log n) via sorting. No global scaling, exact within float.
    """
    c = np.asarray(claims, dtype=np.float64)
    C = float(c.sum())
    if estate <= 0: return np.zeros_like(c)
    if estate >= C: return c.copy()

    # Sort descending
    idx = np.argsort(-c)
    c_sorted = c[idx]
    cumsum = np.cumsum(c_sorted)

    # Find k s.t. λ ∈ [c_{k+1}, c_k], λ = (sum_{i<=k} c_i - estate)/k
    n = c_sorted.size
    for k in range(1, n + 1):
        lam = (cumsum[k-1] - estate) / k
        # next value after top-k (or -inf if k==n)
        next_val = c_sorted[k] if k < n else -np.inf
        if (k == n and lam <= c_sorted[k-1]) or (c_sorted[k-1] >= lam >= next_val):
            x_sorted = np.maximum(0.0, c_sorted - lam)
            x = np.empty_like(x_sorted)
            x[idx] = x_sorted
            return x

    # Fallback (should not hit): numerical safety
    lam = (C - estate) / n
    x = np.maximum(0.0, c - lam)
    return x

def cea_allocation_fast(claims, estate):
    """
    CEA: x_i = min(c_i, λ), with sum(x) = estate.
    O(n log n) via sorting. No global scaling, exact within float.
    """
    c = np.asarray(claims, dtype=np.float64)
    C = float(c.sum())
    if estate <= 0: return np.zeros_like(c)
    if estate >= C: return c.copy()

    # Sort ascending
    idx = np.argsort(c)
    c_sorted = c[idx]
    cumsum = np.cumsum(c_sorted)

    n = c_sorted.size
    for k in range(1, n + 1):
        # suppose first k are below λ, others capped at λ
        # sum(min(c, λ)) = cumsum[k-1] + (n-k)*λ = estate
        denom = n - k + 1
        lam = (estate - cumsum[k-1]) / denom
        prev_val = c_sorted[k-1]
        next_val = c_sorted[k] if k < n else np.inf
        if prev_val <= lam <= next_val:
            x_sorted = np.minimum(c_sorted, lam)
            x = np.empty_like(x_sorted)
            x[idx] = x_sorted
            return x

    # Fallback (should not hit)
    lam = estate / n
    x = np.minimum(c, lam)
    return x

# ---------- Talmud (Aumann–Maschler) via CEA/CEL ----------
def talmud_allocation_via_cel(claims, estate):
    """
    Let C = sum(c).
    If E <= C/2: x = 2 * CEA(c/2, E).
    If E >  C/2: x = c/2 + CEL(c/2, E - C/2).
    """
    c = np.asarray(claims, dtype=np.float64)
    C = float(c.sum())
    if estate <= 0: return np.zeros_like(c)
    if estate >= C: return c.copy()

    half = 0.5 * C
    c_half = 0.5 * c
    if estate <= half:
        x_half = cea_allocation_fast(c_half, estate)
        x = 2.0 * x_half
        return np.clip(x, 0.0, c)
    else:
        E_res = estate - half
        y = cel_allocation_fast(c_half, E_res)
        x = c_half + y
        return np.clip(x, 0.0, c)

def _alloc(claims, estate):
    return talmud_allocation_via_cel(claims, estate) if USE_TALMUD else cel_allocation_fast(claims, estate)

# ---------- Truthful matrix + ID ----------
def truthful_matrix(vec):
    v = np.asarray(vec, dtype=np.float64)
    return np.tile(v, (len(v), 1))

def awards_from_matrix(R, estate=60, scale=100):
    claims = R.mean(0) * scale
    return _alloc(claims, estate)

def impartial_division(R, eps=1e-9):
    R, n = np.asarray(R, dtype=np.float64), R.shape[0]
    phi = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            if a == b: continue
            s = 0.0; cnt = 0
            for l in range(n):
                if l == a or l == b: continue
                if R[l, b] > eps:
                    s += R[l, a] / (R[l, b] + eps)
                    cnt += 1
            phi[a, b] = (s / cnt) if cnt else 1.0

    def psi(res, k, i):
        s = 0.0; cnt = 0
        for l in range(n):
            if l == res or l == k or l == i: continue
            if R[l, i] > eps:
                s += R[l, k] / (R[l, i] + eps)
                cnt += 1
        return (s / cnt) if cnt else 1.0

    total = np.zeros(n, dtype=np.float64)
    for j in range(n):
        f = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == j: continue
            s = 1.0 + phi[j, i]
            for k in range(n):
                if k == i or k == j: continue
                s += psi(j, k, i)
            f[i] = 1.0 / s
        f[j] = 1.0 - f.sum()
        total += f
    shares = total / n
    return shares / shares.sum()

# ---------- Precompute and cache compositions (major speedup) ----------
# For your current params: scale=100, alpha=0.01 => min_units=1; k=3 others; free_units=97
# We'll build a cache dict keyed by (free_units, k).
_COMPOSITIONS_CACHE = {}
def get_compositions_cached(free_units, k):
    key = (free_units, k)
    if key in _COMPOSITIONS_CACHE:
        return _COMPOSITIONS_CACHE[key]
    # build once
    arr = []
    for cuts in itertools.combinations(range(free_units + k - 1), k - 1):
        add = np.diff((-1, *cuts, free_units + k - 1)) - 1
        arr.append(add)
    comp = np.asarray(arr, dtype=np.int32)  # shape: [num, k]
    _COMPOSITIONS_CACHE[key] = comp
    return comp

# ---------- Restricted: manip row with column floor alpha, ID -> allocator ----------
def restricted_best(R0, m, estate=60, scale=100, alpha=0.01, tol=1e-9):
    n = R0.shape[0]
    fixed = float(R0[m, m])
    others = [i for i in range(n) if i != m]
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        return float("-inf")

    free_units = scale - min_units * k
    comp = get_compositions_cached(free_units, k)  # cached array [num, k]

    best_val = -1.0
    rem = 1.0 - fixed

    row = np.zeros(n, dtype=np.float64)
    for add in comp:
        p_units = add + min_units
        p = p_units / p_units.sum()
        row.fill(0.0)
        row[m] = fixed
        row[others] = p * rem

        R = R0.copy()
        R[m] = row
        shares = impartial_division(R)
        val = _alloc(shares * scale, estate)[m]
        if val > best_val + tol:
            best_val = val
    return best_val

# ---------- Unrestricted: manip row -> direct allocator ----------
def unrestricted_best(R0, m, estate=60, scale=100, alpha=0.01, tol=1e-9):
    n = R0.shape[0]
    fixed = float(R0[m, m])
    others = [i for i in range(n) if i != m]
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        return float("-inf")

    free_units = scale - min_units * k
    comp = get_compositions_cached(free_units, k)

    best_val = -1.0
    rem = 1.0 - fixed

    row = np.zeros(n, dtype=np.float64)
    for add in comp:
        p_units = add + min_units
        p = p_units / p_units.sum()
        row.fill(0.0)
        row[m] = fixed
        row[others] = p * rem

        val = _alloc(row * scale, estate)[m]
        if val > best_val + tol:
            best_val = val
    return best_val

# ---------- Multiplicity for (a1,a2,a3) with manipulator fixed (n=4 => 1/3/6) ----------
@np.vectorize
def multiplicity_three(a1, a2, a3):
    if a1 == a2 == a3:
        return 1
    elif a1 == a2 or a2 == a3:
        return 3
    else:
        return 6

# ---------- Count unique triples (for tqdm total) ----------
def count_unique_triples(rest_units, min_units=1):
    cnt = 0
    for a1 in range(min_units, rest_units - 2*min_units + 1):
        for a2 in range(a1, rest_units - a1 - min_units + 1):
            a3 = rest_units - a1 - a2
            if a3 >= a2 and a3 >= min_units:
                cnt += 1
    return cnt

# ---------- Sweep a0 in 0.01..0.97 for fixed E=51 ----------
def sweep_unique_combinations_with_progress(E=60, D=100, alpha=0.01, manip_index=0):
    n = 4
    min_units = int(round(alpha * D))  # =1 when alpha=0.01 & D=100
    rows = []

    outer_total = (D - 3*min_units) - 1 + 1   # 0.01..0.97 => 97 steps when D=100
    outer = tqdm(range(1, D - 3*min_units + 1),
                 desc=f"[a0] 0.01→0.97 (E={E}, rule={'Talmud' if USE_TALMUD else 'CEL'})",
                 total=outer_total, position=0, leave=True, ncols=100)

    # compositions for restricted/unrestricted are constant across a0; cache once
    # (already cached by get_compositions_cached inside)

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

                v_units = np.array([a0_units, a1_units, a2_units, a3_units], dtype=np.float64)
                v = v_units / D
                R0 = truthful_matrix(v)

                # Original: truthful (row-averaged shares) -> allocator
                orig = awards_from_matrix(R0, estate=E, scale=D)[manip_index]
                # Restricted: best manip row under column floors -> ID -> allocator
                r_best = restricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha)
                # Unrestricted: best manip row -> direct allocator
                u_best = unrestricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha)

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

# ---------- Run: E=51 only ----------
if __name__ == "__main__":
    D, alpha, manip_index = 100, 0.01, 0
    E = 60

    df = sweep_unique_combinations_with_progress(E=E, D=D, alpha=alpha, manip_index=manip_index)
    print(df.head()); print(df.tail())

    plt.figure(figsize=(8, 5))
    plt.plot(df["a0"], df["original_avg"],     label="Original",     linewidth=2)
    plt.plot(df["a0"], df["restricted_avg"],   label="Restricted",   linewidth=2)
    plt.plot(df["a0"], df["unrestricted_avg"], label="Unrestricted", linewidth=2)
    plt.xlabel("Manipulator initial claim a0")
    plt.ylabel("Final reward of manipulator")
    plt.title(f"{'TALMUD' if USE_TALMUD else 'CEL'} – Manipulator Reward vs a0 "
              f"(unique combos; E={E}, D={D}, n=4)\nweighted_by_multiplicity={WEIGHT_BY_MULTIPLICITY}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    tag = "talmud" if USE_TALMUD else "cel"
    plt.savefig(f"{tag}_sweep_a0_E{E}_D{D}.png", dpi=200)
    plt.show()
