import math, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
from functools import lru_cache
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

# -------------------- Core methods --------------------
def cea_allocation(claims, estate, iters=60):
    """CG-CEA: halve claims first, then run standard CEA"""
    c = np.asarray(claims, dtype=np.float64) / 2.0  # Halve for CG
    total = c.sum()

    if total <= estate:
        base = c.copy()
        remaining = estate - total
        equal_increase = remaining / len(c)
        return base + equal_increase

    lo, hi = 0.0, float(c.max())
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        if np.minimum(c, lam).sum() > estate:
            hi = lam
        else:
            lo = lam
    return np.minimum(c, hi)

def truthful_matrix(claims):
    claims = np.asarray(claims, dtype=np.float64)
    n = len(claims)
    total = claims.sum()
    R = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        others_total = total - claims[i]
        if others_total > 0:
            for j in range(n):
                if i != j:
                    R[i, j] = claims[j] / others_total
    return R

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
    n = int(np.sqrt(len(R_tuple)))
    R = np.array(R_tuple, dtype=np.float64).reshape(n, n)
    return tuple(_impartial_division_core(R, eps))

def impartial_division(R, eps=1e-9):
    return np.array(impartial_division_cached(tuple(R.flatten()), eps))

# -------------------- Fast candidate search --------------------
def fast_restricted_search(R0, m, estate, scale=100, alpha=0.01):
    """Fast search using strategic candidates"""
    n = R0.shape[0]
    k = n - 1
    min_val = alpha

    # Truthful baseline
    shares_truth = impartial_division(R0)
    best_val = float(cea_allocation(shares_truth * scale, estate)[m])

    # Candidate strategies
    candidates = []

    # 1. Equal distribution
    row = np.ones(n) * ((1.0 - 0) / k)
    row[m] = 0
    candidates.append(row)

    # 2. One-heavy: give most to one agent
    for target in range(n):
        if target == m:
            continue
        row = np.ones(n) * min_val
        row[m] = 0
        remaining = 1.0 - min_val * (k - 1)
        row[target] = remaining
        candidates.append(row)

    # 3. Two-heavy: split between two agents
    for i in range(n):
        for j in range(i+1, n):
            if i == m or j == m:
                continue
            row = np.ones(n) * min_val
            row[m] = 0
            remaining = 1.0 - min_val * (k - 2)
            row[i] = remaining / 2
            row[j] = remaining / 2
            candidates.append(row)

    # Test all candidates
    for row in candidates:
        R_test = R0.copy()
        R_test[m] = row
        shares = impartial_division(R_test)
        val = cea_allocation(shares * scale, estate)[m]
        if val > best_val:
            best_val = val

    return best_val

def fast_unrestricted_search(manip_claim, estate, scale=100, alpha=0.01, n=4, manip_index=0):
    """
    Unrestricted search: manipulator's claim is FIXED (from impartial division),
    but can arbitrarily redistribute the remaining (scale - manip_claim) among others.
    """
    min_units = alpha * scale
    remaining = scale - manip_claim
    k = n - 1
    others = [i for i in range(n) if i != manip_index]

    # Truthful baseline (equal)
    claims = np.zeros(n)
    claims[manip_index] = manip_claim
    claims[others] = remaining / k
    best_val = float(cea_allocation(claims, estate)[manip_index])

    # Candidate strategies
    candidates = []

    # 1. Equal
    claims = np.zeros(n)
    claims[manip_index] = manip_claim
    claims[others] = remaining / k
    candidates.append(claims.copy())

    # 2. One-heavy: give most to one other agent
    for target in others:
        claims = np.zeros(n)
        claims[manip_index] = manip_claim
        claims[others] = min_units
        claims[target] = remaining - min_units * (k - 1)
        if claims[target] >= min_units:
            candidates.append(claims.copy())

    # 3. Two-heavy: split between two agents
    for i in range(len(others)):
        for j in range(i+1, len(others)):
            claims = np.zeros(n)
            claims[manip_index] = manip_claim
            claims[others] = min_units
            remaining_split = remaining - min_units * (k - 2)
            claims[others[i]] = remaining_split / 2
            claims[others[j]] = remaining_split / 2
            if claims[others[i]] >= min_units:
                candidates.append(claims.copy())

    # 4. Gradual distributions
    if len(others) >= 2:
        for frac in [0.6, 0.7, 0.8]:
            claims = np.zeros(n)
            claims[manip_index] = manip_claim
            claims[others] = min_units
            claims[others[0]] = (remaining - min_units * (k - 1)) * frac + min_units
            claims[others[1]] = remaining - sum(claims[others])
            if claims[others[1]] >= min_units:
                candidates.append(claims.copy())

    # Test all candidates
    for claims in candidates:
        if abs(claims.sum() - scale) > 1e-6:
            continue
        val = cea_allocation(claims, estate)[manip_index]
        if val > best_val:
            best_val = val

    return best_val

# -------------------- Main experiment --------------------
def run_experiment():
    E_FIXED = 45
    D_FIXED = 100
    SCALE = 100
    ALPHA = 0.01

    n_values = [4, 5, 6, 7]
    results = []

    print(f"CG-CEA Experiment (Fast Search - Corrected Maximum Manipulability)")
    print(f"D={D_FIXED}, E={E_FIXED} (CEA regime since E <= D/2={D_FIXED/2})")
    print(f"=" * 60)

    for n in n_values:
        print(f"\nn={n}")

        # CORRECTED Setup for maximum manipulability in CEA (E <= D/2):
        # unanimous vector: (2E/n, ..., 2E/n, D - 2(n-1)E/n)
        # n-1 agents get 2E/n each, manipulator gets D - 2(n-1)E/n
        equal_share = 2 * E_FIXED / n
        manip_share = D_FIXED - 2 * (n - 1) * E_FIXED / n

        claims = [equal_share] * (n - 1) + [manip_share]
        m = n - 1  # manipulator is the last agent (biggest claim)

        claims_arr = np.array(claims, dtype=np.float64)
        R_truth = truthful_matrix(claims_arr)

        print(f"  Starting claims: {claims_arr}")
        print(f"  Manipulator claim: {manip_share:.6f}")

        # Original
        shares_orig = impartial_division(R_truth)
        awards_orig = cea_allocation(shares_orig * SCALE, E_FIXED)
        orig = float(awards_orig[m])
        print(f"  Original:      {orig:.6f}")

        # Restricted (fast search)
        rest_best = fast_restricted_search(R_truth, m, E_FIXED, SCALE, ALPHA)
        print(f"  Restricted:    {rest_best:.6f}")

        # Unrestricted (fast search)
        unbd_best = fast_unrestricted_search(manip_share, E_FIXED, SCALE, ALPHA, n, manip_index=m)
        print(f"  Unrestricted:  {unbd_best:.6f}")

        results.append({
            'n': n,
            'original': orig,
            'restricted': rest_best,
            'unrestricted': unbd_best
        })

    return results

def plot_results(results):
    ns = [r['n'] for r in results]
    orig = [r['original'] for r in results]
    rest = [r['restricted'] for r in results]
    unbd = [r['unrestricted'] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(ns, orig, 'o-', linewidth=2, markersize=8, label='Original')
    plt.plot(ns, rest, 's--', linewidth=2, markersize=8, label='Restricted')
    plt.plot(ns, unbd, '^:', linewidth=2, markersize=8, label='Unrestricted')

    plt.xlabel('Number of agents (n)', fontsize=12)
    plt.ylabel('Manipulator payoff', fontsize=12)
    plt.title('CG-CEA: Manipulator Payoff vs n (E=45, D=100, Corrected)', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(ns)
    plt.tight_layout()

    plt.savefig('CEA_fast1_n4567.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('CEA_fast1_n4567.png', dpi=200, bbox_inches='tight')
    print(f"\n✅ Saved: CEA_fast1_n4567.pdf & CEA_fast1_n4567.png")

if __name__ == "__main__":
    results = run_experiment()

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'n':<5} {'Original':<12} {'Restricted':<12} {'Unrestricted':<12}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['n']:<5} {r['original']:<12.6f} {r['restricted']:<12.6f} {r['unrestricted']:<12.6f}")

    plot_results(results)
