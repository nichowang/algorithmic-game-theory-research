import numpy as np
import matplotlib.pyplot as plt
from math import comb

# -------------------- Core methods --------------------
def cel_allocation(claims, estate, tol=1e-12, iters=60):
    """CEL: reduce equally from full claim until total = estate"""
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

def contested_garment(claims, estate):
    """
    Contested Garment Rule (CG):
    - If E <= D/2: give CEA on claims/2
    - If E > D/2: give each agent claims/2, then run CEL on the remainder
    """
    c = np.asarray(claims, float)
    D = c.sum()

    if estate > D/2:
        # CEL regime: give half first, then CEL on remainder
        half_claims = c / 2
        remaining_estate = estate - D/2
        cel_part = cel_allocation(half_claims, remaining_estate)
        return half_claims + cel_part
    else:
        raise ValueError("This script is for CEL regime (E > D/2)")

def truthful_matrix(claims):
    claims = np.asarray(claims, float)
    n = len(claims)
    total = claims.sum()
    R = np.zeros((n, n))

    for i in range(n):
        others_total = total - claims[i]
        if others_total > 0:
            for j in range(n):
                if i != j:
                    R[i, j] = claims[j] / others_total
    return R

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

    def psi(residual, k, i):
        reporters = [l for l in range(n)
                     if l not in (residual, k, i) and R[l, i] > eps]
        if not reporters:
            return 1.0
        vals = [R[l, k] / (R[l, i] + eps) for l in reporters]
        return sum(vals) / len(vals)

    total = np.zeros(n)
    for j in range(n):
        f = np.zeros(n)
        for i in range(n):
            if i == j:
                continue
            term1 = phi[j, i]
            term2 = sum(psi(j, k, i) for k in range(n)
                         if k not in (i, j))
            f[i] = 1.0 / (1.0 + term1 + term2)
        f[j] = 1.0 - f.sum()
        total += f

    shares = total / n
    shares = np.maximum(shares, 0.0)
    shares /= shares.sum()
    return shares

# -------------------- Fast candidate search --------------------
def fast_restricted_search(R0, m, estate, scale=100, alpha=0.01):
    """Fast search using strategic candidates"""
    n = R0.shape[0]
    k = n - 1
    min_val = alpha

    # Truthful baseline
    shares_truth = impartial_division(R0)
    claims_truth = shares_truth * scale
    best_val = float(contested_garment(claims_truth, estate)[m])

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
        claims = shares * scale
        val = contested_garment(claims, estate)[m]
        if val > best_val:
            best_val = val

    return best_val

def fast_unrestricted_search(manip_claim, estate, scale=100, alpha=0.01, n=4, manip_index=0):

    min_units = alpha * scale
    remaining = scale - manip_claim
    k = n - 1

    # Baseline: equal distribution
    claims = np.zeros(n)
    claims[manip_index] = manip_claim
    others = [i for i in range(n) if i != manip_index]
    claims[others] = remaining / k
    best_val = float(contested_garment(claims, estate)[manip_index])

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
        for frac in [0.5, 0.6, 0.7, 0.8, 0.9]:
            claims = np.zeros(n)
            claims[manip_index] = manip_claim
            claims[others] = min_units
            heavy_amount = (remaining - min_units * (k - 1)) * frac + min_units
            claims[others[0]] = heavy_amount
            claims[others[1]] = remaining - sum(claims[others])
            if claims[others[1]] >= min_units:
                candidates.append(claims.copy())


    for claims in candidates:
        if abs(claims.sum() - scale) > 1e-6:
            continue
        val = contested_garment(claims, estate)[manip_index]
        if val > best_val:
            best_val = val

    return best_val

# -------------------- Main experiment --------------------
def run_experiment():
    E = 60
    D = 100
    alpha = 0.01
    n_values = [4, 5, 6, 7]

    results = []

    print(f"CG-CEL Experiment (Fast Search - Corrected Maximum Manipulability)")
    print(f"D={D}, E={E} (CEL regime since E > D/2)")
    print(f"=" * 60)

    for n in n_values:
        print(f"\nn={n}")

        # CORRECTED Setup for maximum manipulability in CEL (E > D/2):
        # For CEL, after CG we have: remaining_estate = E - D/2
        # Unanimous vector for CEL should be analogous to CEA:
        # After halving: (2E'/n, 2E'/n, ..., 2E'/n, D/2 - (n-1)*2E'/n)
        # where E' = E - D/2
        # Before halving (original claims): (2*2E'/n, ..., 2*2E'/n, D - 2(n-1)*2E'/n)
        # = (4E'/n, ..., 4E'/n, D - 4(n-1)E'/n)
        # Manipulator has SMALLEST claim (benefits from CEL)

        E_prime = E - D/2  # remaining estate after CG step 1
        equal_share = 4 * E_prime / n  # smallest claims
        manip_share = D - (n - 1) * equal_share  # largest claim

        # Put manipulator at index 0 (smallest claim)
        manip_index = 0
        claims = np.zeros(n)
        for i in range(n - 1):
            claims[i] = equal_share  # n-1 agents with small equal claims
        claims[n - 1] = manip_share  # 1 agent with large claim

        # Actually, manipulator should have smallest, so swap
        claims[manip_index] = equal_share
        # Wait, they're all equal except last one is largest
        # Manipulator (smallest) should be in first n-1, not last
        # Let me reorganize: manip is one of the equal small claims
        # So claims = (equal, equal, ..., equal, manip_share)
        # But manipulator has smallest, so it's in the first n-1

        # Simpler: just make manip_index point to one of the small claims
        claims = np.array([equal_share] * (n-1) + [manip_share])
        manip_index = 0  # manipulator is first agent with smallest claim

        R0 = truthful_matrix(claims)

        print(f"  Starting claims: {claims}")
        print(f"  Manipulator claim (smallest): {claims[manip_index]:.6f}")
        print(f"  After halving: {claims/2}")
        print(f"  Remaining estate for CEL: {E - D/2}")


        shares = impartial_division(R0)
        claims_id = shares * D
        awards = contested_garment(claims_id, E)
        orig = float(awards[manip_index])
        print(f"  Original:      {orig:.6f}")


        rest_best = fast_restricted_search(R0, manip_index, E, D, alpha)
        print(f"  Restricted:    {rest_best:.6f}")


        unbd_best = fast_unrestricted_search(claims[manip_index], E, D, alpha, n, manip_index)
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
    plt.title('CG-CEL: Manipulator Payoff vs n (E=60, D=100, Corrected)', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(ns)
    plt.tight_layout()

    plt.savefig('CEL_fast1_n4567.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('CEL_fast1_n4567.png', dpi=200, bbox_inches='tight')
    print(f"\n✅ Saved: CEL_fast1_n4567.pdf & CEL_fast1_n4567.png")

if __name__ == "__main__":
    results = run_experiment()

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'n':<5} {'Original':<12} {'Restricted':<12} {'Unrestricted':<12}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['n']:<5} {r['original']:<12.6f} {r['restricted']:<12.6f} {r['unrestricted']:<12.6f}")

    plot_results(results)
