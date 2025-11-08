import numpy as np
import itertools
from math import comb
from tqdm import tqdm
import matplotlib.pyplot as plt


def cel_allocation(claims, estate, tol=1e-12, iters=60):
    """
    CEL: Start each agent at their claim, then reduce equally until total = estate
    """
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

def awards_from_matrix_CG(R, estate, scale=100):

    shares = impartial_division(R)
    claims = shares * scale
    return contested_garment(claims, estate)


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


def compositions_nonneg(free_units, k):
    for cuts in itertools.combinations(range(free_units + k - 1), k - 1):
        add = np.diff((-1, *cuts, free_units + k - 1)) - 1
        yield add


def restricted_best(R0, m, estate, scale=100, alpha=0.01, tol=1e-9, show_progress=False):
    """
    Manipulator m changes their row, applies impartial division, then CG
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

    # Count total compositions for progress bar
    total_comps = comb(free_units + k - 1, k - 1)

    iterator = compositions_nonneg(free_units, k)
    if show_progress:
        iterator = tqdm(iterator, total=total_comps, desc="  Restricted", leave=False, position=1)

    for add in iterator:
        p_units = np.asarray(add) + min_units
        p = p_units / p_units.sum()
        rem = 1.0 - fixed

        row = np.zeros(n, dtype=float)
        row[m] = fixed
        row[others] = p * rem

        R = R0.copy()
        R[m] = row
        shares = impartial_division(R)
        val = contested_garment(shares * scale, estate)[m]
        if val > best_val + tol:
            best_val = val

    return best_val


def unrestricted_best(R0, m, estate, scale=100, alpha=0.01, tol=1e-9, show_progress=False, manip_claim=None):

    n = R0.shape[0]

    # Get manipulator's fixed share (proportion, not absolute)
    if manip_claim is None:
        shares = impartial_division(R0)
        fixed_share = shares[m]
    else:
        fixed_share = manip_claim / scale

    others = [i for i in range(n) if i != m]
    k = len(others)

    min_share_per_agent = alpha  
    remaining_share = 1.0 - fixed_share 

    if min_share_per_agent * k > remaining_share:
        return float("-inf")

    best_val = -1.0

    remaining_claim_total = scale - manip_claim
    remaining_units = int(round(remaining_claim_total))
    min_units_int = int(round(alpha * scale))

    # Check feasibility
    if min_units_int * k > remaining_units:
        return float("-inf")


    free_units = remaining_units - min_units_int * k
    total_comps = comb(free_units + k - 1, k - 1)

    if show_progress:
        pbar = tqdm(total=total_comps, desc="  Unrestricted", leave=False, position=1)

    # Iterate over all compositions
    for add in compositions_nonneg(free_units, k):
        # Build claims vector (MUST sum to scale)
        claims = np.zeros(n, dtype=float)
        claims[m] = manip_claim


        others_claims = np.asarray(add) + min_units_int
        claims[others] = others_claims.astype(float)

        val = contested_garment(claims, estate)[m]
        if val > best_val + tol:
            best_val = val

        if show_progress:
            pbar.update(1)

    if show_progress:
        pbar.close()

    return best_val

def run_cel_experiment(E=55, D=100, n_values=[4, 5, 6, 7], alpha=0.01, manip_index=0):
    """
    For CEL regime (E > D/2):
    Starting division: manipulator gets (D-E)/2,
                       one agent gets (D+E)/2 - (n-2)*0.01,
                       others get 0.01
    """
    results = []

    for n in tqdm(n_values, desc="Testing n values"):
      
        manip_share = (D - E) / (2 * D)  # (D-E)/2 out of D
        min_share = 0.01

        remainder_share = ((D + E) / 2 - (n - 2) * 0.01) / D

        # Check if valid
        if remainder_share < min_share or manip_share < 0:
            print(f"Invalid configuration for n={n}")
            continue

        claims = np.zeros(n)
        claims[manip_index] = (D - E) / 2  # Absolute value
        claims[1] = (D + E) / 2 - (n - 2) * 0.01
        for i in range(2, n):
            claims[i] = 0.01

        print(f"\n{'='*60}")
        print(f"n={n}")
        print(f"{'='*60}")
        print(f"Input claims: {claims}")

        R0 = truthful_matrix(claims)

        # ORIGINAL
        print(f"\nORIGINAL:")
        shares_orig = impartial_division(R0)
        print(f"  ID shares: {shares_orig}")
        claims_from_id = shares_orig * D
        print(f"  Claims: {claims_from_id}")
        awards_orig = contested_garment(claims_from_id, E)
        print(f"  Awards: {awards_orig}")
        orig = awards_orig[manip_index]
        print(f"  → Manipulator gets: {orig:.6f}")

        # RESTRICTED
        print(f"\nRESTRICTED (test 1 example):")
        R_restricted_example = R0.copy()
        example_row = np.zeros(n)
        example_row[1] = 0.97
        for i in range(n):
            if i != manip_index and i != 1:
                example_row[i] = 0.01
        example_row[manip_index] = 0.0
        R_restricted_example[manip_index] = example_row

        shares_restr = impartial_division(R_restricted_example)
        print(f"  Manipulated R[{manip_index}]: {example_row}")
        print(f"  ID shares: {shares_restr}")
        claims_restr = shares_restr * D
        print(f"  Claims: {claims_restr}")
        awards_restr = contested_garment(claims_restr, E)
        print(f"  Awards: {awards_restr}")
        print(f"  → Manipulator gets: {awards_restr[manip_index]:.6f}")

        print(f"\nSearching restricted manipulations...")
        r_best = restricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha, show_progress=True)
        print(f"  → Best: {r_best:.6f}")

        # UNRESTRICTED
        print(f"\nUNRESTRICTED (test 2 examples):")
        remaining_total = D - claims[manip_index]
        min_units = int(round(alpha * D))

        # Test 1: equal
        claims_test = np.zeros(n)
        claims_test[manip_index] = claims[manip_index]
        for i, idx in enumerate([j for j in range(n) if j != manip_index]):
            claims_test[idx] = remaining_total / 3
        print(f"  Equal: claims={claims_test}")
        awards_test = contested_garment(claims_test, E)
        print(f"         awards={awards_test}, manip={awards_test[manip_index]:.6f}")

        # Test 2: extreme
        claims_test2 = np.zeros(n)
        claims_test2[manip_index] = claims[manip_index]
        claims_test2[1] = remaining_total - 2*min_units
        claims_test2[2] = min_units
        claims_test2[3] = min_units
        print(f"  Extreme: claims={claims_test2}")
        awards_test2 = contested_garment(claims_test2, E)
        print(f"           awards={awards_test2}, manip={awards_test2[manip_index]:.6f}")

        k = n - 1
        free_units = int(round(remaining_total)) - min_units * k
        total_combs = comb(free_units + k - 1, k - 1)

        print(f"\nSearching {total_combs} unrestricted manipulations...")
        u_best = unrestricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha, show_progress=True, manip_claim=claims[manip_index])
        print(f"  → Best: {u_best:.6f}")

        results.append({
            'n': n,
            'original': round(orig, 6),
            'restricted': round(r_best, 6),
            'unrestricted': round(u_best, 6),
        })

    return results


def plot_results(results, E, D):
    n_vals = [r['n'] for r in results]
    orig_vals = [r['original'] for r in results]
    rest_vals = [r['restricted'] for r in results]
    unrest_vals = [r['unrestricted'] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(n_vals, orig_vals, 'o-', label='Original', linewidth=2, markersize=8)
    plt.plot(n_vals, rest_vals, 's-', label='Restricted', linewidth=2, markersize=8)
    plt.plot(n_vals, unrest_vals, '^-', label='Unrestricted', linewidth=2, markersize=8)

    plt.xlabel('Number of agents (n)', fontsize=12)
    plt.ylabel('Final reward of manipulator (CEL)', fontsize=12)
    plt.title(f'CEL Manipulator Reward vs n (E={E}, D={D})\n' +
              f'Starting: manipulator=({D}-{E})/2, one agent≈({D}+{E})/2, others=0.01',
              fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xticks(n_vals)
    plt.tight_layout()
    plt.savefig('CEL_CG_manip_vs_n.png', dpi=200)
    plt.show()


if __name__ == "__main__":
    E, D = 55, 100
    n_values = [4]  

    print(f"Running CEL Contested Garment experiment:")
    print(f"D={D}, E={E} (CEL regime since E > D/2={D/2})")
    print(f"Testing n = {n_values}")
    print(f"Starting division: manipulator={(D-E)/2}, one agent≈{(D+E)/2}, others=0.01\n")

    results = run_cel_experiment(E=E, D=D, n_values=n_values)

    print("\n=== Final Results ===")
    for r in results:
        print(f"n={r['n']}: Original={r['original']}, Restricted={r['restricted']}, Unrestricted={r['unrestricted']}")

    plot_results(results, E, D)
