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

def awards_from_matrix(R, estate=35, scale=100):

    shares = impartial_division(R)
    return cea_allocation(shares * scale, estate)

# -------- Optimized impartial_division with numba --------
@njit(cache=True)
def _impartial_division_core(R, eps=1e-9):
    """Numba-optimized core computation"""
    n = R.shape[0]
    phi = np.ones((n, n), dtype=np.float64)

    # Compute phi
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

    # Compute shares
    total = np.zeros(n, dtype=np.float64)
    for j in range(n):
        f = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == j:
                continue
            s = 1.0 + phi[j, i]
            # Add psi terms
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
    """Cached version - takes tuple instead of array"""
    n = int(np.sqrt(len(R_tuple)))
    R = np.array(R_tuple, dtype=np.float64).reshape(n, n)
    return tuple(_impartial_division_core(R, eps))

def impartial_division(R, eps=1e-9):
    """Wrapper that converts array to tuple for caching"""
    return np.array(impartial_division_cached(tuple(R.flatten()), eps))


def compositions_nonneg_iter(total, parts, out_arr=None):
    if parts <= 0:
        return
    if out_arr is None:
        x = np.zeros(parts, dtype=np.int32)
    else:
        x = out_arr
        x.fill(0)

    def rec(pos, rem):
        if pos == parts - 1:
            x[pos] = rem
            yield x
        else:
            for v in range(rem + 1):
                x[pos] = v
                yield from rec(pos + 1, rem - v)
    yield from rec(0, total)

def n_compositions(free_units, k):
    return math.comb(free_units + k - 1, k - 1)

def all_opt_rows_restricted(R0, m, estate=20, scale=100, alpha=0.01, pbar=None):
    n = R0.shape[0]
    fixed = float(R0[m, m])
    others = np.array([i for i in range(n) if i != m])
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        if pbar: pbar.close()
        return float('-inf'), [], []

    free_units = scale - min_units * k


    truthful_awards = cea_allocation(R0.mean(axis=0) * scale, estate)
    best_val = float(truthful_awards[m])

    rem = 1.0 - fixed
    inv_scale = 1.0 / scale

    row = np.empty(n, dtype=np.float64)
    R = R0.copy()
    buf = np.zeros(k, dtype=np.int32)

    for add_units in compositions_nonneg_iter(free_units, k, out_arr=buf):
        row[m] = fixed
        row[others] = (add_units + min_units) * inv_scale * rem
        R[m, :] = row

        shares = impartial_division(R)
        val = cea_allocation(shares * scale, estate)[m]

        if pbar: pbar.update(1)
        if val > best_val:
            best_val = val

    if pbar: pbar.close()
    return float(best_val), [], []

def all_opt_rows_unbounded(R0, m, estate=20, scale=100, alpha=0.01, pbar=None, manip_claim=None):
    """
    Unrestricted: manipulator directly sets claims, keeping own claim fixed.
    """
    n = R0.shape[0]

    # Get manipulator's fixed claim
    if manip_claim is None:
        shares = impartial_division(R0)
        manip_claim_val = shares[m] * scale
    else:
        manip_claim_val = manip_claim

    others = np.array([i for i in range(n) if i != m])
    k = len(others)

    min_units = int(round(alpha * scale))
    remaining_units = scale - int(round(manip_claim_val))

    if min_units * k > remaining_units:
        if pbar: pbar.close()
        return float('-inf'), []


    shares = impartial_division(R0)
    truthful_awards = cea_allocation(shares * scale, estate)
    best_val = float(truthful_awards[m])


    manip_claim_fixed = manip_claim_val
    # Remaining claim must be distributed among others (MUST sum to scale - manip_claim)
    remaining_claim_total = scale - manip_claim_fixed


    remaining_units = int(round(remaining_claim_total))

    # Check if we can give everyone at least min_units
    if min_units * k > remaining_units:
        if pbar: pbar.close()
        return float('-inf'), []


    free_units = remaining_units - min_units * k
    buf = np.zeros(k, dtype=np.int32)

    for add_units in compositions_nonneg_iter(free_units, k, out_arr=buf):
   
        claims_vec = np.empty(n, dtype=np.float64)
        claims_vec[m] = manip_claim_fixed


        others_claims = add_units + min_units
        claims_vec[others] = others_claims.astype(np.float64)

        val = cea_allocation(claims_vec, estate)[m]

        if pbar: pbar.update(1)
        if val > best_val:
            best_val = val

    if pbar: pbar.close()
    return float(best_val), []

E_FIXED = 40
D_FIXED = 100
SCALE = 100
ALPHA = 0.01

BASE_NS = [4]  
MUTATE_NS = []

_ns_set = set(BASE_NS)
for d in MUTATE_NS:
    if d > 0:
        _ns_set.add(d)
    elif d < 0:
        _ns_set.discard(abs(d))
NS_TRACK = sorted(_ns_set)

if __name__ == "__main__":
    ns, orig_line, rest_line, unbd_line = [], [], [], []

    print(f"CEA Contested Garment Experiment:")
    print(f"D={D_FIXED}, E={E_FIXED} (CEA regime since E <= D/2={D_FIXED/2})")
    print(f"Tracking ns = {NS_TRACK}")
    print(f"Starting: n-1 agents get E/n each, manipulator gets D-(n-1)E/n\n")
    print("n | manip awards (Original / Restricted / Unrestricted)")
    print("-" * 78)

    for n in NS_TRACK:
        # CEA worst-case: n-1 agents get E/n, manipulator gets D-(n-1)E/n
        equal_share = E_FIXED / n
        manip_share = D_FIXED - (n - 1) * equal_share
        claims = [equal_share] * (n - 1) + [manip_share]
        m = n - 1

        print(f"\n{'='*60}")
        print(f"n={n}")
        print(f"{'='*60}")
        claims_arr = np.array(claims, dtype=np.float64)
        print(f"Input claims: {claims_arr}")

        R_truth = truthful_matrix(claims_arr)

        # ORIGINAL
        print(f"\nORIGINAL:")
        shares_orig = impartial_division(R_truth)
        print(f"  ID shares: {shares_orig}")
        claims_from_id = shares_orig * SCALE
        print(f"  Claims: {claims_from_id}")
        awards_orig = cea_allocation(claims_from_id, E_FIXED)
        print(f"  Awards: {awards_orig}")
        manip_orig = float(awards_orig[m])
        print(f"  → Manipulator gets: {manip_orig:.6f}")

        # RESTRICTED
        print(f"\nRESTRICTED (test 1 example):")
        R_restricted_example = R_truth.copy()
        example_row = np.zeros(n)
        example_row[0] = 0.97
        for i in range(1, n):
            if i != m:
                example_row[i] = 0.01
        example_row[m] = 0.0
        R_restricted_example[m] = example_row

        shares_restr = impartial_division(R_restricted_example)
        print(f"  Manipulated R[{m}]: {example_row}")
        print(f"  ID shares: {shares_restr}")
        claims_restr = shares_restr * SCALE
        print(f"  Claims: {claims_restr}")
        awards_restr = cea_allocation(claims_restr, E_FIXED)
        print(f"  Awards: {awards_restr}")
        print(f"  → Manipulator gets: {awards_restr[m]:.6f}")

        k = n - 1
        min_units = int(round(ALPHA * SCALE))
        free_units = max(0, SCALE - min_units * k)
        total_candidates = n_compositions(free_units, k)

        print(f"\nSearching {total_candidates} restricted manipulations...")
        pbar_r = tqdm(total=total_candidates, desc=f"  Restricted", unit="cand", leave=False, position=1)
        rest_best, _, _ = all_opt_rows_restricted(
            R_truth, m, estate=E_FIXED, scale=SCALE, alpha=ALPHA, pbar=pbar_r
        )
        print(f"  → Best: {float(rest_best):.6f}")

        # UNRESTRICTED
        print(f"\nUNRESTRICTED (test 2 examples):")
        remaining = SCALE - manip_share

        # Test 1: equal
        claims_test = np.zeros(n)
        claims_test[m] = manip_share
        for i, idx in enumerate([j for j in range(n) if j != m]):
            claims_test[idx] = remaining / 3
        print(f"  Equal: claims={claims_test}")
        awards_test = cea_allocation(claims_test, E_FIXED)
        print(f"         awards={awards_test}, manip={awards_test[m]:.6f}")

        # Test 2: extreme
        claims_test2 = np.zeros(n)
        claims_test2[m] = manip_share
        claims_test2[0] = remaining - 2*min_units
        claims_test2[1] = min_units
        claims_test2[2] = min_units
        print(f"  Extreme: claims={claims_test2}")
        awards_test2 = cea_allocation(claims_test2, E_FIXED)
        print(f"           awards={awards_test2}, manip={awards_test2[m]:.6f}")

        print(f"\nSearching {total_candidates} unrestricted manipulations...")
        pbar_u = tqdm(total=total_candidates, desc=f"  Unrestricted", unit="cand", leave=False, position=1)
        unbd_best, _ = all_opt_rows_unbounded(
            R_truth, m, estate=E_FIXED, scale=SCALE, alpha=ALPHA, pbar=pbar_u, manip_claim=manip_share
        )
        print(f"  → Best: {float(unbd_best):.6f}")

        ns.append(n)
        orig_line.append(manip_orig)
        rest_line.append(float(rest_best))
        unbd_line.append(float(unbd_best))


    print("\n=== SAVED DATA POINTS (n, Original, Restricted, Unrestricted) ===")
    for i in range(len(ns)):
        print(f"({ns[i]}, {orig_line[i]:.6f}, {rest_line[i]:.6f}, {unbd_line[i]:.6f})")


    plt.figure(figsize=(6.6, 4.2))
    plt.plot(ns, orig_line, marker="o", linestyle="-", linewidth=2, label="Original")
    plt.plot(ns, rest_line, marker="s", linestyle="--", linewidth=2, label="Restricted")
    plt.plot(ns, unbd_line, marker="^", linestyle=":", linewidth=2, label="Unrestricted")
    plt.xlabel("Number of agents (n)")
    plt.ylabel(f"Manipulator payoff (CG-CEA, E={E_FIXED})")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.title(f"CG-CEA – manipulator payoff vs n (E={E_FIXED}, D={D_FIXED})")
    plt.legend()
    plt.tight_layout()

    ns_tag = "-".join(map(str, ns))
    out_name = f"CEA_CG_E{int(E_FIXED)}_D{int(D_FIXED)}_ns{ns_tag}.png"
    plt.savefig(out_name, dpi=240)
    plt.close()
    print(f"Saved: {out_name}")
