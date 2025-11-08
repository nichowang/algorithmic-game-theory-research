# =================== CEL: run + plot for E=[55,60,65], n=4,5,6,7 - CSV + Ratio + PDF ===================
import math, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
from functools import lru_cache
import pandas as pd

# Try to import numba with fallback
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# -------------------- 1) CEL (exact water-filling) --------------------
def cel_allocation(claims, estate):
    c = np.asarray(claims, dtype=np.float64)
    D = float(c.sum())
    if estate <= 0:
        return np.zeros_like(c)
    if estate >= D:
        return c.copy()

    idx = np.argsort(-c)
    cs  = c[idx]
    ps  = np.cumsum(cs)
    n   = len(cs)

    lam = None
    for k in range(1, n + 1):
        lam_k = (ps[k - 1] - estate) / k
        if cs[k - 1] > lam_k and (k == n or lam_k >= cs[k]):
            lam = lam_k; break
    if lam is None:
        lam = (ps[-1] - estate) / n

    as_ = np.maximum(0.0, cs - lam)
    a   = np.zeros_like(as_)
    a[idx] = as_
    s = a.sum()
    if s > 0 and abs(s - estate) > 1e-12:
        a *= (estate / s)
    return a

def truthful_matrix(vec):
    v = np.asarray(vec, dtype=np.float64)
    return np.tile(v, (len(v), 1))

def awards_from_matrix(R, estate=70, scale=100):
    return cel_allocation(R.mean(axis=0) * scale, estate)

# -------------------- 2) ID: cache pieces for restricted --------------------
def build_restricted_id_cache(R0, manip_index, eps=1e-9):
    R0 = np.asarray(R0, dtype=float)
    n  = R0.shape[0]
    m  = int(manip_index)

    pos_mask = (R0 > eps)

    S_b_mask = np.zeros((n, n), dtype=bool)
    base_cnt_b = np.zeros(n, dtype=np.int32)
    sums_by_a = np.zeros((n, n), dtype=float)
    self_term_phi = np.zeros((n, n), dtype=float)

    for b in range(n):
        mask = pos_mask[:, b].copy()
        if mask[b]: mask[b] = False
        if mask[m]: mask[m] = False
        S_b_mask[b] = mask
        base_cnt_b[b] = int(mask.sum())
        if base_cnt_b[b] == 0:
            continue
        denom = R0[mask, b]
        inv   = 1.0 / denom
        sums_by_a[:, b] = (R0[mask, :] * inv[:, None]).sum(axis=0)
        idxs = np.nonzero(mask)[0]
        for a in idxs:
            self_term_phi[a, b] = R0[a, a] / R0[a, b]

    S_i_mask   = np.zeros((n, n), dtype=bool)
    base_cnt_i = np.zeros(n, dtype=np.int32)
    base_sum_psi = np.zeros((n, n), dtype=float)
    inv_col_i = np.zeros((n, n), dtype=float)

    for i in range(n):
        mask = pos_mask[:, i].copy()
        if mask[i]: mask[i] = False
        if mask[m]: mask[m] = False
        S_i_mask[i] = mask
        base_cnt_i[i] = int(mask.sum())
        if base_cnt_i[i] == 0:
            continue
        denom = R0[mask, i]
        inv   = 1.0 / denom
        inv_col_i[i, mask] = inv
        base_sum_psi[i, :] = (R0[mask, :] * inv[:, None]).sum(axis=0)

    C = dict(
        R=R0, n=n, m=m, eps=eps,
        S_b_mask=S_b_mask, base_cnt_b=base_cnt_b, sums_by_a=sums_by_a, self_term_phi=self_term_phi,
        S_i_mask=S_i_mask, base_cnt_i=base_cnt_i, base_sum_psi=base_sum_psi, inv_col_i=inv_col_i
    )
    return C

@jit(nopython=True, cache=True)
def _compute_phi_core(n, m, R, S_b_mask, base_cnt_b, sums_by_a, self_term_phi):
    phi = np.ones((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            cnt = base_cnt_b[b]
            s = sums_by_a[a, b]
            if S_b_mask[b, a]:
                s -= self_term_phi[a, b]
                cnt -= 1
            if a != m and R[m, b] != 0.0:
                s += R[m, a] / R[m, b]
                cnt += 1
            phi[a, b] = (s / cnt) if cnt > 0 else 1.0
    return phi

@jit(nopython=True, cache=True)
def _compute_psi_and_total(n, m, R, phi, S_i_mask, base_cnt_i, base_sum_psi, inv_col_i):
    total = np.zeros(n, dtype=np.float64)
    for j in range(n):
        f = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == j:
                continue
            sum_psi = 0.0
            base_cnt = base_cnt_i[i]

            for k in range(n):
                if k == i or k == j:
                    continue
                s = base_sum_psi[i, k]
                cnt = base_cnt

                if S_i_mask[i, j]:
                    s -= R[j, k] * inv_col_i[i, j]
                    cnt -= 1
                if S_i_mask[i, k]:
                    s -= R[k, k] * inv_col_i[i, k]
                    cnt -= 1
                if (j != m) and (k != m) and R[m, i] != 0.0:
                    s += R[m, k] / R[m, i]
                    cnt += 1

                psi_jki = (s / cnt) if cnt > 0 else 1.0
                sum_psi += psi_jki

            f[i] = 1.0 / (1.0 + phi[j, i] + sum_psi)

        f[j] = 1.0 - f.sum()
        total += f

    return total

def impartial_division_restricted_fast(R, cache):
    R = np.asarray(R, dtype=np.float64)
    n = cache['n']; m = cache['m']
    S_b_mask = cache['S_b_mask']
    base_cnt_b = cache['base_cnt_b']
    sums_by_a = cache['sums_by_a']
    self_term_phi = cache['self_term_phi']

    S_i_mask = cache['S_i_mask']
    base_cnt_i = cache['base_cnt_i']
    base_sum_psi = cache['base_sum_psi']
    inv_col_i = cache['inv_col_i']

    phi = _compute_phi_core(n, m, R, S_b_mask, base_cnt_b, sums_by_a, self_term_phi)
    total = _compute_psi_and_total(n, m, R, phi, S_i_mask, base_cnt_i, base_sum_psi, inv_col_i)

    shares = total / n
    s = shares.sum()
    if s != 1.0 and s > 0:
        shares /= s
    return shares

# -------------------- 3) Composition generator --------------------
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

# -------------------- 4) Two searches --------------------
def all_opt_rows_restricted(R0, m, estate=70, scale=100, tol=1e-9, alpha=0.01, pbar=None):
    n = R0.shape[0]
    fixed = float(R0[m, m])
    others = np.array([i for i in range(n) if i != m], dtype=np.int32)
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        if pbar: pbar.close()
        return float('-inf'), [], []

    free_units = scale - min_units * k
    best_val = -1.0
    rem = 1.0 - fixed
    inv_scale = 1.0 / scale

    R = R0.copy()
    row = R[m, :]
    row[m] = fixed
    cache = build_restricted_id_cache(R, manip_index=m, eps=1e-9)

    buf = np.zeros(k, dtype=np.int32)
    min_units_arr = np.full(k, min_units, dtype=np.float64)

    for add_units in compositions_nonneg_iter(free_units, k, out_arr=buf):
        row[m] = fixed
        row[others] = (add_units + min_units_arr) * inv_scale * rem

        shares = impartial_division_restricted_fast(R, cache)
        val = cel_allocation(shares * scale, estate)[m]

        if pbar: pbar.update(1)
        if val > best_val + tol:
            best_val = val

    if pbar: pbar.close()
    if best_val < 0:
        return float('-inf'), [], []
    return float(best_val), [], []

def all_opt_rows_unrestricted(R0, m, estate=70, scale=100, tol=1e-9, alpha=0.01, pbar=None):
    n = R0.shape[0]
    fixed = float(R0[m, m])
    others = np.array([i for i in range(n) if i != m], dtype=np.int32)
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        if pbar: pbar.close()
        return float('-inf'), []

    free_units = scale - min_units * k
    best_val = -1.0
    rem = 1.0 - fixed
    inv_scale = 1.0 / scale

    row = np.zeros(n, dtype=np.float64)
    buf = np.zeros(k, dtype=np.int32)
    min_units_arr = np.full(k, min_units, dtype=np.float64)
    scaled_row = np.zeros(n, dtype=np.float64)

    for add_units in compositions_nonneg_iter(free_units, k, out_arr=buf):
        row[m] = fixed
        row[others] = (add_units + min_units_arr) * inv_scale * rem

        np.multiply(row, scale, out=scaled_row)
        val = cel_allocation(scaled_row, estate)[m]

        if pbar: pbar.update(1)
        if val > best_val + tol:
            best_val = val

    if pbar: pbar.close()
    if best_val < 0:
        return float('-inf'), []
    return float(best_val), []

# -------------------- 5) Suggested entitlements --------------------
def suggested_entitlements(n, manip=0, E=70.0, D=100.0, foil=None):
    assert 0 < E < D and n >= 2
    manip = int(manip) % n
    foil  = (manip + 1) % n if foil is None else (int(foil) % n)
    assert foil != manip
    di = 0.5 * (D - E)
    di = max(0.0, min(di, D))
    d = np.zeros(n, dtype=np.float64)
    d[manip] = di
    d[foil]  = D - di
    return d

# -------------------- 6) Main computation --------------------
SCALE     = 100
ALPHA     = 0.01
AGENT_M   = 0
FOIL_ID   = None
NS_TRACK  = [4, 5, 6, 7]
E_VALUES  = [55, 60, 65]

if __name__ == "__main__":
    print(f"Numba acceleration: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")

    all_results = {}

    for E_FIXED in E_VALUES:
        print(f"\n{'='*70}")
        print(f"Running for E={E_FIXED}")
        print(f"{'='*70}")

        ns, orig_line, rest_line, unres_line = [], [], [], []

        for n in NS_TRACK:
            claims_base = suggested_entitlements(n, manip=AGENT_M, E=E_FIXED, D=SCALE, foil=FOIL_ID)
            shares_base = claims_base / SCALE
            R_truth = truthful_matrix(shares_base)

            awards_orig = awards_from_matrix(R_truth, estate=E_FIXED, scale=SCALE)
            manip_orig  = float(awards_orig[AGENT_M])

            k = n - 1
            min_units = int(round(ALPHA * SCALE))
            free_units = max(0, SCALE - min_units * k)
            total_candidates = n_compositions(free_units, k)

            pbar_r = tqdm(total=total_candidates, desc=f"[E={E_FIXED}, n={n}] restricted", unit="cand", leave=True)
            rest_best, _, _ = all_opt_rows_restricted(
                R_truth, AGENT_M, estate=E_FIXED, scale=SCALE, alpha=ALPHA, pbar=pbar_r
            )

            pbar_u = tqdm(total=total_candidates, desc=f"[E={E_FIXED}, n={n}] unrestricted", unit="cand", leave=True)
            unres_best, _ = all_opt_rows_unrestricted(
                R_truth, AGENT_M, estate=E_FIXED, scale=SCALE, alpha=ALPHA, pbar=pbar_u
            )

            ns.append(n)
            orig_line.append(manip_orig)
            rest_line.append(float(rest_best))
            unres_line.append(float(unres_best))

            print(f"n={n} | Original: {manip_orig:.6f}, Restricted: {float(rest_best):.6f}, Unrestricted: {float(unres_best):.6f}")

        all_results[E_FIXED] = {
            'n': ns,
            'original': orig_line,
            'restricted': rest_line,
            'unrestricted': unres_line
        }

    # Save CSV files (combined absolute + ratio)
    print(f"\n{'='*70}")
    print("💾 Saving CSV files...")
    print(f"{'='*70}")

    for E in E_VALUES:
        data = all_results[E]

        df = pd.DataFrame({
            'n': data['n'],
            'original': data['original'],
            'restricted': data['restricted'],
            'unrestricted': data['unrestricted'],
            'restricted/original': np.array(data['restricted']) / np.array(data['original']),
            'unrestricted/original': np.array(data['unrestricted']) / np.array(data['original'])
        })
        csv_file = f"CEL_n_E{E}.csv"
        df.to_csv(csv_file, index=False)
        print(f"✅ Saved {csv_file}")

    # Generate PDF plots
    print(f"\n{'='*70}")
    print("📊 Generating PDF plots...")
    print(f"{'='*70}")

    for E in E_VALUES:
        data = all_results[E]

        plt.figure(figsize=(8, 6))
        plt.plot(data['n'], data['original'], marker="o", linestyle="-", linewidth=2, markersize=8, label="Original")
        plt.plot(data['n'], data['restricted'], marker="s", linestyle="--", linewidth=2, markersize=8, label="Restricted")
        plt.plot(data['n'], data['unrestricted'], marker="^", linestyle=":", linewidth=2, markersize=8, label="Unrestricted")

        plt.xlabel("Number of agents (n)", fontsize=12)
        plt.ylabel(f"Manipulator payoff", fontsize=12)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        plt.title(f"CEL – Manipulator payoff vs n (E={E})", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        pdf_file = f"CEL_n_E{E}.pdf"
        plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved {pdf_file}")
        plt.close()

    print(f"\n{'='*70}")
    print("✅ ALL TASKS COMPLETED!")
    print(f"{'='*70}")
    print(f"\nGenerated files for E={E_VALUES}:")
    for E in E_VALUES:
        print(f"  - CEL_n_E{E}.csv (absolute + ratio)")
        print(f"  - CEL_n_E{E}.pdf")
