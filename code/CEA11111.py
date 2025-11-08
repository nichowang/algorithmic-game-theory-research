# =================== CG-CEA: E = 35, 40, 45, 50 ===================
import math, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from functools import lru_cache
import pandas as pd

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: numba not installed. Install with: pip install numba")
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

# -------------------- Core methods (Numba optimized) --------------------
@njit(cache=True)
def _cg_cea_core(claims, estate, iters=60):
    n = len(claims)
    c_halved = np.empty(n, dtype=np.float64)
    max_val = 0.0
    total = 0.0
    for i in range(n):
        c_halved[i] = claims[i] * 0.5
        total += c_halved[i]
        if c_halved[i] > max_val:
            max_val = c_halved[i]
    if total <= estate:
        return c_halved
    lo, hi = 0.0, max_val
    for _ in range(iters):
        lam = (lo + hi) * 0.5
        s = 0.0
        for i in range(n):
            s += min(c_halved[i], lam)
        if s > estate:
            hi = lam
        else:
            lo = lam
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = min(c_halved[i], hi)
    return result

def cg_cea_allocation(claims, estate, iters=60):
    c = np.asarray(claims, dtype=np.float64)
    return _cg_cea_core(c, estate, iters)

def truthful_matrix(vec):
    vec = np.asarray(vec, dtype=np.float64)
    return np.tile(vec, (len(vec), 1))

def awards_from_matrix(R, estate, scale):
    return cg_cea_allocation(R.mean(axis=0) * scale, estate)

@njit(cache=True)
def _impartial_division_core(R, eps=1e-9):
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

@lru_cache(maxsize=32768)
def impartial_division_cached(R_tuple, eps=1e-9):
    n = int(np.sqrt(len(R_tuple)))
    R = np.array(R_tuple, dtype=np.float64).reshape(n, n)
    return tuple(_impartial_division_core(R, eps))

def impartial_division(R, eps=1e-9):
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

def all_opt_rows_restricted(R0, m, estate, scale, alpha, pbar=None):
    n = R0.shape[0]
    fixed = float(R0[m, m])
    others = np.array([i for i in range(n) if i != m])
    k = len(others)
    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        if pbar: pbar.close()
        return float('-inf'), [], []
    free_units = scale - min_units * k
    best_val = -1.0
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
        awards = _cg_cea_core(shares * scale, estate)
        val = awards[m]
        if pbar: pbar.update(1)
        if val > best_val:
            best_val = val
    if pbar: pbar.close()
    return float(best_val), [], []

def all_opt_rows_unbounded(R0, m, estate, scale, alpha, pbar=None):
    n = R0.shape[0]
    fixed = float(R0[m, m])
    others = np.array([i for i in range(n) if i != m])
    k = len(others)
    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        if pbar: pbar.close()
        return float('-inf'), []
    free_units = scale - min_units * k
    best_val = -1.0
    rem = 1.0 - fixed
    inv_scale = 1.0 / scale
    row = np.empty(n, dtype=np.float64)
    buf = np.zeros(k, dtype=np.int32)
    for add_units in compositions_nonneg_iter(free_units, k, out_arr=buf):
        row[m] = fixed
        row[others] = (add_units + min_units) * inv_scale * rem
        awards = _cg_cea_core(row * scale, estate)
        val = awards[m]
        if pbar: pbar.update(1)
        if val > best_val:
            best_val = val
    if pbar: pbar.close()
    return float(best_val), []

# -------------------- Main --------------------
SCALE = 100
ALPHA = 0.01
D_TOTAL = 100.0
E_VALUES = [35, 40, 45, 50]
NS = [4, 5, 6, 7]

if __name__ == "__main__":
    print("Warming up Numba JIT...")
    dummy = np.array([5.0, 5.0, 5.0, 85.0])
    for _ in range(10):
        _cg_cea_core(dummy, 20)
    print("✓ JIT ready\n")

    all_results = []

    pdf_filename = "cg_cea_E_35_40_45_50.pdf"
    with PdfPages(pdf_filename) as pdf:
        for E in E_VALUES:
            print("=" * 78)
            print(f"Running E = {E}")
            print("=" * 78)

            results = []

            for n in NS:
                # New setting: claims = [E/n] * (n-1) + [D - (n-1)*E/n]
                equal_share = E / n
                manip_claim = D_TOTAL - (n - 1) * equal_share
                claims = [equal_share] * (n - 1) + [manip_claim]
                m = n - 1
                vec = np.array(claims, dtype=np.float64) / SCALE
                R_truth = truthful_matrix(vec)

                awards_orig = awards_from_matrix(R_truth, E, SCALE)
                orig = float(awards_orig[m])

                k = n - 1
                min_units = int(round(ALPHA * SCALE))
                free_units = max(0, SCALE - min_units * k)
                total = n_compositions(free_units, k)

                pbar_r = tqdm(total=total, desc=f"[E={E},n={n}] rest", ncols=70)
                rest, _, _ = all_opt_rows_restricted(R_truth, m, E, SCALE, ALPHA, pbar_r)

                pbar_u = tqdm(total=total, desc=f"[E={E},n={n}] unbd", ncols=70)
                unbd, _ = all_opt_rows_unbounded(R_truth, m, E, SCALE, ALPHA, pbar_u)

                rest_final = max(rest, orig)
                unbd_final = max(unbd, orig)

                results.append({
                    'E': E, 'n': n, 'D': D_TOTAL,
                    'Original': orig, 'Restricted': rest_final, 'Unrestricted': unbd_final
                })
                all_results.append(results[-1])

                print(f"n={n} | {orig:.6f} / {rest_final:.6f} / {unbd_final:.6f}")

            # Plot for this E
            df = pd.DataFrame(results)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(df['n'], df['Original'], 'o-', linewidth=2, markersize=8, label='Original')
            ax.plot(df['n'], df['Restricted'], 's--', linewidth=2, markersize=8, label='Restricted')
            ax.plot(df['n'], df['Unrestricted'], '^:', linewidth=2, markersize=8, label='Unrestricted')
            ax.set_xlabel('Number of agents (n)', fontsize=11)
            ax.set_ylabel(f'Manipulator payoff (E={E})', fontsize=11)
            ax.set_title(f'CG-CEA: E={E}, D={D_TOTAL}', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            print(f"✓ Added E={E} to PDF\n")

    # Save all results to CSV
    df_all = pd.DataFrame(all_results)
    csv_filename = "cg_cea_E_35_40_45_50.csv"
    df_all.to_csv(csv_filename, index=False)

    print("=" * 78)
    print(f"✅ Saved PDF: {pdf_filename}")
    print(f"✅ Saved CSV: {csv_filename}")
    print("=" * 78)
