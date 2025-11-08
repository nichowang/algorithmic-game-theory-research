# =================== CEL: run + one-figure plot (3 lines, per-n tqdm progress) [FAST w/ ID cache] ===================
import math, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

# -------------------- 1) CEL (exact water-filling; same results as binary search) --------------------
def cel_allocation(claims, estate):
    c = np.asarray(claims, dtype=np.float64)
    D = float(c.sum())
    if estate <= 0:  # all zero
        return np.zeros_like(c)
    if estate >= D:  # full pay
        return c.copy()

    idx = np.argsort(-c)              # descending
    cs  = c[idx]
    ps  = np.cumsum(cs)
    n   = len(cs)

    lam = None
    for k in range(1, n + 1):
        lam_k = (ps[k - 1] - estate) / k
        if cs[k - 1] > lam_k and (k == n or lam_k >= cs[k]):
            lam = lam_k; break
    if lam is None:  # numeric guard
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

def awards_from_matrix(R, estate=60, scale=100):
    # Original: mean column shares -> scale (D) -> CEL
    return cel_allocation(R.mean(axis=0) * scale, estate)

# -------------------- 2) ID: cache pieces that don't change when only row m varies --------------------
def build_restricted_id_cache(R0, manip_index, eps=1e-9):
    """
    For restricted search we only modify row m. Everything else in R is constant.
    We precompute sums & counts needed by phi and psi excluding row m, then
    add/subtract the (row m) contribution on the fly. Exactly the same math,
    fewer allocations.

    Returns a dict 'C' used by 'impartial_division_restricted_fast'.
    """
    R0 = np.asarray(R0, dtype=float)
    n  = R0.shape[0]
    m  = int(manip_index)

    # masks/sets that exclude row m (since row m changes every iteration)
    pos_mask = (R0 > eps)

    # ---------- Precompute for phi[a,b] ----------
    # S_b_base = { l : l!=b, l!=m, R[l,b]>eps }
    S_b_mask = np.zeros((n, n), dtype=bool)   # [b, l] membership
    base_cnt_b = np.zeros(n, dtype=np.int32)
    # sums_by_a[a,b] = sum_{l in S_b_base} R[l,a]/R[l,b]
    sums_by_a = np.zeros((n, n), dtype=float)
    # self_term_phi[a,b] = R[a,a]/R[a,b] if a in S_b_base else 0
    self_term_phi = np.zeros((n, n), dtype=float)

    for b in range(n):
        mask = pos_mask[:, b].copy()
        if mask[b]: mask[b] = False
        if mask[m]: mask[m] = False
        S_b_mask[b] = mask
        base_cnt_b[b] = int(mask.sum())
        if base_cnt_b[b] == 0:
            continue
        denom = R0[mask, b]                      # shape (Lb,)
        inv   = 1.0 / denom                      # safe; denom>eps
        # For all a at once: sum_l R[l,a]/R[l,b]
        sums_by_a[:, b] = (R0[mask, :] * inv[:, None]).sum(axis=0)
        # self-term if 'a' itself is in the base set of column b
        idxs = np.nonzero(mask)[0]
        for a in idxs:
            self_term_phi[a, b] = R0[a, a] / R0[a, b]

    # ---------- Precompute for psi(res=j,k,i) ----------
    # S_i_base = { l : l!=i, l!=m, R[l,i]>eps }
    S_i_mask   = np.zeros((n, n), dtype=bool)         # [i, l] membership
    base_cnt_i = np.zeros(n, dtype=np.int32)
    # base_sum_psi[i,k] = sum_{l in S_i_base} R[l,k]/R[l,i]
    base_sum_psi = np.zeros((n, n), dtype=float)
    # inv_col_i[l] = 1/R[l,i] if l in S_i_base else 0
    inv_col_i = np.zeros((n, n), dtype=float)

    for i in range(n):
        mask = pos_mask[:, i].copy()
        if mask[i]: mask[i] = False
        if mask[m]: mask[m] = False
        S_i_mask[i] = mask
        base_cnt_i[i] = int(mask.sum())
        if base_cnt_i[i] == 0:
            continue
        denom = R0[mask, i]                       # shape (Li,)
        inv   = 1.0 / denom
        inv_col_i[i, mask] = inv
        base_sum_psi[i, :] = (R0[mask, :] * inv[:, None]).sum(axis=0)

    C = dict(
        R=R0, n=n, m=m, eps=eps,
        # phi cache
        S_b_mask=S_b_mask, base_cnt_b=base_cnt_b, sums_by_a=sums_by_a, self_term_phi=self_term_phi,
        # psi cache
        S_i_mask=S_i_mask, base_cnt_i=base_cnt_i, base_sum_psi=base_sum_psi, inv_col_i=inv_col_i
    )
    return C

def impartial_division_restricted_fast(R, cache):
    """
    Same math as your original 'impartial_division', but uses the cache built
    above. R differs from cache['R'] only at row m; others are identical.
    """
    R = np.asarray(R, dtype=float)
    n = cache['n']; m = cache['m']
    S_b_mask     = cache['S_b_mask']
    base_cnt_b   = cache['base_cnt_b']
    sums_by_a    = cache['sums_by_a']
    self_term_phi= cache['self_term_phi']

    S_i_mask     = cache['S_i_mask']
    base_cnt_i   = cache['base_cnt_i']
    base_sum_psi = cache['base_sum_psi']
    inv_col_i    = cache['inv_col_i']

    # ---------- phi[a,b] with row-m incremental term ----------
    # phi[a,b] = mean_{l in S_b_base \ {a}} R[l,a]/R[l,b], then + row-m term if allowed
    phi = np.ones((n, n), dtype=float)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            cnt = base_cnt_b[b]
            s   = sums_by_a[a, b]
            # subtract self term if a is in S_b_base
            if S_b_mask[b, a]:
                s   -= self_term_phi[a, b]
                cnt -= 1
            # add row-m term if a!=m  (l=m cannot equal a)
            # in restricted, R[m,b] > 0 for all b (alpha>0 and fixed>0), so include it
            if a != m:
                s   += R[m, a] / R[m, b]
                cnt += 1
            phi[a, b] = (s / cnt) if cnt > 0 else 1.0

    # ---------- f^j and total ----------
    total = np.zeros(n, dtype=float)
    for j in range(n):
        f = np.zeros(n, dtype=float)
        for i in range(n):
            if i == j:
                continue
            # sum_k psi(j,k,i)
            # start from base over l in S_i_base, then remove l=j, remove l=k, possibly add l=m
            sum_psi = 0.0
            # auxiliary: base count/sum for i
            base_cnt = base_cnt_i[i]
            # prefetch vectorized things for speed
            base_sum_vec = base_sum_psi[i, :]     # length n
            inv_vec      = inv_col_i[i, :]        # length n, zero where not in base

            for k in range(n):
                if k == i or k == j:
                    continue
                s = base_sum_vec[k]
                cnt = base_cnt

                # subtract l=j if j ∈ S_i_base
                if S_i_mask[i, j]:
                    s   -= R[j, k] * inv_vec[j]
                    cnt -= 1
                # subtract l=k if k ∈ S_i_base and k != j
                if S_i_mask[i, k]:
                    s   -= R[k, k] * inv_vec[k]    # l=k term uses R[k,k]/R[k,i]
                    cnt -= 1
                # add l=m if allowed: l cannot be j or k
                if (j != m) and (k != m):
                    s   += R[m, k] / R[m, i]
                    cnt += 1

                psi_jki = (s / cnt) if cnt > 0 else 1.0
                sum_psi += psi_jki

            f[i] = 1.0 / (1.0 + phi[j, i] + sum_psi)

        f[j] = 1.0 - f.sum()
        total += f

    shares = total / n
    s = shares.sum()
    if s != 1.0 and s > 0:
        shares /= s
    return shares

# -------------------- 3) Stars-and-bars generator (same order; no pre-storage) --------------------
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

# -------------------- 4) Two searches (identical search space; use ID cache in restricted) --------------------
def all_opt_rows_restricted(R0, m, estate=60, scale=100, tol=1e-9, alpha=0.01, pbar=None):
    n = R0.shape[0]
    fixed  = float(R0[m, m])
    others = [i for i in range(n) if i != m]
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        if pbar: pbar.close()
        return float('-inf'), [], []

    free_units = scale - min_units * k
    best_val, best_rows, best_shares = -1.0, [], []

    # Pre-allocate & reuse
    row = np.zeros(n, dtype=np.float64); row[m] = fixed
    R   = R0.copy()
    buf = np.zeros(k, dtype=np.int32)
    rem = 1.0 - fixed
    inv_scale = 1.0 / scale

    # Build ID cache once (everything except row m is constant across candidates)
    R[m, :] = row   # make a consistent shape once
    cache = build_restricted_id_cache(R, manip_index=m, eps=1e-9)

    for add_units in compositions_nonneg_iter(free_units, k, out_arr=buf):
        if pbar: pbar.update(1)

        row.fill(0.0); row[m] = fixed
        row[others] = (add_units + min_units) * inv_scale * rem
        R[m, :] = row

        shares = impartial_division_restricted_fast(R, cache)
        val    = cel_allocation(shares * scale, estate)[m]

        if val > best_val + tol:
            best_val = val
            best_rows   = [np.round(row, 3).tolist()]
            best_shares = [np.round(shares, 3).tolist()]
        elif abs(val - best_val) < tol:
            best_rows.append(np.round(row, 3).tolist())
            best_shares.append(np.round(shares, 3).tolist())

    if pbar: pbar.close()
    if best_val < 0:
        return float('-inf'), [], []
    return float(best_val), best_rows, best_shares

def all_opt_rows_unrestricted(R0, m, estate=60, scale=100, tol=1e-9, alpha=0.01, pbar=None):
    n = R0.shape[0]
    fixed  = float(R0[m, m])
    others = [i for i in range(n) if i != m]
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        if pbar: pbar.close()
        return float('-inf'), []

    free_units = scale - min_units * k
    best_val, best_rows = -1.0, []

    row = np.zeros(n, dtype=np.float64); row[m] = fixed
    buf = np.zeros(k, dtype=np.int32)
    rem = 1.0 - fixed
    inv_scale = 1.0 / scale

    for add_units in compositions_nonneg_iter(free_units, k, out_arr=buf):
        if pbar: pbar.update(1)

        row.fill(0.0); row[m] = fixed
        row[others] = (add_units + min_units) * inv_scale * rem

        val = cel_allocation(row * scale, estate)[m]
        if val > best_val + tol:
            best_val, best_rows = val, [np.round(row, 3).tolist()]
        elif abs(val - best_val) < tol:
            best_rows.append(np.round(row, 3).tolist())

    if pbar: pbar.close()
    if best_val < 0:
        return float('-inf'), []
    return float(best_val), best_rows

# -------------------- 5) Suggested entitlements (your CEL base) --------------------
def suggested_entitlements(n, manip=0, E=60.0, D=100.0, foil=None):
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

# -------------------- 6) Main (same outer loop, same search count & order) --------------------
E_FIXED   = 60
SCALE     = 100
ALPHA     = 0.01
AGENT_M   = 0
FOIL_ID   = None
BASE_NS   = [4, 5, 6, 7]
MUTATE_NS = []

_ns_set = set(BASE_NS)
for d in MUTATE_NS:
    if d > 0: _ns_set.add(d)
    elif d < 0: _ns_set.discard(abs(d))
NS_TRACK = sorted(_ns_set)

ns, orig_line, rest_line, unres_line = [], [], [], []

print(f"Tracking ns = {NS_TRACK}")
print("n | manip awards (Original / Restricted / Unrestricted)")
print("-"*78)

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

    pbar_r = tqdm(total=total_candidates, desc=f"[n={n}] restricted", unit="cand", leave=True)
    rest_best, _, _ = all_opt_rows_restricted(
        R_truth, AGENT_M, estate=E_FIXED, scale=SCALE, alpha=ALPHA, pbar=pbar_r
    )

    pbar_u = tqdm(total=total_candidates, desc=f"[n={n}] unrestricted", unit="cand", leave=True)
    unres_best, _ = all_opt_rows_unrestricted(
        R_truth, AGENT_M, estate=E_FIXED, scale=SCALE, alpha=ALPHA, pbar=pbar_u
    )

    ns.append(n)
    orig_line.append(manip_orig)
    rest_line.append(float(rest_best))
    unres_line.append(float(unres_best))

    print(f"{n} | {manip_orig:.6f} / {float(rest_best):.6f} / {float(unres_best):.6f}")

# -------------------- 7) Plot --------------------
plt.figure(figsize=(6.6, 4.2))
plt.plot(ns, orig_line,  marker="o", linestyle="-",  linewidth=2, label="Original")
plt.plot(ns, rest_line,  marker="s", linestyle="--", linewidth=2, label="Restricted")
plt.plot(ns, unres_line, marker="^", linestyle=":",  linewidth=2, label="Unrestricted")
plt.xlabel("Number of agents (n)")
plt.ylabel(f"Manipulator payoff (CEL, E={E_FIXED})")
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
plt.title("CEL – manipulator payoff vs n (base: d_i=(D−E)/2, only one foil gets the rest)")
plt.legend()
plt.tight_layout()
ns_tag = "-".join(map(str, ns))
out_name = f"cel_E{int(E_FIXED)}_ns{ns_tag}_alpha{int(ALPHA*SCALE)}.png"
plt.savefig(out_name, dpi=240); plt.close()
print(f"✅ Saved: {out_name}")