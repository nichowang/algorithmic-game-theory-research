# =================== CEA sweep (unique combos, weighted average) ===================
import numpy as np
import itertools
from math import comb
from tqdm import tqdm
import matplotlib.pyplot as plt

# ======= 全局开关：是否按多重度加权（推荐 True） =======
WEIGHT_BY_MULTIPLICITY = True

# ---------- Core: CEA ----------
def cea_allocation(claims, estate, tol=1e-12, iters=60):
    c = np.asarray(claims, float)
    lo, hi = 0.0, c.max()
    for _ in range(iters):
        lam = (lo + hi) / 2
        if np.minimum(c, lam).sum() > estate:
            hi = lam
        else:
            lo = lam
    return np.minimum(c, hi)

def truthful_matrix(vec):
    v = np.asarray(vec, float)
    return np.tile(v, (len(v), 1))

def awards_from_matrix(R, estate=40, scale=100):
    return cea_allocation(R.mean(0) * scale, estate)

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

# ---------- 枚举把 free_units 分到 k 个桶（可为 0）的所有组合 ----------
def compositions_nonneg(free_units, k):
    for cuts in itertools.combinations(range(free_units + k - 1), k - 1):
        add = np.diff((-1, *cuts, free_units + k - 1)) - 1
        yield add

# ---------- Restricted：操作者在行内分配（列下限 alpha），ID->CEA，返回 best 值 ----------
def restricted_best(R0, m, estate=40, scale=100, alpha=0.01, tol=1e-9):
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
        p_units = np.asarray(add) + min_units         # 每列至少 1 个单位
        p = p_units / p_units.sum()                   # 归一化到 1
        rem = 1.0 - fixed

        row = np.zeros(n, dtype=float)
        row[m] = fixed
        row[others] = p * rem

        R = R0.copy(); R[m] = row
        shares = impartial_division(R)
        val = cea_allocation(shares * scale, estate)[m]
        if val > best_val + tol:
            best_val = val

    return best_val

# ---------- Unrestricted：操作者行直接做 CEA，返回 best 值 ----------
def unrestricted_best(R0, m, estate=40, scale=100, alpha=0.01, tol=1e-9):
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

        val = cea_allocation(row * scale, estate)[m]
        if val > best_val + tol:
            best_val = val

    return best_val

# ---------- 计算 (a1,a2,a3) 的多重度：等价排列个数（n=4 时是 1/3/6） ----------
def multiplicity_three(a1, a2, a3):
    # 4 个玩家里固定了操作者 a0，这里只看后三个的排列数
    if a1 == a2 == a3:
        return 1
    elif a1 == a2 or a2 == a3:
        return 3
    else:
        return 6

# ---------- 统计唯一三元组总数（用于设置 tqdm total） ----------
def count_unique_triples(rest_units, min_units=1):
    cnt = 0
    for a1 in range(min_units, rest_units - 2*min_units + 1):
        for a2 in range(a1, rest_units - a1 - min_units + 1):
            a3 = rest_units - a1 - a2
            if a3 >= a2 and a3 >= min_units:
                cnt += 1
    return cnt

# ---------- 主流程：对 a0=0.01..0.97 扫描，双层进度条 ----------
def sweep_unique_combinations_with_progress(E=40, D=100, alpha=0.01, manip_index=0):
    n = 4
    min_units = int(round(alpha * D))  # =1
    rows = []

    outer_total = (D - 3*min_units) - 1 + 1   # 0.01..0.97 共 97 步
    outer = tqdm(range(1, D - 3*min_units + 1),
                 desc="[a0] 0.01→0.97",
                 total=outer_total, position=0, leave=True, ncols=100)

    for a0_units in outer:
        a0 = a0_units / D
        rest_units = D - a0_units

        total_unique = count_unique_triples(rest_units, min_units=min_units)

        # 3 条线的（加权）累计
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

                # 组合 -> 向量
                v_units = np.array([a0_units, a1_units, a2_units, a3_units], dtype=float)
                v = v_units / D
                R0 = truthful_matrix(v)

                # Original：如实（平均行）-> CEA
                orig = awards_from_matrix(R0, estate=E, scale=D)[manip_index]
                # Restricted：操作者行在列下限下最优 -> ID -> CEA
                r_best = restricted_best(R0, m=manip_index, estate=E, scale=D, alpha=alpha)
                # Unrestricted：操作者行最优 -> 直接 CEA
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

# ---------- 运行 & 画图 ----------
if __name__ == "__main__":
    E, D, alpha = 40, 100, 0.01
    df = sweep_unique_combinations_with_progress(E=E, D=D, alpha=alpha, manip_index=0)

    print(df.head()); print(df.tail())

    plt.figure(figsize=(8, 5))
    plt.plot(df["a0"], df["original_avg"],     label="Original",     linewidth=2)
    plt.plot(df["a0"], df["restricted_avg"],   label="Restricted",   linewidth=2)
    plt.plot(df["a0"], df["unrestricted_avg"], label="Unrestricted", linewidth=2)
    plt.xlabel("Manipulator initial claim a0")
    plt.ylabel("Final reward of manipulator (CEA)")
    plt.title(f"CEA – Manipulator Reward vs a0 (unique combos; E={E}, D={D}, n=4)\n"
              f"weighted_by_multiplicity={WEIGHT_BY_MULTIPLICITY}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("cea_sweep_a0_unique_weighted.png", dpi=200)
    plt.show()
