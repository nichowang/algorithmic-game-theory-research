# =================== CEA: run + plots (multi-E) ===================
import itertools, json, math, numpy as np, pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast
from pathlib import Path

# -------------------- 1) Core methods --------------------
def cea_allocation(claims, estate, tol=1e-12, iters=60):
    c = np.asarray(claims, float)
    lo, hi = 0.0, c.max()
    for _ in range(iters):
        lam = (lo + hi) / 2
        hi, lo = (lam, lo) if np.minimum(c, lam).sum() > estate else (hi, lam)
    return np.minimum(c, hi)

def truthful_matrix(vec):
    return np.tile(vec, (len(vec), 1))

def awards_from_matrix(R, estate=35, scale=100):
    return cea_allocation(R.mean(0) * scale, estate)

def impartial_division(R, eps=1e-9):
    R, n = np.asarray(R, float), R.shape[0]
    phi = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b: continue
            ratios = [R[l, a] / (R[l, b] + eps)
                      for l in range(n) if l not in (a, b) and R[l, b] > eps]
            phi[a, b] = sum(ratios)/len(ratios) if ratios else 1.0
    def psi(res, k, i):
        reps = [l for l in range(n) if l not in (res, k, i) and R[l, i] > eps]
        return sum(R[l, k]/(R[l, i]+eps) for l in reps)/len(reps) if reps else 1.0
    total = np.zeros(n)
    for j in range(n):
        f = np.zeros(n)
        for i in range(n):
            if i == j: continue
            f[i] = 1.0 / (1.0 + phi[j, i] +
                          sum(psi(j, k, i) for k in range(n) if k not in (i, j)))
        f[j] = 1.0 - f.sum()
        total += f
    shares = total / n
    return shares / shares.sum()

def compositions_nonneg(total, parts):
    for cuts in itertools.combinations(range(total + parts - 1), parts - 1):
        xs = np.diff((-1, *cuts, total + parts - 1)) - 1
        yield xs

def all_opt_rows_restricted(R0, m, estate=35, scale=100, tol=1e-9, alpha=0.01):
    n = R0.shape[0]
    fixed  = float(R0[m, m])
    others = [i for i in range(n) if i != m]
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        return float('-inf'), [], []

    free_units = scale - min_units * k
    best_val, best_rows, best_shares = -1.0, [], []

    for add_units in compositions_nonneg(free_units, k):
        p_units = [min_units + u for u in add_units]          
        p = np.array(p_units, dtype=float) / scale           
        rem = 1.0 - fixed
        row = np.zeros(n); row[m] = fixed
        for idx, pj in zip(others, p):
            row[idx] = pj * rem

        R = R0.copy(); R[m] = row
        shares = impartial_division(R)
        val    = cea_allocation(shares * scale, estate)[m]

        if val > best_val + tol:
            best_val = val
            best_rows = [np.round(row, 3).tolist()]
            best_shares = [np.round(shares, 3).tolist()]
        elif abs(val - best_val) < tol:
            best_rows.append(np.round(row, 3).tolist())
            best_shares.append(np.round(shares, 3).tolist())

    if best_val < 0:
        return float('-inf'), [], []
    return round(best_val, 3), best_rows, best_shares

def all_opt_rows_unbounded(R0, m, estate=35, scale=100, tol=1e-9, alpha=0.01):
    n = R0.shape[0]
    fixed  = float(R0[m, m])   
    others = [i for i in range(n) if i != m]
    k = len(others)

    min_units = int(round(alpha * scale))
    if min_units * k > scale:
        return float('-inf'), []

    free_units = scale - min_units * k
    best_val, best_rows = -1.0, []

    for add_units in compositions_nonneg(free_units, k):
        p_units = [min_units + u for u in add_units]
        p = np.array(p_units, dtype=float) / scale
        rem = 1.0 - fixed

        row = np.zeros(n); row[m] = fixed
        for idx, pj in zip(others, p):
            row[idx] = pj * rem

        val = cea_allocation(row * scale, estate)[m]

        if val > best_val + tol:
            best_val, best_rows = val, [np.round(row, 3).tolist()]
        elif abs(val - best_val) < tol:
            best_rows.append(np.round(row, 3).tolist())

    if best_val < 0:
        return float('-inf'), []
    return round(best_val, 3), best_rows

# -------------------- 2) Scenarios --------------------
share_vectors = [
    np.array([0.08, 0.19, 0.07, 0.66]),
    np.array([0.48, 0.40, 0.05, 0.07]),
    np.array([0.22, 0.24, 0.09, 0.45]),
    np.array([0.35, 0.05, 0.24, 0.36]),
    np.array([0.15, 0.19, 0.13, 0.53]),
    np.array([0.21, 0.23, 0.08, 0.48]),
    np.array([0.10, 0.19, 0.09, 0.62]),
    np.array([0.18, 0.16, 0.09, 0.57]),
    np.array([0.22, 0.20, 0.14, 0.44]),
    np.array([0.35, 0.10, 0.24, 0.31]),
    np.array([0.10, 0.13, 0.09, 0.68]),
    np.array([0.28, 0.42, 0.25, 0.05]),
    np.array([0.18, 0.16, 0.19, 0.47]),
    np.array([0.25, 0.20, 0.11, 0.44]),
    np.array([0.35, 0.04, 0.30, 0.31]),
    np.array([0.15, 0.19, 0.06, 0.60]),
]

# -------------------- 3) Run all estates --------------------
records_all = []
estate_values = list(range(15, 50, 5))  # 15,20,...,45
ALPHA = 0.01
SCALE = 100

for E in estate_values:
    for sv in tqdm(share_vectors, desc=f"CEA scenarios @ D={E}"):
        R_truth = truthful_matrix(sv)
        orig_aw = awards_from_matrix(R_truth, estate=E, scale=SCALE)
        for m in range(len(sv)):
            rest_best, rest_rows, rest_sh = all_opt_rows_restricted(
                R_truth, m, estate=E, scale=SCALE, alpha=ALPHA)
            unb_best,  unb_rows           = all_opt_rows_unbounded(
                R_truth, m, estate=E, scale=SCALE, alpha=ALPHA)

            records_all.append({
                "Estate":           E,
                "Scenario":         np.round(sv,3).tolist(),
                "Manipulator":      m,
                "Orig_Best":        round(orig_aw[m],3),
                "Rest_Best":        rest_best,
                "Unbd_Best":        unb_best,
                "Optimal_Rows":     json.dumps(rest_rows),
                "Unbd_Rows":        json.dumps(unb_rows),
                "Impartial_Shares": json.dumps(rest_sh),
            })

df_all = pd.DataFrame(records_all)
df_all.to_excel("CEA_detailed_awards_summary_multiE.xlsx", index=False)
print("✅ CEA_detailed_awards_summary_multiE.xlsx written")

# -------------------- 4) Plots --------------------
outdir = Path(".")
outdir.mkdir(exist_ok=True)

def _len_scenario(x):
    try:
        return len(x)
    except Exception:
        return len(ast.literal_eval(x)) if isinstance(x, str) else None

df_all["NumAgents"] = df_all["Scenario"].apply(_len_scenario)
df_all["Rest_Gain"] = df_all["Rest_Best"] - df_all["Orig_Best"]
df_all["Unbd_Gain"] = df_all["Unbd_Best"] - df_all["Orig_Best"]

# --- Plot 1: Max manipulability vs Estate D ---
g_estate = (df_all.groupby("Estate")[["Rest_Gain","Unbd_Gain"]]
                  .max().reset_index().sort_values("Estate"))
plt.figure()
plt.plot(g_estate["Estate"], g_estate["Rest_Gain"], marker="o", label="Restricted (max gain)")
plt.plot(g_estate["Estate"], g_estate["Unbd_Gain"], marker="s", label="Unbounded (max gain)")
plt.xlabel("Estate $D$")
plt.ylabel("Maximum manipulability (gain)")
plt.title("CEA — Max manipulability vs estate $D$ (D=15..45)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(outdir / "cea_max_manip_vs_estate_multiE.png", dpi=200)
plt.close()
print("📈 cea_max_manip_vs_estate_multiE.png written")

# --- Plot 2: Max manipulability vs NumAgents ---
plt.figure()
for E, group in df_all.groupby("Estate"):
    g = (group.groupby("NumAgents")[["Rest_Gain","Unbd_Gain"]]
               .max().reset_index().sort_values("NumAgents"))
    plt.plot(g["NumAgents"], g["Rest_Gain"], marker="o", label=f"Restricted, D={E}")
    plt.plot(g["NumAgents"], g["Unbd_Gain"], marker="s", label=f"Unbounded, D={E}")
plt.xlabel("Number of agents (n)")
plt.ylabel("Maximum manipulability (gain)")
plt.title("CEA — Max manipulability vs number of agents (multi-E)")
plt.legend(ncol=2, fontsize=8); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(outdir / "cea_max_manip_vs_agents_multiE.png", dpi=200)
plt.close()
print("📈 cea_max_manip_vs_agents_multiE.png written")

# --- Plot 3: Each D one figure with all scenarios ---
def _parse_scenario_key(s):
    if isinstance(s, (list, tuple, np.ndarray)):
        vals = list(s)
    elif isinstance(s, str):
        try:
            vals = ast.literal_eval(s)
        except Exception:
            s2 = s.strip("[]() ")
            vals = [float(x) for x in s2.split(",")]
    else:
        vals = [float(s)]
    return tuple(round(float(v), 3) for v in vals)

df_all["ScenarioKey"] = df_all["Scenario"].apply(_parse_scenario_key)

plot_dir = outdir / "plots_per_estate_CEA"
plot_dir.mkdir(parents=True, exist_ok=True)

for E, gD in df_all.groupby("Estate"):
    scenarios = sorted(gD["ScenarioKey"].unique(), key=lambda x: str(x))
    n_scen = len(scenarios)

    ncols = 4
    nrows = math.ceil(n_scen / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4*ncols, 3*nrows), squeeze=False)

    for idx, scen_key in enumerate(scenarios):
        ax = axes[idx // ncols, idx % ncols]
        gg = (gD[gD["ScenarioKey"] == scen_key]
                .sort_values("Manipulator")
                .reset_index(drop=True))

        n  = len(gg)
        agents = np.arange(n)
        orig = gg["Orig_Best"].astype(float).to_numpy()
        rest = gg["Rest_Best"].astype(float).to_numpy()
        unbd = gg["Unbd_Best"].astype(float).to_numpy()

        width = 0.26
        x0, x1, x2 = agents - width, agents, agents + width

        ax.bar(x0, orig, width, label="Orig")
        ax.bar(x1, rest, width, label="Rest")
        ax.bar(x2, unbd, width, label="Unbd")

        ax.set_xticks(agents)
        ax.set_xticklabels([f"A{i}" for i in agents], fontsize=8)
        scen_str = ", ".join(f"{v:.2f}" for v in scen_key)
        ax.set_title(scen_str, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    for blank in range(n_scen, nrows*ncols):
        fig.delaxes(axes[blank // ncols, blank % ncols])

    fig.suptitle(f"CEA — Per-agent allocations (Estate D={E})", fontsize=14)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_file = plot_dir / f"CEA_allocations_D{E}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()

print(f"📊 Estate-level CEA figures saved to: {plot_dir.resolve()}")


# -------------------- 2) Scenarios --------------------
share_vectors = [
    np.array([0.08, 0.19, 0.07, 0.66]),
    np.array([0.48, 0.40, 0.05, 0.07]),
    np.array([0.22, 0.24, 0.09, 0.45]),
    np.array([0.35, 0.05, 0.24, 0.36]),
    np.array([0.15, 0.19, 0.13, 0.53]),
    np.array([0.21, 0.23, 0.08, 0.48]),
    np.array([0.10, 0.19, 0.09, 0.62]),
    np.array([0.18, 0.16, 0.09, 0.57]),
    np.array([0.22, 0.20, 0.14, 0.44]),
    np.array([0.35, 0.10, 0.24, 0.31]),
    np.array([0.10, 0.13, 0.09, 0.68]),
    np.array([0.28, 0.42, 0.25, 0.05]),
    np.array([0.18, 0.16, 0.19, 0.47]),
    np.array([0.25, 0.20, 0.11, 0.44]),
    np.array([0.35, 0.04, 0.30, 0.31]),
    np.array([0.15, 0.19, 0.06, 0.60]),
]

# -------------------- 3) Run all estates --------------------
records_all = []
estate_values = list(range(15, 50, 5))  # 15,20,...,45
ALPHA = 0.01
SCALE = 100

for E in estate_values:
    for sv in tqdm(share_vectors, desc=f"CEA scenarios @ D={E}"):
        R_truth = truthful_matrix(sv)
        orig_aw = awards_from_matrix(R_truth, estate=E, scale=SCALE)
        for m in range(len(sv)):
            rest_best, rest_rows, rest_sh = all_opt_rows_restricted(
                R_truth, m, estate=E, scale=SCALE, alpha=ALPHA)
            unb_best,  unb_rows           = all_opt_rows_unbounded(
                R_truth, m, estate=E, scale=SCALE, alpha=ALPHA)

            records_all.append({
                "Estate":           E,
                "Scenario":         np.round(sv, 3).tolist(),  # 展示保留3位即可
                "Manipulator":      m,
                "Orig_Best":        float(orig_aw[m]),         # << 不 round
                "Rest_Best":        float(rest_best),          # << 不 round
                "Unbd_Best":        float(unb_best),           # << 不 round
                "Optimal_Rows":     json.dumps(rest_rows),
                "Unbd_Rows":        json.dumps(unb_rows),
                "Impartial_Shares": json.dumps(rest_sh),
            })

df_all = pd.DataFrame(records_all)

# -------------------- 4) One figure: compare Original / Restricted / Unrestricted --------------------
df_plot = df_all.copy()
for col in ["Orig_Best", "Rest_Best", "Unbd_Best"]:
    df_plot.loc[~np.isfinite(df_plot[col]), col] = np.nan

# 先在 (Estate, Manipulator) 上对所有 Scenario 求平均
by_em = (df_plot.groupby(["Estate","Manipulator"], as_index=False)
                [["Orig_Best","Rest_Best","Unbd_Best"]].mean())

# 再在 Estate 上聚合（跨操作者取整体平均）
agg = (by_em.groupby("Estate")
            .agg(Orig=("Orig_Best","mean"),
                 Rest=("Rest_Best","mean"),
                 Unbd=("Unbd_Best","mean"))
            .reset_index())

x_base = agg["Estate"].to_numpy()
# 仅用于视觉分离，不改变数据
jitter = 0.12
x_blue  = x_base - jitter
x_orng  = x_base
x_green = x_base + jitter

plt.figure(figsize=(9,5))
plt.plot(x_blue,  agg["Orig"], marker="o",  linestyle="-",  linewidth=2,
         label="Original (no manipulation)", color="tab:blue")
plt.plot(x_orng,  agg["Rest"], marker="s",  linestyle="--", linewidth=2,
         label="Restricted manipulation",   color="tab:orange")
plt.plot(x_green, agg["Unbd"], marker="^",  linestyle=":",  linewidth=2,
         label="Unrestricted manipulation", color="tab:green")
plt.xlabel("Estate (D)")
plt.ylabel("Average payoff of manipulator")
# 4 位小数，确保 0.001 级别可读
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
plt.title("CEA – Average manipulator payoff (three conditions)")
plt.legend()
plt.tight_layout()
plt.savefig("cea_three_conditions_one_plot_precise.png", dpi=240)
plt.close()
print("✅ Saved: cea_three_conditions_one_plot_precise.png")