"""Generate all ILS and GRASP analysis figures (PT-BR and English).

Reads experiment outputs from ``results/ils`` and ``results/grasp`` plus the
walkability datasets under ``data/location`` and produces every figure used in
the thesis chapters (boxplots, convergence curves, alpha analysis, scatter
plots, per-hexagon IQC distributions, skewness/kurtosis migration).

Usage (from the repository root):

    python scripts/generate_figures.py                 # both languages
    python scripts/generate_figures.py --lang pt       # only PT-BR
    python scripts/generate_figures.py --lang en       # only English
    python scripts/generate_figures.py --refresh       # ignore cached derived data

Outputs:
    docs/figures/*.pdf|png       (PT-BR, the paths referenced by the .tex files)
    docs/figures/en/*.pdf|png    (English)

All text shown inside the figures comes from the LABELS dictionary below, so
titles/legends/axis names can be edited in one place per language.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metaheuristics.core.budget import POI_DIMENSION_COLUMNS
from metaheuristics.core.evaluation import (
    allocation_items_to_candidate_matrix,
    build_final_indicator_matrix_nd,
    build_objective_state_nd,
    get_available_dimensions,
    objective_function,
)

# --------------------------------------------------------------------------- #
# Style (palette validated for color-vision deficiency; see docs)             #
# --------------------------------------------------------------------------- #
INK = "#0b0b0b"; SEC = "#52514e"; MUT = "#898781"; GRID = "#e1e0d9"; BASE = "#c3c2b7"
BLUE = "#2a78d6"; AQUA = "#1baf7a"; YEL = "#eda100"; GRAY = "#898781"

plt.rcParams.update({
    "font.size": 8.5, "font.family": "sans-serif", "text.color": INK,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": SEC,
    "axes.titlesize": 9, "axes.titlecolor": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUT, "ytick.color": MUT,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

LOCS = ["av_paulista", "itajuba_centro", "metro_ana_rosa", "milao_italia", "shinjuku_toquio"]
PROFS = ["athlete", "average_adult", "elderly"]
PROFCOL = {"athlete": BLUE, "average_adult": AQUA, "elderly": YEL}

# Instances highlighted in convergence / distribution / scatter figures.
CONV_INSTANCES = [("itajuba_centro", "average_adult"), ("milao_italia", "elderly")]
ALPHA_SCATTER_INSTANCES = [("itajuba_centro", "average_adult"), ("milao_italia", "elderly")]
CONSTR_LS_INSTANCE = ("metro_ana_rosa", "average_adult")
INIT_BEST_INSTANCES = [("av_paulista", "elderly"), ("itajuba_centro", "average_adult")]

# --------------------------------------------------------------------------- #
# Every visible string, per language. Edit freely.                            #
# --------------------------------------------------------------------------- #
LABELS = {
    "pt": {
        "loc": {"av_paulista": "Av. Paulista", "itajuba_centro": "Itajubá",
                "metro_ana_rosa": "Ana Rosa", "milao_italia": "Milão",
                "shinjuku_toquio": "Shinjuku"},
        "prof": {"athlete": "atleta", "average_adult": "adulto", "elderly": "idoso"},
        "prof_short": {"athlete": "atl", "average_adult": "adu", "elderly": "ido"},
        "improvement_pct": "melhoria sobre a base (%)",
        "strength2": "força 2", "strength10": "força 10",
        "evals_log": "avaliações (escala log)",
        "best_sum_mean": "melhor soma de IQC (média)",
        "alpha_best_iter": r"$\alpha$ da melhor iteração",
        "seeds": "sementes",
        "uniform": "uniforme",
        "agg_dist": "Distribuição agregada",
        "by_instance": "Por instância",
        "mean_best_alpha": r"média do melhor $\alpha$ (IC 95%)",
        "iter_alpha": r"$\alpha$ da iteração",
        "after_ls": "soma de IQC após busca local",
        "rho_mean": r"$\bar{\rho}_s$",
        "constructed": "soma de IQC da solução construída",
        "after_ls_y": "soma de IQC após busca local",
        "before": "antes", "after": "depois",
        "hex_iqc": "IQC do hexágono",
        "hexagons": "hexágonos",
        "skewness": r"assimetria ($g_1$)",
        "kurtosis": r"excesso de curtose ($g_2$)",
        "initial_obj": "objetivo da solução inicial",
        "best_obj": "melhor objetivo da execução",
    },
    "en": {
        "loc": {"av_paulista": "Av. Paulista", "itajuba_centro": "Itajubá",
                "metro_ana_rosa": "Ana Rosa", "milao_italia": "Milan",
                "shinjuku_toquio": "Shinjuku"},
        "prof": {"athlete": "athlete", "average_adult": "adult", "elderly": "elderly"},
        "prof_short": {"athlete": "ath", "average_adult": "adu", "elderly": "eld"},
        "improvement_pct": "improvement over baseline (%)",
        "strength2": "strength 2", "strength10": "strength 10",
        "evals_log": "evaluations (log scale)",
        "best_sum_mean": "best IQC sum (mean)",
        "alpha_best_iter": r"$\alpha$ of the best iteration",
        "seeds": "seeds",
        "uniform": "uniform",
        "agg_dist": "Aggregate distribution",
        "by_instance": "By instance",
        "mean_best_alpha": r"mean best $\alpha$ (95% CI)",
        "iter_alpha": r"iteration $\alpha$",
        "after_ls": "IQC sum after local search",
        "rho_mean": r"$\bar{\rho}_s$",
        "constructed": "IQC sum of constructed solution",
        "after_ls_y": "IQC sum after local search",
        "before": "before", "after": "after",
        "hex_iqc": "hexagon IQC",
        "hexagons": "hexagons",
        "skewness": r"skewness ($g_1$)",
        "kurtosis": r"excess kurtosis ($g_2$)",
        "initial_obj": "initial solution objective",
        "best_obj": "best objective of the run",
    },
}


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #
def load_summaries(method, pert=None):
    """Load runs_summary.csv per (location, profile); filter ILS by strength."""
    out = {}
    for r in (ROOT / "results" / method).glob("*/*/*/*"):
        cfg_path = r / "config" / "experiment_config.json"
        if not cfg_path.exists():
            continue
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        if pert is not None and int(cfg["runtime_config"].get("perturbation_strength", -1)) != pert:
            continue
        out[(r.parts[-4], r.parts[-3])] = dict(
            df=pd.read_csv(r / "summary" / "runs_summary.csv"), dir=r,
            base=cfg["baseline_iqc_total"])
    return out


def load_traj_sample(rdir, cols, n_seeds=25):
    frames = []
    for sd in sorted((rdir / "seed_runs").glob("seed_*"))[:n_seeds]:
        frames.append(pd.read_csv(sd / "trajectory.csv", usecols=cols))
    return pd.concat(frames, ignore_index=True)


def dense_convergence(rdir, grid):
    """Mean and quartiles of best-so-far value at each evaluation checkpoint."""
    mats = []
    for sd in sorted((rdir / "seed_runs").glob("seed_*")):
        t = pd.read_csv(sd / "trajectory.csv", usecols=["eval_count", "best_sum_iqc"])
        ev = t["eval_count"].values
        bs = t["best_sum_iqc"].values
        mats.append([bs[max(np.searchsorted(ev, g, side="right") - 1, 0)] for g in grid])
    a = np.array(mats)
    return a.mean(0), np.percentile(a, 25, 0), np.percentile(a, 75, 0)


def compute_iqc_arrays():
    """Per-hexagon IQC before and after the global best allocation of each method."""
    ils2 = load_summaries("ils", 2)
    grasp = load_summaries("grasp")
    arrays, shape = {}, []
    for (loc, prof), g in sorted(grasp.items()):
        ddir = ROOT / "data" / "location" / loc / prof / "resolution_9" / "2000" / "csv" / "walkability_index"
        wk = pd.read_csv(next(ddir.glob("*walkability_index*.csv")))
        ht = pd.read_csv(next(ddir.glob("*hex_time_matrix*.csv")))
        dims = get_available_dimensions(wk, POI_DIMENSION_COLUMNS)
        st = build_objective_state_nd(wk, ht, dims)
        before = np.asarray(objective_function(
            final_indicator_matrix=st.baseline_matrix.copy())["iqc_values"], float)

        def after(run_dir):
            items = pd.read_csv(run_dir / "summary" / "global_best_allocation.csv").to_dict("records")
            cm = allocation_items_to_candidate_matrix(allocation_items=items, objective_state=st)
            fm = build_final_indicator_matrix_nd(candidate_matrix=cm, objective_state=st)
            return np.asarray(objective_function(final_indicator_matrix=fm)["iqc_values"], float)

        arrays[f"{loc}|{prof}|before"] = before
        arrays[f"{loc}|{prof}|grasp"] = after(g["dir"])
        arrays[f"{loc}|{prof}|ils"] = after(ils2[(loc, prof)]["dir"])
        row = dict(loc=loc, profile=prof)
        for tag in ("before", "grasp", "ils"):
            x = arrays[f"{loc}|{prof}|{tag}"]
            row[tag] = dict(skew=float(stats.skew(x, bias=False)),
                            kurt=float(stats.kurtosis(x, bias=False)))
        shape.append(row)
        print(f"  IQC arrays: {loc}/{prof}")
    return arrays, shape


def build_cache(cache_dir, refresh):
    """Derived data that is slow to compute (trajectory scans, IQC re-evaluation)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz = cache_dir / "derived.npz"
    meta = cache_dir / "derived.json"
    if npz.exists() and meta.exists() and not refresh:
        print("Using cached derived data (pass --refresh to recompute).")
        data = dict(np.load(npz))
        info = json.load(open(meta, encoding="utf-8"))
        return data, info

    print("Computing derived data (trajectories + IQC re-evaluation)...")
    grasp = load_summaries("grasp")
    ils2 = load_summaries("ils", 2)
    ils10 = load_summaries("ils", 10)

    grid = np.unique(np.round(np.logspace(np.log10(30), np.log10(30000), 40)).astype(int))
    data = {"grid": grid}
    for loc, prof in CONV_INSTANCES:
        for key, runs in (("grasp", grasp), ("ils2", ils2), ("ils10", ils10)):
            m, q1, q3 = dense_convergence(runs[(loc, prof)]["dir"], grid)
            data[f"conv|{loc}|{prof}|{key}|mean"] = m
            data[f"conv|{loc}|{prof}|{key}|q25"] = q1
            data[f"conv|{loc}|{prof}|{key}|q75"] = q3
        print(f"  convergence: {loc}/{prof}")

    arrays, shape = compute_iqc_arrays()
    data.update(arrays)
    data["alpha_pool"] = np.concatenate(
        [grasp[k]["df"]["best_alpha"].values for k in sorted(grasp)])

    np.savez_compressed(npz, **data)
    info = {"shape": shape}
    json.dump(info, open(meta, "w", encoding="utf-8"))
    return dict(np.load(npz)), info


# --------------------------------------------------------------------------- #
# Figure helpers                                                               #
# --------------------------------------------------------------------------- #
def styled_box(ax, data, positions, color, width=0.55):
    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker="o", markersize=2, markerfacecolor=color,
                                    markeredgecolor="none", alpha=0.45),
                    medianprops=dict(color=INK, linewidth=1.1),
                    whiskerprops=dict(color=SEC, linewidth=0.8),
                    capprops=dict(color=SEC, linewidth=0.8),
                    boxprops=dict(linewidth=0.8))
    for b in bp["boxes"]:
        b.set_facecolor(color)
        b.set_alpha(0.55)
        b.set_edgecolor(SEC)


def save(fig, out_dir, name, formats):
    for ext in formats:
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight",
                    **({"dpi": 150} if ext == "png" else {}))
    plt.close(fig)
    print(f"  saved {name}")


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def fig_grasp_box(G, L, out, formats):
    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    for gi, loc in enumerate(LOCS):
        for pi, p in enumerate(PROFS):
            styled_box(ax, [G[(loc, p)]["df"]["delta_pct_vs_baseline"].values],
                       [gi * 4 + pi], PROFCOL[p])
    ax.set_xticks([gi * 4 + 1 for gi in range(5)])
    ax.set_xticklabels([L["loc"][l] for l in LOCS])
    ax.set_ylabel(L["improvement_pct"])
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=PROFCOL[p], alpha=0.55, edgecolor=SEC)
               for p in PROFS]
    ax.legend(handles, [L["prof"][p] for p in PROFS], loc="upper right", ncols=3, fontsize=8)
    ax.set_axisbelow(True)
    save(fig, out, "grasp_box_delta", formats)


def fig_ils_box(I2, I10, L, out, formats):
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    labels, xticks, centers = [], [], []
    pos = 0.0
    for loc in LOCS:
        start = pos
        for p in PROFS:
            styled_box(ax, [I2[(loc, p)]["df"]["delta_pct_vs_baseline"].values], [pos], BLUE, 0.8)
            styled_box(ax, [I10[(loc, p)]["df"]["delta_pct_vs_baseline"].values], [pos + 0.9], AQUA, 0.8)
            xticks.append(pos + 0.45)
            labels.append(L["prof_short"][p])
            pos += 2.4
        centers.append((start + pos - 2.4 + 0.9) / 2)
        pos += 1.4
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, fontsize=7)
    ylim = ax.get_ylim()
    for gi, loc in enumerate(LOCS):
        ax.text(centers[gi], ylim[1] * 0.98, L["loc"][loc], ha="center", va="top",
                fontsize=8, color=SEC)
    ax.set_ylabel(L["improvement_pct"])
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.55, edgecolor=SEC)
               for c in (BLUE, AQUA)]
    ax.legend(handles, [L["strength2"], L["strength10"]], loc="center right", fontsize=8)
    ax.set_axisbelow(True)
    save(fig, out, "ils_box_strength", formats)


def _conv_panel(ax, data, grid, keys, L, title):
    for key, color, lab in keys:
        ax.plot(grid, data[f"{key}|mean"], color=color, lw=1.6, label=lab)
        ax.fill_between(grid, data[f"{key}|q25"], data[f"{key}|q75"],
                        color=color, alpha=0.16, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel(L["evals_log"])
    ax.set_title(title, loc="left")
    ax.set_axisbelow(True)


def fig_convergence(data, L, out, formats):
    grid = data["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    for ax, (loc, prof) in zip(axes, CONV_INSTANCES):
        _conv_panel(ax, data, grid, [(f"conv|{loc}|{prof}|grasp", BLUE, None)], L,
                    f"{L['loc'][loc]} / {L['prof'][prof]}")
    axes[0].set_ylabel(L["best_sum_mean"])
    save(fig, out, "grasp_convergence", formats)

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    for ax, (loc, prof) in zip(axes, CONV_INSTANCES):
        _conv_panel(ax, data, grid,
                    [(f"conv|{loc}|{prof}|ils2", BLUE, L["strength2"]),
                     (f"conv|{loc}|{prof}|ils10", AQUA, L["strength10"])], L,
                    f"{L['loc'][loc]} / {L['prof'][prof]}")
    axes[0].set_ylabel(L["best_sum_mean"])
    axes[0].legend(loc="lower right", fontsize=8)
    save(fig, out, "ils_convergence", formats)


def fig_grasp_alpha(G, data, L, out, formats):
    pool = data["alpha_pool"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9),
                             gridspec_kw=dict(width_ratios=[1, 1.05], wspace=0.62))
    ax = axes[0]
    ax.hist(pool, bins=20, range=(0, 1), color=BLUE, alpha=0.6,
            edgecolor="white", linewidth=0.5)
    ax.axhline(len(pool) / 20, color=MUT, lw=1.1, ls="--")
    ax.text(0.03, len(pool) / 20 + 4, L["uniform"], ha="left", va="bottom",
            fontsize=7.5, color=SEC)
    ax.set_xlabel(L["alpha_best_iter"])
    ax.set_ylabel(L["seeds"])
    ax.set_title(L["agg_dist"], loc="left")

    ax = axes[1]
    rows = []
    for loc in LOCS:
        for p in PROFS:
            d = G[(loc, p)]["df"]["best_alpha"]
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            rows.append((f"{L['loc'][loc]} - {L['prof'][p]}", d.mean(), h))
    rows.sort(key=lambda r: r[1])
    ys = np.arange(len(rows))
    ax.errorbar([r[1] for r in rows], ys, xerr=[r[2] for r in rows], fmt="o",
                color=BLUE, ms=3.5, elinewidth=1, capsize=0)
    ax.axvline(0.5, color=MUT, lw=1, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    ax.set_xlabel(L["mean_best_alpha"])
    ax.set_title(L["by_instance"], loc="left")
    ax.set_xlim(0, 1)
    ax.set_axisbelow(True)
    save(fig, out, "grasp_alpha", formats)


def fig_grasp_alpha_scatter(G, L, out, formats):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
    for ax, (loc, p) in zip(axes, ALPHA_SCATTER_INSTANCES):
        t = load_traj_sample(G[(loc, p)]["dir"], ["alpha_used", "after_ls_sum_iqc"])
        # Mean per-seed Spearman correlation shown in the panel title.
        rhos = []
        for sd in sorted((G[(loc, p)]["dir"] / "seed_runs").glob("seed_*"))[:100]:
            tt = pd.read_csv(sd / "trajectory.csv", usecols=["alpha_used", "after_ls_sum_iqc"])
            if len(tt) >= 10:
                rhos.append(stats.spearmanr(tt["alpha_used"], tt["after_ls_sum_iqc"])[0])
        rho = float(np.mean(rhos))
        ax.scatter(t["alpha_used"], t["after_ls_sum_iqc"], s=4, color=BLUE,
                   alpha=0.18, linewidths=0)
        bins = np.linspace(0, 1, 11)
        mids = (bins[:-1] + bins[1:]) / 2
        bm = [t.loc[(t["alpha_used"] >= a) & (t["alpha_used"] < b), "after_ls_sum_iqc"].mean()
              for a, b in zip(bins[:-1], bins[1:])]
        ax.plot(mids, bm, color=INK, lw=1.6)
        ax.set_title(f"{L['loc'][loc]} / {L['prof'][p]}  ({L['rho_mean']}={rho:+.2f})", loc="left")
        ax.set_xlabel(L["iter_alpha"])
        ax.set_axisbelow(True)
    axes[0].set_ylabel(L["after_ls"])
    save(fig, out, "grasp_alpha_scatter", formats)


def fig_grasp_constr_ls(G, L, out, formats):
    loc, p = CONSTR_LS_INSTANCE
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    t = load_traj_sample(G[(loc, p)]["dir"],
                         ["construction_sum_iqc", "after_ls_sum_iqc"], n_seeds=30)
    lims = [min(t["construction_sum_iqc"].min(), t["after_ls_sum_iqc"].min()) - 0.3,
            max(t["construction_sum_iqc"].max(), t["after_ls_sum_iqc"].max()) + 0.3]
    ax.plot(lims, lims, color=MUT, lw=1, ls="--")
    ax.scatter(t["construction_sum_iqc"], t["after_ls_sum_iqc"], s=4, color=BLUE,
               alpha=0.18, linewidths=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(L["constructed"])
    ax.set_ylabel(L["after_ls_y"])
    ax.text(lims[0] + 0.2, lims[1] - 0.35, "y = x", fontsize=8, color=SEC)
    ax.set_title(f"{L['loc'][loc]} / {L['prof'][p]}", loc="left")
    ax.set_axisbelow(True)
    save(fig, out, "grasp_constr_ls", formats)


def fig_iqc_dist(data, info, method, L, out, formats):
    shp = {(s["loc"], s["profile"]): s for s in info["shape"]}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
    for ax, (loc, p) in zip(axes, CONV_INSTANCES):
        b = data[f"{loc}|{p}|before"]
        a = data[f"{loc}|{p}|{method}"]
        bins = np.linspace(min(b.min(), a.min()), max(b.max(), a.max()), 18)
        ax.hist(b, bins=bins, color=GRAY, alpha=0.45, edgecolor="white",
                linewidth=0.4, label=L["before"])
        ax.hist(a, bins=bins, color=BLUE, alpha=0.55, edgecolor="white",
                linewidth=0.4, label=L["after"])
        s = shp[(loc, p)]
        txt = (f"{L['before']}:  g1={s['before']['skew']:+.2f}  g2={s['before']['kurt']:+.2f}\n"
               f"{L['after']}: g1={s[method]['skew']:+.2f}  g2={s[method]['kurt']:+.2f}")
        ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=SEC,
                bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.3"))
        ax.set_title(f"{L['loc'][loc]} / {L['prof'][p]}", loc="left")
        ax.set_xlabel(L["hex_iqc"])
        ax.set_axisbelow(True)
    axes[0].set_ylabel(L["hexagons"])
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, out, f"{method}_iqc_dist", formats)


def fig_skewkurt(info, method, L, out, formats):
    shape = info["shape"]
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    for s in shape:
        ax.annotate("", xy=(s[method]["skew"], s[method]["kurt"]),
                    xytext=(s["before"]["skew"], s["before"]["kurt"]),
                    arrowprops=dict(arrowstyle="-|>", color=BASE, lw=0.9,
                                    shrinkA=3, shrinkB=3))
    ax.scatter([s["before"]["skew"] for s in shape], [s["before"]["kurt"] for s in shape],
               s=22, facecolors="white", edgecolors=GRAY, linewidths=1.2,
               label=L["before"], zorder=3)
    ax.scatter([s[method]["skew"] for s in shape], [s[method]["kurt"] for s in shape],
               s=22, color=BLUE, label=L["after"], zorder=3)
    ax.set_xlabel(L["skewness"])
    ax.set_ylabel(L["kurtosis"])
    ax.legend(loc="upper left", fontsize=8)
    ax.set_axisbelow(True)
    save(fig, out, f"{method}_skewkurt", formats)


def fig_ils_init_best(I2, L, out, formats):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
    for ax, (loc, p) in zip(axes, INIT_BEST_INSTANCES):
        df = I2[(loc, p)]["df"]
        rho, pv = stats.spearmanr(df["initial_objective_value"], df["best_objective_value"])
        ax.scatter(df["initial_objective_value"], df["best_objective_value"],
                   s=9, color=BLUE, alpha=0.55, linewidths=0)
        ax.set_title(f"{L['loc'][loc]} / {L['prof'][p]}  "
                     rf"($\rho_s$={rho:+.2f}, p={pv:.1g})", loc="left")
        ax.set_xlabel(L["initial_obj"])
        ax.set_axisbelow(True)
    axes[0].set_ylabel(L["best_obj"])
    save(fig, out, "ils_init_best", formats)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=["pt", "en", "both"], default="both")
    parser.add_argument("--out", default=str(ROOT / "docs" / "figures"),
                        help="output directory for PT figures (EN goes to <out>/en)")
    parser.add_argument("--formats", default="pdf,png",
                        help="comma-separated list, e.g. pdf,png")
    parser.add_argument("--refresh", action="store_true",
                        help="recompute cached derived data")
    args = parser.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    out_pt = Path(args.out)
    out_en = out_pt / "en"

    G = load_summaries("grasp")
    I2 = load_summaries("ils", 2)
    I10 = load_summaries("ils", 10)
    if not G or not I2 or not I10:
        raise SystemExit("Missing results: expected runs under results/grasp and results/ils.")

    data, info = build_cache(out_pt / "_cache", args.refresh)

    langs = ["pt", "en"] if args.lang == "both" else [args.lang]
    for lang in langs:
        L = LABELS[lang]
        out = out_pt if lang == "pt" else out_en
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Generating figures [{lang}] -> {out} ===")
        fig_grasp_box(G, L, out, formats)
        fig_ils_box(I2, I10, L, out, formats)
        fig_convergence(data, L, out, formats)
        fig_grasp_alpha(G, data, L, out, formats)
        fig_grasp_alpha_scatter(G, L, out, formats)
        fig_grasp_constr_ls(G, L, out, formats)
        fig_iqc_dist(data, info, "grasp", L, out, formats)
        fig_iqc_dist(data, info, "ils", L, out, formats)
        fig_skewkurt(info, "grasp", L, out, formats)
        fig_skewkurt(info, "ils", L, out, formats)
        fig_ils_init_best(I2, L, out, formats)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
