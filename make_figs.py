#!/usr/bin/env python3
"""
make_figs.py - paper figures from cp_harness JSON outputs.

  python make_figs.py --alpha 0.1 --out paper \
      "ICEWS18|CyGNet|scores/copy_icews18_cp_results.json" "GDELT|CyGNet|scores/copy_gdelt_cp_results.json"

fig_tradeoff.pdf : zero-shot coverage vs mean set size (fraction of vocabulary), one panel per run
fig_drift.pdf    : per-window observed vs Prop.-1-predicted coverage of split CP (chronological split)
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated reference categorical palette (fixed order) + distinct markers as secondary encoding
STYLE = {
    "std":       ("Split CP",    "#2a78d6", "o"),
    "pas":       ("PAS",         "#eb6834", "s"),
    "interp":    ("Interp-Q",    "#1baf7a", "^"),
    "clustered": ("Clustered",   "#eda100", "v"),
    "mond_f":    ("Mondrian-F",  "#e87ba4", "D"),
    "mond_bo":   ("FRB",         "#008300", "P"),
    "fcp":       ("Feature-norm.", "#52514e", "X"),
    "fnm":       ("FNM (ours)",  "#4a3aa7", "*"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--alpha", default="0.1")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = [(s.split("|")[0], s.split("|")[1], json.load(open(s.split("|")[2]))) for s in a.runs]
    target = 1 - float(a.alpha)

    # ---- trade-off
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(3.25 * min(n, 2) if n <= 2 else 2.3 * n, 2.3), squeeze=False)
    for ax, (ds, sc, R) in zip(axes[0], runs):
        S = R["by_alpha"][a.alpha]["random_splits"]
        ax.axhline(target, color=MUTED, lw=1, ls="--", zorder=1)
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        for m, (lab, col, mk) in STYLE.items():
            if m not in S or S[m].get("size_frac_vocab") is None:
                continue
            x, y = S[m]["size_frac_vocab"]["mean"], S[m]["cov_g0"]["mean"]
            if x is None or y is None:
                continue
            ax.scatter(max(x, 1e-4), y, s=90 if mk == "*" else 30, color=col, marker=mk,
                       edgecolor="white", linewidth=1.2, zorder=3, label=lab)
        ax.set_xscale("log")
        from matplotlib.ticker import NullFormatter
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{ds} / {sc}", fontsize=8, color=INK)
        ax.set_xlabel("mean set size / $|\\mathcal{E}|$")
    axes[0][0].set_ylabel("zero-shot coverage")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(a.out, "fig_tradeoff.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(a.out, "fig_tradeoff.png"), dpi=200, bbox_inches="tight")

    # ---- drift (Prop. 1)
    fig, ax = plt.subplots(figsize=(3.25, 2.6))
    lo, hi = 1, 0
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    mks = ["o", "s", "^", "v", "D", "P"]
    for i, (ds, sc, R) in enumerate(runs):
        st = R["by_alpha"][a.alpha].get("chronological", {}).get("std", {})
        dr = (st.get("drift_k") or {}).get("fine") or st.get("drift")
        if not dr or not dr["windows"]:
            continue
        x = [w["cov_pred"] for w in dr["windows"]]; y = [w["cov_obs"] for w in dr["windows"]]
        lo, hi = min(lo, *x, *y), max(hi, *x, *y)
        r = dr.get("corr", dr.get("corr_obs_vs_pred"))
        ax.scatter(x, y, s=16, color=cols[i % 6], marker=mks[i % 6], edgecolor="white", linewidth=0.8,
                   label=f"{ds}/{sc} ($r$={r:.2f})" if r is not None else f"{ds}/{sc}", zorder=3)
    pad = 0.02
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1, ls="--", zorder=2)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel("predicted coverage (Prop. 1, fine strata)")
    ax.set_ylabel("observed coverage")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_drift.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(a.out, "fig_drift.png"), dpi=200, bbox_inches="tight")
    print("wrote", a.out + "/fig_tradeoff.pdf", a.out + "/fig_drift.pdf")


if __name__ == "__main__":
    main()
