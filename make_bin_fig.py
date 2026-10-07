#!/usr/bin/env python3
"""make_bin_fig.py: coverage per fine training-frequency bin (mean +/- s.d. over random splits).
  python make_bin_fig.py --out paper "ICEWS18|path.json" "GDELT|path.json" "WIKI|path.json" """
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINS = ["0", "1", "2–4", "5–9", "10–24", "25–49", "50–99", "100+"]
STYLE = [("std", "Split CP", "#2a78d6", "o"), ("mond_f", "Mondrian-F", "#eb6834", "s"),
         ("fcp", "Feature-norm. (RCP-style)", "#1baf7a", "^"), ("fnm", "FNM (ours)", "#4a3aa7", "D")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--alpha", default="0.1")
ap.add_argument("--out", default="paper"); a = ap.parse_args()
fig, axes = plt.subplots(1, len(a.runs), figsize=(7.0, 2.2), sharey=True, squeeze=False)
for ax, spec in zip(axes[0], a.runs):
    ds, path = spec.split("|"); S = json.load(open(path))["by_alpha"][a.alpha]["random_splits"]
    ax.axhline(1 - float(a.alpha), color=MUTED, lw=1, ls="--", zorder=1)
    ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
    for j, (m, lab, col, mk) in enumerate(STYLE):
        if m not in S: continue
        xs, ys, es = [], [], []
        for k in range(len(BINS)):
            c = S[m].get(f"cov_fine{k}")
            if c and c["mean"] is not None and S[m][f"n_fine{k}"]["mean"] >= 10:
                xs.append(k + (j - 1.5) * 0.09); ys.append(c["mean"]); es.append(c["std"] or 0)
        ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, ms=4, lw=1.4, elinewidth=0.8, capsize=0,
                    markeredgecolor="white", markeredgewidth=0.6, label=lab, zorder=3)
    n = [S["std"][f"n_fine{k}"]["mean"] for k in range(len(BINS))]
    ax.set_xticks(range(len(BINS)))
    fmt = lambda x: f"{x/1000:.0f}k" if x >= 10000 else (f"{x/1000:.1f}k" if x >= 1000 else f"{int(round(x))}")
    ax.set_xticklabels([f"{b}\n{fmt(x)}" for b, x in zip(BINS, n)], fontsize=5.5)
    for xb in (0.5, 3.5): ax.axvline(xb, color=GRID, lw=1.0, ls=":", zorder=0)
    ax.set_title(ds, fontsize=8, color=INK); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("training occurrences of answer / # eval. queries", fontsize=6)
axes[0][0].set_ylabel("coverage")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.08, 1, 1))
os.makedirs(a.out, exist_ok=True)
fig.savefig(os.path.join(a.out, "fig_bins.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(a.out, "fig_bins.png"), dpi=200, bbox_inches="tight")
print("wrote", os.path.join(a.out, "fig_bins.pdf"))
