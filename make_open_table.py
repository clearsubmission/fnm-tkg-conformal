"""make_open_table.py OUT.tex "DS|closed.json|open.json" ... -> open-label comparison table + macros"""
import json, sys
out, runs = sys.argv[1], sys.argv[2:]
P = lambda x: "--" if x is None else f"{100*x:.1f}"
L = ["% auto-generated", "\\begin{tabular}{ll rrrr}", "\\toprule",
     "Data & Method & Cov.$_\\text{new}$ & Worst bin & $|C|$ (closed $\\to$ open) & NEW in set \\\\", "\\midrule"]
mac = []
names = {"std": "Split CP", "mond_f": "Mondrian-F", "fnm": "\\textbf{FNM (ours)}"}
for i, r in enumerate(runs):
    ds, fc, fo = r.split("|")
    C = json.load(open(fc))["by_alpha"]["0.1"]["random_splits"]
    O = json.load(open(fo))["by_alpha"]["0.1"]["random_splits"]
    for j, m in enumerate(["std", "mond_f", "fnm"]):
        g = lambda S, k: S[m][k]["mean"] if k in S[m] else None
        sc, so = g(C, "size_mean"), g(O, "size_mean")
        red = 1 - so / sc
        lead = ds if j == 0 else ""
        L.append(f"{lead} & {names[m]} & {P(g(O,'cov_g0'))} & {P(g(O,'worst_fine_cov'))} & "
                 f"{sc:,.0f} $\\to$ {so:,.0f} ($-${100*red:.0f}\\%) & {P(g(O,'inset_frac_cg0'))} \\\\")
        tag = {"std": "std", "mond_f": "mondf", "fnm": "fnm"}[m]
        mac += [f"\\expandafter\\def\\csname open@{ds}@{tag}@red\\endcsname{{{100*red:.0f}}}",
                f"\\expandafter\\def\\csname open@{ds}@{tag}@size\\endcsname{{{so:,.0f}}}".replace(",", "{,}"),
                f"\\expandafter\\def\\csname open@{ds}@{tag}@worstf\\endcsname{{{P(g(O,'worst_fine_cov'))}}}",
                f"\\expandafter\\def\\csname open@{ds}@{tag}@newin\\endcsname{{{P(g(O,'inset_frac_cg0'))}}}",
                f"\\expandafter\\def\\csname open@{ds}@{tag}@covnew\\endcsname{{{P(g(O,'cov_g0'))}}}"]
    L.append("\\midrule" if i < len(runs) - 1 else "\\bottomrule")
L.append("\\end{tabular}")
open(out, "w").write("\n".join(L) + "\n")
open(out.replace(".tex", "_macros.tex"), "w").write(
    "\\makeatletter\n\\providecommand{\\OL}[3]{\\@ifundefined{open@#1@#2@#3}{\\textbf{??}}{\\csname open@#1@#2@#3\\endcsname}}\n"
    + "\n".join(mac) + "\n\\makeatother\n")
print("wrote", out)
