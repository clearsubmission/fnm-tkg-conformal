"""Compact main-text rolling table: never-seen coverage / worst bin (window means) per run."""
import sys
txt = open(sys.argv[1]).read(); out = sys.argv[2]
NAMES = {"std": "Split CP", "aci": "ACI", "mond_f": "Mondrian-F", "aci_mond": "Mondrian-ACI",
         "fcp": "Feat.-norm.", "fnm": "FNM", "da_fnm": "DA-FNM"}
ORDER = ["std", "aci", "mond_f", "aci_mond", "fcp", "fnm", "da_fnm"]
RUNS = [("copy_icews18", "ICEWS18"), ("copy_gdelt", "GDELT"), ("copy_wiki", "WIKI"), ("cenet_icews18", "ICEWS18$^\\dagger$")]
D = {}
for b in txt.strip().split("## ")[1:]:
    lines = b.strip().split("\n"); key = lines[0].split()[0]; D[key] = {}
    for r in lines[1:]:
        f = r.split(); D[key][f[0]] = {"cov0": f[4].split("/")[0], "worst": f[6].split("/")[0], "size": f[10]}
L = ["\\begin{tabular}{l " + " ".join(["rr"] * len(RUNS)) + "}", "\\toprule",
     " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in RUNS) + " \\\\",
     "".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(RUNS))),
     "Method & " + " & ".join(["zs & bin"] * len(RUNS)) + " \\\\", "\\midrule"]
for m in ORDER:
    row = [NAMES[m]]
    for k, _ in RUNS:
        v = D[k][m]; row += [v["cov0"], v["worst"]]
    L.append(" & ".join(row) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(out, "w").write("\n".join(L) + "\n"); print("wrote", out)
