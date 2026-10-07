"""print_summary.py [alpha] - one table per dataset from scores/*_cp_results.json (all methods)."""
import glob, json, sys
a = sys.argv[1] if len(sys.argv) > 1 else "0.1"
P = lambda x: "  -  " if x is None else f"{x:.3f}"
for f in sorted(glob.glob("scores/copy_*_cp_results.json")):
    R = json.load(open(f)); B = R["by_alpha"].get(a)
    if not B: continue
    print(f"#### {f}  Q={R['Q']} E={R['E']} groups={R['group_sizes']} edges={R.get('edges')}  alpha={a}")
    print(f"{'method':<10} {'cov':>6} {'worstG':>6} {'worstF':>6} {'g0':>6} {'g1':>6} {'g2':>6} {'never':>6} {'size':>9} {'size/E':>7} {'chr_cov':>7} {'chr_g0':>6}")
    for m, s in B["random_splits"].items():
        g = lambda k: s.get(k, {}).get("mean") if isinstance(s.get(k), dict) else None
        ch = B.get("chronological", {}).get(m, {})
        sz = g("size_mean")
        print(f"{m:<10} {P(g('cov')):>6} {P(g('worst_group_cov')):>6} {P(g('worst_fine_cov')):>6} {P(g('cov_g0')):>6} "
              f"{P(g('cov_g1')):>6} {P(g('cov_g2')):>6} {P(g('cov_fine0')):>6} {('-' if sz is None else f'{sz:.0f}'):>9} "
              f"{P(g('size_frac_vocab')):>7} {P(ch.get('cov')):>7} {P(ch.get('cov_g0')):>6}")
    print("oracle m*/|E_k|:", {k: round(v["m_star_frac"], 3) for k, v in B.get("rank_oracle_split0", {}).items()})
    dr = B.get("chronological", {}).get("std", {}).get("drift")
    if dr: print(f"drift(std): pi_cal={dr['pi_cal']:.3f} pi_test={dr['pi_test']:.3f} corr={dr['corr_obs_vs_pred']} mae={dr['mae_obs_vs_pred']}")
    print()
