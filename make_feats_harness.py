"""Writes cp_harness_feats.py = cp_harness.py + --feats {freq,rec,freq+rec} for the FNM/Feature-norm. regressor.
The original cp_harness.py is not modified."""
import re
s = open("cp_harness.py").read()
old = re.search(r"def _feat\(D, u, e\):\n(?:    .*\n)+?    return np\.stack\(f, 1\)\n", s).group(0)
new = '''FEATS = ("freq", "rec")


def _feat(D, u, e):
    f = []
    if "freq" in FEATS:
        f.append(np.log1p(D.freq[e]))
    if "rec" in FEATS and D.has_rec:
        r = D.rec[u, e]
        f.append(np.log1p(np.where(np.isinf(r), REC_CAP, r)))
    if not f:
        raise SystemExit("--feats leaves no feature (rec needs recency info)")
    return np.stack(f, 1)
'''
s = s.replace(old, new)
s = s.replace("    args = ap.parse_args()\n",
              '    ap.add_argument("--feats", default="freq,rec", help="FNM regressor features: freq, rec or freq,rec")\n'
              "    args = ap.parse_args()\n    global FEATS\n    FEATS = tuple(args.feats.split(\",\"))\n    print(\"[feats]\", FEATS)\n", 1)
open("cp_harness_feats.py", "w").write(s)
print("wrote cp_harness_feats.py")
