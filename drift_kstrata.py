"""drift_kstrata.py PREFIX [PREFIX ...] [--alpha 0.1]
Prop. 1 check with K strata. Chronological split (first half of test timestamps calibrates),
split CP; per test timestamp, compare observed coverage with sum_k pi_k^t c_k for three
partitions: {novel, seen}, the three frequency strata, and the fine frequency bins.
Writes by_alpha[alpha].chronological.std.drift_k into PREFIX_cp_results.json."""
import sys, json, numpy as np

FINE = [1, 2, 5, 10, 25, 50, 100]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
alpha = float(sys.argv[sys.argv.index("--alpha") + 1]) if "--alpha" in sys.argv else 0.1
args = [a for a in args if a != str(alpha)]


def conf_q(s, a):
    n = len(s); k = int(np.ceil((n + 1) * (1 - a)))
    return np.inf if k > n else float(np.partition(s, k - 1)[k - 1])


for pre in args:
    m = np.load(pre + "_meta.npz")
    if "ts" not in m.files:
        print(f"[drift] {pre}: no timestamps, skipped"); continue
    P = np.load(pre + "_probs.npy", mmap_mode="r")
    y, freq, ts = m["y"], m["ent_freq"], m["ts"]
    nc = np.empty(len(y), np.float32)
    for i in range(0, len(y), 8192):
        blk = np.asarray(P[i:i + 8192], dtype=np.float32)
        nc[i:i + 8192] = 1 - blk[np.arange(len(blk)), y[i:i + 8192]]
    t_index = np.unique(ts); med = np.median(t_index)
    cal, ev = np.where(ts < med)[0], np.where(ts >= med)[0]
    hit = nc[ev] <= conf_q(nc[cal], alpha)
    fa = freq[y]
    parts = {"novel_seen": (fa > 0).astype(int), "strata3": np.digitize(fa, [1, 10]), "fine": np.digitize(fa, FINE)}
    tev = ts[ev]; out = {}
    for name, lab in parts.items():
        le = lab[ev]; K = lab.max() + 1
        ck = np.array([hit[le == k].mean() if (le == k).any() else 0.0 for k in range(K)])
        obs, pred, rows = [], [], []
        for t in np.unique(tev):
            mk = tev == t
            if mk.sum() < 30: continue
            pik = np.bincount(le[mk], minlength=K) / mk.sum()
            o, p = float(hit[mk].mean()), float(pik @ ck)
            obs.append(o); pred.append(p); rows.append({"t": int(t), "n": int(mk.sum()), "cov_obs": o, "cov_pred": p})
        obs, pred = np.array(obs), np.array(pred)
        ok = len(obs) > 2 and obs.std() > 0 and pred.std() > 0
        out[name] = {"n_windows": len(obs), "corr": float(np.corrcoef(obs, pred)[0, 1]) if ok else None,
                     "mae": float(np.abs(obs - pred).mean()) if len(obs) else None,
                     "obs_std": float(obs.std()) if len(obs) else None, "windows": rows}
    f = pre + "_cp_results.json"
    R = json.load(open(f))
    R["by_alpha"].setdefault(str(alpha), {}).setdefault("chronological", {}).setdefault("std", {})["drift_k"] = out
    json.dump(R, open(f, "w"), indent=1)
    print(f"[drift] {pre}: " + "  ".join(
        f"{k}: n={v['n_windows']} r={v['corr'] if v['corr'] is None else round(v['corr'], 2)} "
        f"mae={100*v['mae']:.2f}pt" for k, v in out.items()) + f"  (obs s.d. {100*out['fine']['obs_std']:.2f}pt)")
