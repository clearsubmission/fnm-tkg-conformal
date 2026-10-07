#!/usr/bin/env python3
"""rolling_cp.py - forward-only (rolling) chronological evaluation.

Test timestamps are cut into B consecutive blocks. For every block b >= 2, all methods calibrate on the
queries of ALL earlier blocks only (expanding window) and are evaluated on block b. Nothing from the
evaluated block, or later, is used for calibration, normalisation or the FNM regressor.

Methods: split CP (std), Mondrian-F, Feature-norm. (RCP-style, fcp), FNM, plus two online baselines
from adaptive conformal inference (Gibbs & Candes 2021): ACI on the marginal threshold (aci) and
per-stratum ACI on Mondrian-F thresholds (aci_mond). ACI updates its level after every timestamp of the
evaluated block from that timestamp's realised miscoverage (labels revealed after each step).

  python rolling_cp.py --prefix scores/copy_icews18 --blocks 10 --out scores/copy_icews18_rolling.json
"""
import argparse, json, time
from types import SimpleNamespace
import numpy as np, torch
import cp_harness as H

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True)
ap.add_argument("--alpha", type=float, default=0.1)
ap.add_argument("--blocks", type=int, default=10)
ap.add_argument("--edges", default="1,10")
ap.add_argument("--rec-edges", default="2,8,31")
ap.add_argument("--gamma-aci", type=float, default=0.05)
ap.add_argument("--size-rows", type=int, default=1000, help="queries per block used for set size")
ap.add_argument("--min-bin", type=int, default=30, help="min queries for a fine bin to count in worst-bin")
ap.add_argument("--methods", default="std,aci,mond_f,aci_mond,fcp,fnm,da_fnm")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default=None)
a = ap.parse_args()
hargs = SimpleNamespace(min_n=10, min_class=10, k=10, gamma=0.3, w=0.5, bo_frac=0.2, chunk=1024)
alpha = a.alpha
D = H.Data(a.prefix, [float(x) for x in a.edges.split(",")], [float(x) for x in a.rec_edges.split(",")], a.device)
assert D.has_ts, "needs timestamps (add_time.py)"
S_true = D.nc["1mp"]
methods = a.methods.split(",")
fg_t = torch.as_tensor(D.fgrp, device=a.device)

order_t = np.unique(D.tsu)                                   # timestamp indices present in test
cuts = np.array_split(order_t, a.blocks)
block_of = np.empty(D.U, dtype=np.int64); block_of[:] = -1
for b, us in enumerate(cuts):
    block_of[us] = b
qblock = block_of[D.tsu]


def window_stats(hit, rows):
    ya = D.y[rows]; g = D.fgrp[ya]; fb = D.fine[ya]
    out = {"n": int(len(rows)), "cov": float(hit.mean()),
           "cov_g0": float(hit[g == 0].mean()) if (g == 0).any() else None, "n_g0": int((g == 0).sum()),
           "pi0": float((g == 0).mean())}
    fine = [hit[fb == k].mean() for k in range(len(H.FINE_EDGES) + 1) if (fb == k).sum() >= a.min_bin]
    out["worst_fine"] = float(min(fine)) if fine else None
    return out


def sizes_from_thr(rows, thr_rows):
    """thr_rows: tensor [n, E] thresholds on the 1-p score."""
    S = D.nc_rows(rows, "1mp")
    return (S <= thr_rows).float().sum(1).cpu().numpy()


ACI_STATE = {}          # ACI levels carry over from block to block (online)


def aci_run(cal, ev, per_stratum):
    """ACI: fixed calibration pool, level alpha_t updated after each timestamp of the block."""
    K = D.n_fg if per_stratum else 1
    gc = D.fgrp[D.y[cal]] if per_stratum else np.zeros(len(cal), int)
    pools = [np.sort(S_true[cal][gc == k]) for k in range(K)]
    a_t = ACI_STATE.setdefault("aci_mond" if per_stratum else "aci", np.full(K, alpha))
    hit = np.zeros(len(ev), bool); tau_rows = np.zeros((len(ev), K))
    ue = D.tsu[ev]; ge = D.fgrp[D.y[ev]] if per_stratum else np.zeros(len(ev), int)
    for u in np.unique(ue):
        mk = np.where(ue == u)[0]
        tau = np.empty(K)
        for k in range(K):
            n = len(pools[k])
            if n < 10 or a_t[k] <= 0: tau[k] = np.inf; continue
            if a_t[k] >= 1: tau[k] = -np.inf; continue
            j = int(np.ceil((n + 1) * (1 - a_t[k])))
            tau[k] = np.inf if j > n else pools[k][j - 1]
        h = S_true[ev[mk]] <= tau[ge[mk]]
        hit[mk] = h; tau_rows[mk] = tau
        for k in range(K):                                   # online update from realised errors at u
            sel = ge[mk] == k
            if sel.any():
                a_t[k] += a.gamma_aci * (alpha - (1.0 - h[sel].mean()))
    return hit, tau_rows


def da_fnm_run(cal, ev, b):
    """DA-FNM: FNM normalisation fitted on the past (D_A), per-stratum residual pools (D_B), and a
    per-stratum ACI level updated after every timestamp: tau_{k,t} = A_k(H_<t)."""
    reg, B, rB = H._fcp_fit(D, cal, alpha, np.random.default_rng(b))
    gB = D.fgrp[D.y[B]]
    pools = [np.sort(rB[gB == k]) for k in range(D.n_fg)]
    a_t = ACI_STATE.setdefault("da_fnm", np.full(D.n_fg, alpha))
    ue = D.tsu[ev]; ge = D.fgrp[D.y[ev]]
    r_ev = S_true[ev] - reg.predict(H._feat(D, ue, D.y[ev]))
    hit = np.zeros(len(ev), bool); tau_rows = np.zeros((len(ev), D.n_fg))
    for u in np.unique(ue):
        mk = np.where(ue == u)[0]
        tau = np.empty(D.n_fg)
        for k in range(D.n_fg):
            n = len(pools[k])
            if n < 10 or a_t[k] <= 0: tau[k] = np.inf; continue
            if a_t[k] >= 1: tau[k] = -np.inf; continue
            j = int(np.ceil((n + 1) * (1 - a_t[k])))
            tau[k] = np.inf if j > n else pools[k][j - 1]
        h = r_ev[mk] <= tau[ge[mk]]
        hit[mk] = h; tau_rows[mk] = tau
        for k in range(D.n_fg):
            sel = ge[mk] == k
            if sel.any():
                a_t[k] += a.gamma_aci * (alpha - (1.0 - h[sel].mean()))
    return hit, tau_rows, reg


def drift_ks(cal, ev, b):
    """Within-stratum drift of FNM residuals: KS distance between calibration (D_B) residuals and the
    evaluated block's residuals, per stratum (the quantity Delta_k in the shift bound)."""
    reg, B, rB = H._fcp_fit(D, cal, alpha, np.random.default_rng(b))
    gB = D.fgrp[D.y[B]]; ge = D.fgrp[D.y[ev]]
    r_ev = S_true[ev] - reg.predict(H._feat(D, D.tsu[ev], D.y[ev]))
    out = {}
    for k in range(D.n_fg):
        x, y_ = np.sort(rB[gB == k]), np.sort(r_ev[ge == k])
        if len(x) < 10 or len(y_) < 10: continue
        grid = np.concatenate([x, y_])
        out[f"g{k}"] = float(np.abs(np.searchsorted(x, grid, side="right") / len(x)
                                    - np.searchsorted(y_, grid, side="right") / len(y_)).max())
    return out


res = {"prefix": a.prefix, "alpha": alpha, "blocks": a.blocks, "methods": methods, "windows": {m: [] for m in methods}}
t0 = time.time()
rng = np.random.default_rng(0)
for b in range(1, a.blocks):
    cal = np.where(qblock < b)[0]; ev = np.where(qblock == b)[0]
    if len(ev) < 50:
        continue
    sz_idx = np.sort(rng.choice(len(ev), min(a.size_rows, len(ev)), replace=False))
    for m in methods:
        if m == "da_fnm":
            hit, tau_rows, reg = da_fnm_run(cal, ev, b)
            sizes = []
            for i in sz_idx:
                u = D.tsu[ev[i]]
                thr = reg.predict(H._feat(D, np.full(D.E, u), np.arange(D.E))) + tau_rows[i][D.fgrp]
                thr = torch.as_tensor(np.where(np.isinf(thr), np.sign(thr) * 1e9, thr), dtype=torch.float32, device=a.device)
                sizes.append(sizes_from_thr(ev[[i]], thr[None]))
            sizes = np.concatenate(sizes)
        elif m in ("aci", "aci_mond"):
            hit, tau_rows = aci_run(cal, ev, per_stratum=(m == "aci_mond"))
            tr = torch.as_tensor(np.where(np.isinf(tau_rows[sz_idx]), np.sign(tau_rows[sz_idx]) * 1e9, tau_rows[sz_idx]),
                                 dtype=torch.float32, device=a.device)
            thr = tr[:, fg_t] if m == "aci_mond" else tr[:, :1].expand(-1, D.E)
            sizes = sizes_from_thr(ev[sz_idx], thr)
        else:
            spec = H.METHODS[m](D, cal, alpha, np.random.default_rng(b), hargs)
            hit = S_true[ev] <= H.spec_true(spec, D, ev)
            cache = {}
            sizes = []
            for i in range(0, len(sz_idx), 256):
                rows = ev[sz_idx[i:i + 256]]
                sizes.append(sizes_from_thr(rows, H.spec_rows(spec, D, rows, cache)))
            sizes = np.concatenate(sizes)
        st = window_stats(hit, ev); st["size_frac"] = float(sizes.mean() / D.E); st["block"] = b
        ge_ = D.fgrp[D.y[ev]]
        st["cov_by_g"] = {f"g{k}": float(hit[ge_ == k].mean()) for k in range(D.n_fg) if (ge_ == k).sum() >= 10}
        if m == "fnm":
            st["ks"] = drift_ks(cal, ev, b)
        res["windows"][m].append(st)
    print(f"[block {b}/{a.blocks - 1}] n={len(ev)} pi0={res['windows'][methods[0]][-1]['pi0']:.3f} {time.time()-t0:.0f}s  "
          + "  ".join(f"{m}:{res['windows'][m][-1]['cov']:.3f}/{res['windows'][m][-1]['cov_g0'] if res['windows'][m][-1]['cov_g0'] is None else round(res['windows'][m][-1]['cov_g0'],3)}" for m in methods), flush=True)


def summ(ws, k):
    v = np.array([w[k] for w in ws if w[k] is not None], float)
    return (float(v.mean()), float(v.min()), float(v.max())) if len(v) else (None, None, None)


res["summary"] = {}
print(f"\n=== rolling forward-only, alpha={alpha}, {len(res['windows'][methods[0]])} windows "
      f"(pi0 range {summ(res['windows'][methods[0]], 'pi0')[1]:.3f}-{summ(res['windows'][methods[0]], 'pi0')[2]:.3f})")
print(f"{'method':<9} {'cov mean':>8} {'cov min':>8} {'cov0 mean':>9} {'cov0 min':>8} {'worst mean':>10} {'worst min':>9} {'size/E':>7}")
for m in methods:
    ws = res["windows"][m]
    for w_ in ws: w_["worst_g"] = min(w_["cov_by_g"].values()) if w_["cov_by_g"] else None
    c, z, w, s = summ(ws, "cov"), summ(ws, "cov_g0"), summ(ws, "worst_fine"), summ(ws, "size_frac")
    res["summary"][m] = {"cov": c, "cov_g0": z, "worst_fine": w, "size_frac": s, "worst_g": summ(ws, "worst_g")}
    f = lambda x: "  -  " if x is None else f"{x:.3f}"
    print(f"{m:<9} {f(c[0]):>8} {f(c[1]):>8} {f(z[0]):>9} {f(z[1]):>8} {f(w[0]):>10} {f(w[1]):>9} {f(s[0]):>7}")
ks = [(w_["block"], w_["ks"]) for w_ in res["windows"].get("fnm", []) if "ks" in w_]
if ks:
    print("FNM within-stratum drift (KS distance, calibration vs block), per stratum:")
    for b_, d_ in ks: print(f"  block {b_}: " + "  ".join(f"{k}={v:.3f}" for k, v in d_.items()))
# long-run per-stratum miscoverage over all evaluated queries (what ACI-type methods target)
print("pooled per-stratum coverage over all windows:")
for m in methods:
    tot = {}
    for w_ in res["windows"][m]:
        for k, v in w_["cov_by_g"].items(): tot.setdefault(k, []).append(v)
    print(f"  {m:<9} " + "  ".join(f"{k}={np.mean(v):.3f}" for k, v in sorted(tot.items())))
out = a.out or a.prefix + "_rolling.json"
json.dump(res, open(out, "w"), indent=1)
print("saved", out)
