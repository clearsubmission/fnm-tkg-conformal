#!/usr/bin/env python3
"""
cp_harness.py - conformal prediction evaluation on saved TKG scores.

Inputs (written by cp_dump.save_scores):
  {prefix}_probs.npy  float16 [Q, E]   p(e | query)
  {prefix}_meta.npz   y [Q], ent_freq [E], optional ts [Q], t_index [U], last_seen [U, E]

All sets are label-conditional:  C(x) = { e : s(x, e) <= tau(e, x) },  covered iff s(x, y) <= tau(y, x).

Methods
  std        split CP, score 1 - p(e|x), one threshold
  pas        split CP with prevalence-adjusted softmax score  -p(e|x) / pi(e)
             (Ding, Fermanian & Salmon, ICLR 2026); pi = Laplace-smoothed training frequency
  interp     Interp-Q (same paper): tau(e) = (1-w) q_std + w q_class(e); classes with < min_class
             calibration points fall back to q_std
  clustered  clustered CP (Ding et al. 2023); rare / unseen labels -> null cluster
  mond_f     Mondrian over answer-frequency strata
  mond_fr    Mondrian over frequency x recency strata                    (needs ts + last_seen)
  mond_bo    FRB: frequency x recency Mondrian with back-off. A held-out 20% of the calibration
             set decides, per stratum, whether it has enough mass; sparse strata back off to their
             frequency stratum, then to the global threshold. Partition independent of the
             calibration points that set thresholds -> stratum-conditional validity holds.
  fcp        feature-normalised CP: quantile regression of s on [log freq, log recency], then
             split-conformalised (marginal validity only; set-size ablation)
  fnm        FNM (proposed): quantile regression on label features fitted on D_A, then per-stratum
             (Mondrian) conformal offsets on the residuals in D_B

Diagnostics
  - R random 50/50 splits (coverage on all, set sizes on the first --size-splits)
  - per candidate-group set size |C(x) ∩ E_k|  (Prop. 4: cost of validity)
  - rank oracle: smallest m s.t. "top-m within group k" covers 1-alpha of group-k answers
  - chronological split (earlier half of test timestamps calibrates)              (needs ts)
  - per-window drift + Prop. 1 prediction vs observed coverage                    (needs ts)
  - Prop. 2 check: coverage range under random reweighting of the group mixture
"""
import argparse, json, os, time
import numpy as np
import torch

FINE_EDGES = [1, 2, 5, 10, 25, 50, 100]
REC_CAP = 1e4


# ----------------------------------------------------------------------------- utils
def conf_q(scores, alpha):
    """Finite-sample split-conformal quantile; inf if not enough points."""
    scores = np.asarray(scores)
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.partition(scores, k - 1)[k - 1])


def agg(dicts):
    out = {}
    for k in dicts[0]:
        vals = [d.get(k) for d in dicts if k in d]
        if isinstance(vals[0], dict):
            out[k] = agg(vals)
        elif isinstance(vals[0], list):
            out[k] = vals[0]
        else:
            v = np.array([np.nan if x is None else x for x in vals], dtype=float)
            out[k] = {"mean": float(np.nanmean(v)) if np.isfinite(v).any() else None,
                      "std": float(np.nanstd(v)) if np.isfinite(v).any() else None}
    return out


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, prefix, edges, rec_edges, device):
        self.device = device
        probs = np.load(prefix + "_probs.npy", mmap_mode="r")
        m = np.load(prefix + "_meta.npz")
        self.Q, self.E = probs.shape
        self.y = m["y"]
        self.freq = m["ent_freq"]
        self.fgrp = np.digitize(self.freq, edges)
        self.n_fg = len(edges) + 1
        self.fine = np.digitize(self.freq, FINE_EDGES)
        self.group_sizes = [int((self.fgrp == k).sum()) for k in range(self.n_fg)]
        t0 = time.time()
        self.P = torch.from_numpy(np.array(probs)).to(device)
        print(f"[data] probs {probs.shape} on {device} in {time.time()-t0:.1f}s")
        yt = torch.as_tensor(self.y, device=device)
        p_true = self.P[torch.arange(self.Q, device=device), yt].float()
        self.nc = {"1mp": (1.0 - p_true).cpu().numpy()}
        prior = (self.freq + 1.0) / (self.freq.sum() + self.E)        # Laplace-smoothed prevalence
        self.prior_t = torch.as_tensor(prior, dtype=torch.float32, device=device)
        self.nc["pas"] = (-(p_true / self.prior_t[yt])).cpu().numpy()
        self.has_ts = "ts" in m.files
        self.has_rec = self.has_ts and "last_seen" in m.files
        if self.has_ts:
            self.ts = m["ts"]
            self.t_index = m["t_index"]
            self.tsu = np.searchsorted(self.t_index, self.ts)
            self.U = len(self.t_index)
        if self.has_rec:
            gran = np.min(np.diff(self.t_index)) if self.U > 1 else 1
            ls = m["last_seen"]
            rec = (self.t_index[:, None] - ls) / gran
            self.rec = np.where(ls < 0, np.inf, rec)
            self.rgrp = np.digitize(self.rec, rec_edges)
            self.n_rg = len(rec_edges) + 1

    def nc_rows(self, rows, score):
        p = self.P[torch.as_tensor(rows, device=self.device)].float()
        return 1.0 - p if score == "1mp" else -(p / self.prior_t[None, :])


# ----------------------------------------------------------------------------- threshold specs
# spec = (kind, value, score) with kind in {scalar, vector[E], matrix[U, E]}, score in {1mp, pas}
def spec_true(spec, D, idx):
    kind, v, _ = spec
    if kind == "lazy":
        reg, cvec = v
        u = D.tsu[idx] if D.has_ts else np.zeros(len(idx), int)
        return reg.predict(_feat(D, u, D.y[idx])) + cvec[D.y[idx]]
    if kind == "scalar":
        return np.full(len(idx), v)
    if kind == "vector":
        return v[D.y[idx]]
    return v[D.tsu[idx], D.y[idx]]


def spec_rows(spec, D, rows_np, cache):
    kind, v, _ = spec
    if kind == "scalar":
        return v
    if kind == "lazy":
        reg, cvec = v
        us = D.tsu[rows_np] if D.has_ts else np.zeros(len(rows_np), int)
        for u in np.unique(us):
            if u not in cache:
                row = reg.predict(_feat(D, np.full(D.E, u), np.arange(D.E))) + cvec
                cache[u] = torch.as_tensor(np.where(np.isinf(row), 1e9, row),
                                           dtype=torch.float32, device=D.device)
        return torch.stack([cache[u] for u in us])
    if "t" not in cache:
        cache["t"] = torch.as_tensor(np.where(np.isinf(v), 1e9, v), dtype=torch.float32, device=D.device)
    t = cache["t"]
    if kind == "vector":
        return t[None, :]
    return t[torch.as_tensor(D.tsu[rows_np], device=D.device)]


def m_std(D, cal, a, rng, args):
    return ("scalar", conf_q(D.nc["1mp"][cal], a), "1mp")


def m_pas(D, cal, a, rng, args):
    return ("scalar", conf_q(D.nc["pas"][cal], a), "pas")


def _classwise_q(D, cal, a, min_class):
    s, yc = D.nc["1mp"][cal], D.y[cal]
    o = np.argsort(yc, kind="stable"); ys, ss = yc[o], s[o]
    uniq, st = np.unique(ys, return_index=True)
    en = np.append(st[1:], len(ys))
    q = np.full(D.E, np.nan)
    for c, i, j in zip(uniq, st, en):
        if j - i >= min_class:
            q[c] = conf_q(ss[i:j], a)
    return q


def m_interp(D, cal, a, rng, args):
    q0 = conf_q(D.nc["1mp"][cal], a)
    qc = _classwise_q(D, cal, a, args.min_class)
    qc = np.where(np.isnan(qc) | np.isinf(qc), q0, qc)
    return ("vector", (1 - args.w) * q0 + args.w * qc, "1mp")


def m_mond_f(D, cal, a, rng, args):
    ag = D.fgrp[D.y[cal]]
    qg = np.array([conf_q(D.nc["1mp"][cal][ag == g], a) if (ag == g).sum() >= args.min_n else np.inf
                   for g in range(D.n_fg)])
    return ("vector", qg[D.fgrp], "1mp")


def m_mond_fr(D, cal, a, rng, args):
    G = D.fgrp[None, :] * D.n_rg + D.rgrp
    gc = G[D.tsu[cal], D.y[cal]]
    qg = np.array([conf_q(D.nc["1mp"][cal][gc == g], a) if (gc == g).sum() >= args.min_n else np.inf
                   for g in range(D.n_fg * D.n_rg)])
    return ("matrix", qg[G].astype(np.float32), "1mp")


def m_mond_bo(D, cal, a, rng, args):
    perm = rng.permutation(cal)
    nA = int(args.bo_frac * len(perm))
    A, B = perm[:nA], perm[nA:]
    m = max(args.min_n, int(np.ceil((1 - a) / a)))               # smallest n with finite quantile
    m_A = int(np.ceil(m * nA / max(len(B), 1)))                   # same mass, measured on A
    if D.has_rec:
        nfr = D.n_fg * D.n_rg
        G = D.fgrp[None, :] * D.n_rg + D.rgrp                     # [U, E]
        cA_fr = np.bincount(G[D.tsu[A], D.y[A]], minlength=nfr)
        cA_f = np.bincount(D.fgrp[D.y[A]], minlength=D.n_fg)
        key = np.empty(nfr, dtype=np.int64)                        # 0..nfr-1 own, nfr+f freq, -1 global
        for s in range(nfr):
            f = s // D.n_rg
            key[s] = s if cA_fr[s] >= m_A else (nfr + f if cA_f[f] >= m_A else -1)
        K = key[G]                                                 # [U, E]
        kB = K[D.tsu[B], D.y[B]]
    else:
        cA_f = np.bincount(D.fgrp[D.y[A]], minlength=D.n_fg)
        key = np.where(cA_f >= m_A, np.arange(D.n_fg), -1)
        K = key[D.fgrp]
        kB = K[D.y[B]]
    sB = D.nc["1mp"][B]
    keys = np.unique(K)
    qv = np.array([conf_q(sB[kB == k], a) for k in keys], dtype=np.float64)
    thr = qv[np.searchsorted(keys, K)].astype(np.float32)
    return ("matrix" if D.has_rec else "vector", thr, "1mp")


def _feat(D, u, e):
    f = [np.log1p(D.freq[e])]
    if D.has_rec:
        r = D.rec[u, e]
        f.append(np.log1p(np.where(np.isinf(r), REC_CAP, r)))
    return np.stack(f, 1)


def _fcp_fit(D, cal, a, rng):
    from sklearn.ensemble import HistGradientBoostingRegressor
    perm = rng.permutation(cal)
    A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
    uA = D.tsu[A] if D.has_ts else np.zeros(len(A), int)
    uB = D.tsu[B] if D.has_ts else np.zeros(len(B), int)
    reg = HistGradientBoostingRegressor(loss="quantile", quantile=1 - a, max_iter=200,
                                        max_leaf_nodes=15, min_samples_leaf=50, random_state=0)
    reg.fit(_feat(D, uA, D.y[A]), D.nc["1mp"][A])
    rB = D.nc["1mp"][B] - reg.predict(_feat(D, uB, D.y[B]))      # normalised calibration scores
    return reg, B, rB


def m_fcp(D, cal, a, rng, args):
    """Feature-normalised split CP: one offset -> marginal validity only."""
    reg, B, rB = _fcp_fit(D, cal, a, rng)
    return ("lazy", (reg, np.full(D.E, conf_q(rB, a))), "1mp")


def m_fnm(D, cal, a, rng, args):
    """FNM (proposed): feature-normalised Mondrian. Quantile regression on label features
    (log frequency, log recency) fitted on D_A pools information across labels, including those
    with no calibration examples; per-frequency-stratum offsets on D_B restore stratum-conditional
    validity (Prop. 2 applied to the normalised score, which is fixed given D_A)."""
    reg, B, rB = _fcp_fit(D, cal, a, rng)
    gB = D.fgrp[D.y[B]]
    c_glob = conf_q(rB, a)
    cg = np.array([conf_q(rB[gB == k], a) if (gB == k).sum() >= args.min_n else c_glob
                   for k in range(D.n_fg)])
    return ("lazy", (reg, cg[D.fgrp]), "1mp")


def m_clustered(D, cal, a, rng, args):
    from sklearn.cluster import KMeans
    perm = rng.permutation(cal)
    nA = int(args.gamma * len(perm))
    A, B = perm[:nA], perm[nA:]
    yA, sA = D.y[A], D.nc["1mp"][A]
    counts = np.bincount(yA, minlength=D.E)
    elig = np.where(counts >= args.min_class)[0]
    lab = np.full(D.E, -1)
    k = min(args.k, len(elig))
    if k >= 2:
        o = np.argsort(yA, kind="stable"); ys, ss = yA[o], sA[o]
        st, en = np.searchsorted(ys, elig), np.searchsorted(ys, elig, side="right")
        emb = np.stack([np.quantile(ss[i:j], [.5, .6, .7, .8, .9]) for i, j in zip(st, en)])
        lab[elig] = KMeans(k, n_init=10, random_state=0).fit(emb).labels_
    cB = lab[D.y[B]]
    q = {c: conf_q(D.nc["1mp"][B][cB == c], a) for c in np.unique(lab)}
    return ("vector", np.array([q[c] for c in lab], dtype=np.float64), "1mp")


METHODS = {"std": m_std, "pas": m_pas, "interp": m_interp, "clustered": m_clustered,
           "mond_f": m_mond_f, "mond_fr": m_mond_fr, "mond_bo": m_mond_bo, "fcp": m_fcp, "fnm": m_fnm}
NEEDS_REC = {"mond_fr"}


# ----------------------------------------------------------------------------- evaluation
def evaluate(spec, D, ev, sizes_on, chunk):
    hit = D.nc[spec[2]][ev] <= spec_true(spec, D, ev)
    sizes = by_cg = None
    if sizes_on:
        sizes = np.empty(len(ev), dtype=np.int64)
        by_cg = np.empty((len(ev), D.n_fg), dtype=np.int64)
        cg = torch.as_tensor(D.fgrp, device=D.device)
        onehot = torch.nn.functional.one_hot(cg, D.n_fg).float()          # [E, K]
        cache = {}
        for i in range(0, len(ev), chunk):
            rows = ev[i:i + chunk]
            inset = (D.nc_rows(rows, spec[2]) <= spec_rows(spec, D, rows, cache)).float()
            sizes[i:i + chunk] = inset.sum(1).cpu().numpy()
            by_cg[i:i + chunk] = (inset @ onehot).cpu().numpy()
    return hit, sizes, by_cg


def summarize(hit, sizes, by_cg, D, ev, args):
    ya = D.y[ev]
    g, fb = D.fgrp[ya], D.fine[ya]
    out = {"cov": float(hit.mean()), "n": int(len(ev))}
    gc = {}
    for k in range(D.n_fg):
        mk = g == k
        out[f"cov_g{k}"] = float(hit[mk].mean()) if mk.any() else None
        out[f"n_g{k}"] = int(mk.sum())
        if mk.sum() >= args.min_n:
            gc[k] = hit[mk].mean()
    out["worst_group_cov"] = float(min(gc.values())) if gc else None
    fine = []
    for k in range(len(FINE_EDGES) + 1):
        mk = fb == k
        out[f"cov_fine{k}"] = float(hit[mk].mean()) if mk.any() else None
        out[f"n_fine{k}"] = int(mk.sum())
        if mk.sum() >= args.min_n:
            fine.append(hit[mk].mean())
    out["worst_fine_cov"] = float(min(fine)) if fine else None
    if sizes is not None:
        out["size_mean"] = float(sizes.mean())
        out["size_median"] = float(np.median(sizes))
        out["size_frac_vocab"] = float(sizes.mean() / D.E)
        for k in range(D.n_fg):
            out[f"inset_frac_cg{k}"] = float(by_cg[:, k].mean() / max(D.group_sizes[k], 1))
    return out


def rank_oracle(D, ev, a, chunk):
    """Prop. 4 diagnostic: within-group rank of the true answer; the (1-a)-quantile is the
    smallest m such that 'top-m candidates of group k' covers 1-a of group-k answers."""
    ranks = np.empty(len(ev), dtype=np.int64)
    cg = torch.as_tensor(D.fgrp, device=D.device)
    for i in range(0, len(ev), chunk):
        rows = ev[i:i + chunk]
        p = D.P[torch.as_tensor(rows, device=D.device)].float()
        yt = torch.as_tensor(D.y[rows], device=D.device)
        py = p[torch.arange(len(rows), device=D.device), yt][:, None]
        same = cg[None, :] == cg[yt][:, None]
        ranks[i:i + chunk] = ((p > py) & same).sum(1).cpu().numpy() + 1
    g = D.fgrp[D.y[ev]]
    out = {}
    for k in range(D.n_fg):
        mk = g == k
        if mk.sum() >= 10:
            mstar = float(np.quantile(ranks[mk], 1 - a, method="higher"))
            out[f"g{k}"] = {"m_star": mstar, "group_size": D.group_sizes[k],
                            "m_star_frac": mstar / max(D.group_sizes[k], 1),
                            "uninformative_frac": 1 - a}
    return out


def mixture_shift(summ, D, draws=2000, seed=0):
    cov = np.array([np.nan if summ[f"cov_g{k}"] is None else summ[f"cov_g{k}"] for k in range(D.n_fg)])
    ok = ~np.isnan(cov)
    w = np.random.default_rng(seed).dirichlet(np.ones(ok.sum()), size=draws)
    m = w @ cov[ok]
    return {"min": float(m.min()), "p05": float(np.quantile(m, .05)),
            "p95": float(np.quantile(m, .95)), "max": float(m.max())}


def windows(hit, D, ev, cal, W, alpha):
    novel_ev = D.freq[D.y[ev]] == 0
    cov_seen = hit[~novel_ev].mean()
    cov_novel = hit[novel_ev].mean() if novel_ev.any() else np.nan
    pi_cal = float((D.freq[D.y[cal]] == 0).mean())
    u = D.tsu[ev]
    order = np.unique(u)
    rows = []
    for i in range(0, len(order), W):
        mk = np.isin(u, order[i:i + W])
        if mk.sum() < 30:
            continue
        pi = float(novel_ev[mk].mean())
        pred = (1 - pi) * cov_seen + pi * (0 if np.isnan(cov_novel) else cov_novel)
        rows.append({"t_start": int(D.t_index[order[i]]), "n": int(mk.sum()), "pi_novel": pi,
                     "cov_obs": float(hit[mk].mean()), "cov_pred": float(pred),
                     "gap_obs": float(hit[mk].mean() - (1 - alpha))})
    obs = np.array([r["cov_obs"] for r in rows]); pr = np.array([r["cov_pred"] for r in rows])
    ok = len(rows) > 2 and obs.std() > 0 and pr.std() > 0
    return {"pi_cal": pi_cal, "pi_test": float(novel_ev.mean()), "cov_seen": float(cov_seen),
            "cov_novel": float(cov_novel), "corr_obs_vs_pred": float(np.corrcoef(obs, pr)[0, 1]) if ok else None,
            "mae_obs_vs_pred": float(np.abs(obs - pr).mean()) if rows else None, "windows": rows}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.1])
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--size-splits", type=int, default=10)
    ap.add_argument("--methods", default="std,pas,interp,clustered,mond_f,mond_fr,mond_bo,fcp,fnm")
    ap.add_argument("--merge", action="store_true", help="add these methods into an existing results JSON")
    ap.add_argument("--edges", default="1,10", help="freq strata edges; match your _bidx")
    ap.add_argument("--rec-edges", default="2,8,31", help="recency buckets in time steps")
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--min-class", type=int, default=10)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--w", type=float, default=0.5, help="Interp-Q weight on classwise quantile")
    ap.add_argument("--bo-frac", type=float, default=0.2)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    D = Data(args.prefix, [float(x) for x in args.edges.split(",")],
             [float(x) for x in args.rec_edges.split(",")], args.device)
    methods = args.methods.split(",")
    if not D.has_rec:
        methods = [m for m in methods if m not in NEEDS_REC]
        print("[info] no recency info -> mond_fr skipped; mond_bo/fcp use frequency only")
    res = {"prefix": args.prefix, "Q": D.Q, "E": D.E, "group_sizes": D.group_sizes,
           "edges": args.edges, "methods": methods, "args": vars(args), "by_alpha": {}}

    for a in args.alpha:
        R = {m: [] for m in methods}
        t0 = time.time()
        for r in range(args.splits):
            rng = np.random.default_rng(r)
            idx = rng.permutation(D.Q)
            cal, ev = idx[: D.Q // 2], idx[D.Q // 2:]
            for m in methods:
                spec = METHODS[m](D, cal, a, np.random.default_rng(1000 + r), args)
                hit, sizes, by_cg = evaluate(spec, D, ev, r < args.size_splits, args.chunk)
                R[m].append(summarize(hit, sizes, by_cg, D, ev, args))
            if r == 0:
                oracle = rank_oracle(D, ev, a, args.chunk)
            if r % 10 == 0:
                print(f"[alpha={a}] split {r}/{args.splits} {time.time()-t0:.0f}s  "
                      + "  ".join(f"{m}:{R[m][-1]['cov']:.3f}/{R[m][-1]['worst_group_cov']}" for m in methods))
        block = {"random_splits": {m: agg(R[m]) for m in methods},
                 "mixture_shift_split0": {m: mixture_shift(R[m][0], D) for m in methods},
                 "rank_oracle_split0": oracle}

        if D.has_ts:
            med = np.median(D.t_index)
            cal = np.where(D.ts < med)[0]; ev = np.where(D.ts >= med)[0]
            chrono = {}
            for m in methods:
                spec = METHODS[m](D, cal, a, np.random.default_rng(0), args)
                hit, sizes, by_cg = evaluate(spec, D, ev, True, args.chunk)
                chrono[m] = summarize(hit, sizes, by_cg, D, ev, args)
                chrono[m]["drift"] = windows(hit, D, ev, cal, args.window, a)
            block["chronological"] = chrono
        res["by_alpha"][str(a)] = block

        print(f"\n=== alpha={a}  (mean over {args.splits} random splits; sizes over {args.size_splits})")
        print(f"{'method':<10} {'cov':>6} {'worstG':>6} {'worstF':>6} "
              + " ".join(f"{'g'+str(k):>6}" for k in range(D.n_fg)) + f" {'size':>9} {'size/E':>7}")
        for m in methods:
            s = block["random_splits"][m]
            f = lambda k: (f"{s[k]['mean']:.3f}" if k in s and s[k]['mean'] is not None else "  -  ")
            print(f"{m:<10} {f('cov'):>6} {f('worst_group_cov'):>6} {f('worst_fine_cov'):>6} "
                  + " ".join(f"{f('cov_g'+str(k)):>6}" for k in range(D.n_fg))
                  + f" {s['size_mean']['mean']:>9.1f} {s['size_frac_vocab']['mean']:>7.3f}")
        print("rank oracle (m*/|E_k|):", {k: round(v['m_star_frac'], 3) for k, v in oracle.items()})

    out = args.out or args.prefix + "_cp_results.json"
    if args.merge and os.path.exists(out):
        old = json.load(open(out))
        for a_, blk in res["by_alpha"].items():
            ob = old["by_alpha"].setdefault(a_, {})
            for sec, val in blk.items():
                if isinstance(val, dict) and sec != "rank_oracle_split0":
                    ob.setdefault(sec, {}).update(val)
                elif sec not in ob:
                    ob[sec] = val
        old["methods"] = list(dict.fromkeys(old.get("methods", []) + res["methods"]))
        res = old
    json.dump(res, open(out, "w"), indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
