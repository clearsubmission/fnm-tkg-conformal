#!/usr/bin/env python3
"""tie_check.py PREFIX [PREFIX ...]  -- precision / tie audit of stored (fp16) scores.
For each score file: fraction of exact-zero probabilities and of tied scores, rank-oracle cold
resolution under optimistic / mid / pessimistic tie-breaking, and Mondrian-F (split 0) with
non-strict (<=, as in the paper) vs strict (<) inclusion. If these agree, ties do not drive results."""
import sys, json, numpy as np, torch
EDGES, ALPHA, CH = [1, 10], 0.1, 1024
dev = "cuda" if torch.cuda.is_available() else "cpu"

def conf_q(s, a):
    n = len(s); k = int(np.ceil((n + 1) * (1 - a)))
    return np.inf if (n == 0 or k > n) else float(np.partition(s, k - 1)[k - 1])

out = {}
for pre in sys.argv[1:]:
    P = torch.from_numpy(np.load(pre + "_probs.npy")).to(dev)
    m = np.load(pre + "_meta.npz"); y = m["y"]; freq = m["ent_freq"]
    Q, E = P.shape; g = np.digitize(freq, EDGES); K = len(EDGES) + 1
    gt = torch.as_tensor(g, device=dev); yt_all = torch.as_tensor(y, device=dev)
    ptrue = P[torch.arange(Q, device=dev), yt_all].float()
    S_true = (1.0 - ptrue).cpu().numpy()
    rng = np.random.default_rng(0); perm = rng.permutation(Q); cal, ev = perm[:Q // 2], perm[Q // 2:]
    gy = g[y]
    tau = np.array([conf_q(S_true[cal][gy[cal] == k], ALPHA) if (gy[cal] == k).sum() >= 10 else np.inf for k in range(K)])
    tau_t = torch.as_tensor(np.where(np.isinf(tau), 1e9, tau), dtype=torch.float32, device=dev)[gt]
    zero = zero0 = S1 = 0; n0 = int((g == 0).sum())
    size_le = size_lt = in0_le = in0_lt = 0.0
    hit_le = (S_true[ev] <= tau[gy[ev]]); hit_lt = (S_true[ev] < tau[gy[ev]])
    r_opt, r_mid, r_pes, ties_g0 = [], [], [], []
    for i in range(0, len(ev), CH):
        rows = torch.as_tensor(ev[i:i + CH], device=dev)
        p = P[rows].float(); S = 1.0 - p
        zero += (p == 0).sum().item(); zero0 += (p[:, gt == 0] == 0).sum().item(); S1 += (S == 1.0).sum().item()
        le = S <= tau_t[None]; lt = S < tau_t[None]
        size_le += le.sum().item(); size_lt += lt.sum().item()
        in0_le += le[:, gt == 0].sum().item(); in0_lt += lt[:, gt == 0].sum().item()
        yt = yt_all[rows]; py = p[torch.arange(len(rows), device=dev), yt][:, None]
        same = gt[None, :] == gt[yt][:, None]
        gtc = ((p > py) & same).sum(1); eqc = ((p == py) & same).sum(1)       # eqc includes the truth
        r_opt.append((gtc + 1).cpu().numpy()); r_pes.append((gtc + eqc).cpu().numpy())
        r_mid.append((gtc + (eqc + 1) / 2.0).cpu().numpy()); ties_g0.append((eqc - 1).cpu().numpy())
    r_opt, r_mid, r_pes, ties = map(np.concatenate, (r_opt, r_mid, r_pes, ties_g0))
    ge = gy[ev]; nev = len(ev); G = [int((g == k).sum()) for k in range(K)]
    res = {"Q": Q, "E": E, "frac_p_zero": zero / (nev * E), "frac_p_zero_neverseen": zero0 / max(nev * n0, 1),
           "frac_S_eq_1": S1 / (nev * E), "tau": tau.tolist(),
           "mondF_cov_le": float(hit_le.mean()), "mondF_cov_lt": float(hit_lt.mean()),
           "mondF_cov0_le": float(hit_le[ge == 0].mean()), "mondF_cov0_lt": float(hit_lt[ge == 0].mean()),
           "mondF_sizefrac_le": size_le / nev / E, "mondF_sizefrac_lt": size_lt / nev / E,
           "mondF_in0_le": in0_le / nev / max(n0, 1), "mondF_in0_lt": in0_lt / nev / max(n0, 1)}
    for k in range(K):
        mk = ge == k
        if mk.sum() < 10: continue
        q = lambda r: float(np.quantile(r[mk], 1 - ALPHA, method="higher")) / max(G[k], 1)
        res[f"g{k}"] = {"coldres_opt": q(r_opt), "coldres_mid": q(r_mid), "coldres_pes": q(r_pes),
                        "frac_queries_true_tied": float((ties[mk] > 0).mean()),
                        "mean_tied_with_true": float(ties[mk].mean())}
    out[pre] = res
    print(f"\n=== {pre}  Q={Q} E={E}")
    print(f" p==0: {100*res['frac_p_zero']:.2f}% of all scores, {100*res['frac_p_zero_neverseen']:.2f}% of never-seen; S==1.0: {100*res['frac_S_eq_1']:.2f}%")
    print(f" Mondrian-F  <= : cov {res['mondF_cov_le']:.3f} cov0 {res['mondF_cov0_le']:.3f} size/E {res['mondF_sizefrac_le']:.3f} in-set0 {res['mondF_in0_le']:.3f}")
    print(f" Mondrian-F  <  : cov {res['mondF_cov_lt']:.3f} cov0 {res['mondF_cov0_lt']:.3f} size/E {res['mondF_sizefrac_lt']:.3f} in-set0 {res['mondF_in0_lt']:.3f}")
    for k in range(K):
        if f"g{k}" in res:
            r = res[f"g{k}"]
            print(f" g{k}: cold res opt/mid/pes {r['coldres_opt']:.3f}/{r['coldres_mid']:.3f}/{r['coldres_pes']:.3f}"
                  f"  true tied in {100*r['frac_queries_true_tied']:.1f}% of queries (mean {r['mean_tied_with_true']:.1f} ties)")
    del P; torch.cuda.empty_cache()
json.dump(out, open("tie_check.json", "w"), indent=1)
print("\nsaved tie_check.json")
