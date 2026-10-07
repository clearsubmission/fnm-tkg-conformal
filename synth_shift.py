#!/usr/bin/env python3
"""synth_shift.py - controlled synthetic check of the shift bound and of DA-FNM.

Queries arrive over T timestamps. The answer's frequency stratum k has a drifting mixture (the
never-seen share grows), and the label feature phi (log frequency) sets the score quantile q(phi).
After T/2 the never-seen stratum also drifts WITHIN itself: its scores shift up by d(t) (in units of
the noise s.d.), which violates the stable-within-stratum assumption of Prop. 2.

Because the score distributions are known, the within-stratum shift Delta_k(t) (Kolmogorov distance)
is exact, and we can check: static FNM coverage of stratum k at t >= 1 - alpha - Delta_k(t)
(shift bound), and DA-FNM's per-stratum long-run coverage under arbitrary drift (ACI bound).

  python synth_shift.py --out paper
"""
import argparse, json, os
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=240)
ap.add_argument("--n", type=int, default=400, help="queries per timestamp")
ap.add_argument("--alpha", type=float, default=0.1)
ap.add_argument("--gamma", type=float, default=0.05)
ap.add_argument("--dmax", type=float, default=0.8, help="max within-stratum shift (noise s.d. units)")
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--out", default="paper")
a = ap.parse_args()
alpha, T, n = a.alpha, a.T, a.n
T_cal = T // 4                                   # calibration period: first quarter (fixed)


def gen(rng):
    t = np.repeat(np.arange(T), n)
    pi0 = 0.03 + 0.17 * t / (T - 1)                     # never-seen share 3% -> 20%
    pi1 = 0.15 * np.ones_like(pi0)
    u = rng.random(len(t))
    k = np.where(u < pi0, 0, np.where(u < pi0 + pi1, 1, 2))
    phi = np.where(k == 0, 0.0, np.where(k == 1, rng.uniform(np.log(2), np.log(10), len(t)),
                                         rng.uniform(np.log(10), np.log(1000), len(t))))
    q = 0.9 - 0.12 * phi                                 # cold answers get worse (higher) scores
    sig = 0.08
    d = np.where((k == 0) & (t >= T // 2), a.dmax * (t - T // 2) / (T - 1 - T // 2), 0.0)   # within-stratum drift
    S = q + sig * (rng.standard_normal(len(t)) + d)
    return t, k, phi, S, d


def cq(x, al):
    x = np.sort(x); m = len(x); j = int(np.ceil((m + 1) * (1 - al)))
    return np.inf if (m == 0 or j > m) else x[j - 1]


def run(seed):
    rng = np.random.default_rng(seed)
    t, k, phi, S, d = gen(rng)
    cal = np.where(t < T_cal)[0]; ev = np.where(t >= T_cal)[0]
    # split CP and Mondrian-F
    tau_std = cq(S[cal], alpha)
    tau_m = np.array([cq(S[cal][k[cal] == g], alpha) for g in range(3)])
    # FNM: quantile regression on phi (D_A), per-stratum offsets (D_B)
    p = rng.permutation(cal); A, B = p[: len(p) // 2], p[len(p) // 2:]
    reg = HistGradientBoostingRegressor(loss="quantile", quantile=1 - alpha, max_iter=100, random_state=0)
    reg.fit(phi[A, None], S[A])
    r = S - reg.predict(phi[:, None])
    tau_f = np.array([cq(r[B][k[B] == g], alpha) for g in range(3)])
    hit = {"std": S[ev] <= tau_std, "mond_f": S[ev] <= tau_m[k[ev]], "fnm": r[ev] <= tau_f[k[ev]]}
    # DA-FNM: per-stratum ACI on the FNM residual pools, updated after every timestamp
    pools = [np.sort(r[B][k[B] == g]) for g in range(3)]
    a_t = np.full(3, alpha); h = np.zeros(len(ev), bool)
    te, ke, re_ = t[ev], k[ev], r[ev]
    for s in range(T_cal, T):
        mk = te == s
        tau = np.array([np.inf if a_t[g] <= 0 else (-np.inf if a_t[g] >= 1 else cq(pools[g], a_t[g])) for g in range(3)])
        hh = re_[mk] <= tau[ke[mk]]; h[mk] = hh
        for g in range(3):
            sel = ke[mk] == g
            if sel.any():
                a_t[g] += a.gamma * (alpha - (1 - hh[sel].mean()))
    hit["da_fnm"] = h
    # per-timestamp coverage of the never-seen stratum, and exact Kolmogorov shift Delta_0(t)
    out = {}
    for m, hm in hit.items():
        out[m] = {"cov0_t": [float(hm[(te == s) & (ke == 0)].mean()) for s in range(T_cal, T)],
                  "cov_t": [float(hm[te == s].mean()) for s in range(T_cal, T)],
                  "cov_by_g_late": [float(hm[(te >= T // 2) & (ke == g)].mean()) for g in range(3)],
                  "cov_late": float(hm[te >= T // 2].mean())}
    dt = np.array([a.dmax * max(s - T // 2, 0) / (T - 1 - T // 2) for s in range(T_cal, T)])
    out["delta0_t"] = (2 * norm.cdf(dt / 2) - 1).tolist()          # KS distance of a location shift dt
    return out


R = [run(s) for s in range(a.seeds)]
ts = np.arange(T_cal, T)
M = ["std", "mond_f", "fnm", "da_fnm"]
mean = {m: np.mean([r_[m]["cov0_t"] for r_ in R], 0) for m in M}
delta = np.array(R[0]["delta0_t"])
bound = 1 - alpha - delta
summary = {}
print(f"=== synthetic controlled shift: T={T}, n={n}/step, alpha={alpha}, within-stratum drift up to {a.dmax} s.d. after t={T//2}, {a.seeds} seeds")
print(f"{'method':<8} {'cov late':>8} {'cov0 late':>9} {'cov1 late':>9} {'cov2 late':>9} {'cov0 final 10%':>14}  {'below bound (5-step avg)':>24}")
for m in M:
    g_late = np.mean([r_[m]["cov_by_g_late"] for r_ in R], 0)
    fin = mean[m][-len(ts) // 10:].mean()
    sm5 = np.convolve(mean[m], np.ones(5) / 5, mode="valid")
    viol = float(np.mean(sm5 < bound[2:-2] - 0.01)) if m in ("mond_f", "fnm") else float("nan")
    summary[m] = {"cov_late": float(np.mean([r_[m]["cov_late"] for r_ in R])), "cov_by_g_late": g_late.tolist(),
                  "cov0_final": float(fin), "frac_t_below_bound": viol}
    print(f"{m:<8} {summary[m]['cov_late']:>8.3f} {g_late[0]:>9.3f} {g_late[1]:>9.3f} {g_late[2]:>9.3f} {fin:>14.3f}  {viol:>24.3f}")
# ACI long-run bound for DA-FNM per stratum: |avg err - alpha| <= (max(alpha,1-alpha)+gamma)/(gamma*N)
N = T - T_cal
print(f"ACI bound on timestamp-averaged miscoverage gap, per stratum: {(max(alpha, 1-alpha) + a.gamma) / (a.gamma * N):.3f}")
summary["aci_bound"] = (max(alpha, 1 - alpha) + a.gamma) / (a.gamma * N)
summary["final_delta0"] = float(delta[-1]); summary["final_bound"] = float(bound[-1])

os.makedirs(a.out, exist_ok=True)
json.dump({"args": vars(a), "summary": summary}, open(os.path.join(a.out, "synth_shift.json"), "w"), indent=1)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.3, 2.2))
ax.axhline(1 - alpha, color=MUTED, lw=1, ls="--", zorder=1)
ax.axvline(T // 2, color=GRID, lw=1, ls=":", zorder=0)
ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
sm = lambda x: np.convolve(x, np.ones(5) / 5, mode="same")
sty = {"std": ("Split CP", "#2a78d6"), "mond_f": ("Mondrian-F", "#eb6834"), "fnm": ("FNM", "#4a3aa7"), "da_fnm": ("DA-FNM", "#1baf7a")}
for m in M:
    ax.plot(ts[2:-2], sm(mean[m])[2:-2], color=sty[m][1], lw=1.6, label=sty[m][0], zorder=3)
ax.plot(ts, bound, color=INK, lw=1.0, ls="-.", label=r"$1-\alpha-\Delta_0(t)$", zorder=2)
ax.set_xlabel("timestamp"); ax.set_ylabel("never-seen coverage")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=6, loc="lower left", ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(a.out, "fig_synth_shift.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(a.out, "fig_synth_shift.png"), dpi=200, bbox_inches="tight")
print("wrote", os.path.join(a.out, "fig_synth_shift.pdf"))
