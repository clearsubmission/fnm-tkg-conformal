"""make_struct.py PREFIX NAME [DATA_DIR] [--lam 2.0 --gamma 1.0]
Structure-aware cold redistribution (no fitting, fixed hyper-parameters).
For each query (s, r, ?, t), the scorer's total probability on never-seen entities E0 is kept,
but redistributed among E0 according to graph history strictly before t:
    z(e) = lam * 1[e has interacted with s before t] + log(1 + degree of e before t)
    p'(e|x) = M(x) * softmax_{E0}(gamma * z)(e),   M(x) = sum_{e in E0} p(e|x)
Probabilities of seen entities are unchanged. Writes PREFIX_struct_{probs.npy,meta.npz}."""
import sys, os, numpy as np

args = [a for a in sys.argv[1:]]
def opt(k, d):
    if k in args:
        i = args.index(k); v = float(args[i + 1]); del args[i:i + 2]; return v
    return d
lam, gamma = opt("--lam", 2.0), opt("--gamma", 1.0)
pre, name = args[0], args[1]


def load(d, f):
    return np.loadtxt(os.path.join(d, f), dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)


def find_dir(n):
    for root in ["data", "../data", "../CENET/data", "../RE-Net/data", "../TiRGN/data"]:
        for c in [n, n.upper(), n.lower()]:
            d = os.path.join(root, c)
            if os.path.exists(os.path.join(d, "test.txt")):
                return d


d = args[2] if len(args) > 2 else find_dir(name)
tr, va, te = load(d, "train.txt"), load(d, "valid.txt"), load(d, "test.txt")
m = dict(np.load(pre + "_meta.npz"))
y, freq, ts = m["y"], m["ent_freq"], m["ts"]
P = np.load(pre + "_probs.npy", mmap_mode="r")
Q, E = P.shape
if len(y) == len(te) and (y == te[:, 2]).all():
    subj = te[:, 0]
elif len(y) == 2 * len(te) and (y == np.concatenate([te[:, 2], te[:, 0]])).all():
    subj = np.concatenate([te[:, 0], te[:, 2]])
else:
    sys.exit("[struct] query order does not match test.txt")
E0 = np.where(freq == 0)[0]
if len(E0) == 0:
    sys.exit("[struct] no never-seen entities; nothing to do")
pos0 = np.full(E, -1); pos0[E0] = np.arange(len(E0))
facts = np.concatenate([tr, va, te]); facts = facts[np.argsort(facts[:, 3], kind="stable")]
deg = np.zeros(E, np.float64)
nbr = [set() for _ in range(E)]
out = np.lib.format.open_memmap(pre + "_struct_probs.npy", mode="w+", dtype=np.float16, shape=(Q, E))
order = np.argsort(ts, kind="stable")
ptr = 0
for u in np.unique(ts):
    end = np.searchsorted(facts[:, 3], u, side="left")
    for s_, _, o_, _ in facts[ptr:end]:
        deg[s_] += 1; deg[o_] += 1
        if o_ != s_:
            nbr[s_].add(o_); nbr[o_].add(s_)
    ptr = end
    idx = np.where(ts == u)[0]
    base = np.log1p(deg[E0])
    for i0 in range(0, len(idx), 2048):
        rows = idx[i0:i0 + 2048]
        p = np.asarray(P[rows], dtype=np.float32)
        M = p[:, E0].sum(1, keepdims=True)
        Z = np.broadcast_to(base, (len(rows), len(E0))).copy()
        for k, q in enumerate(rows):
            nb = [pos0[e] for e in nbr[subj[q]] if pos0[e] >= 0]
            if nb:
                Z[k, nb] += lam
        Z = gamma * Z; Z -= Z.max(1, keepdims=True)
        W = np.exp(Z); W /= W.sum(1, keepdims=True)
        p[:, E0] = M * W
        out[rows] = p.astype(np.float16)
out.flush()
np.savez_compressed(pre + "_struct_meta.npz", **m)
print(f"[struct] {pre}: |E0|={len(E0)} lam={lam} gamma={gamma} -> {pre}_struct")
