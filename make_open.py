"""make_open.py PREFIX  ->  PREFIX_open_{probs.npy,meta.npz}
Open-label reformulation: every entity never seen in training collapses into one label NEW,
with p(NEW | x) = sum of p(e | x) over never-seen e. The prediction set then reads
"the answer is one of these known entities, or a new one".  NEW has calibration examples
(every query whose answer is unseen), so no label lacks calibration data."""
import sys, numpy as np

pre = sys.argv[1]
P = np.load(pre + "_probs.npy", mmap_mode="r")
m = dict(np.load(pre + "_meta.npz"))
Q, E = P.shape
freq = m["ent_freq"]
new = freq == 0
seen = np.where(~new)[0]
S = len(seen)
out = np.lib.format.open_memmap(pre + "_open_probs.npy", mode="w+", dtype=np.float16, shape=(Q, S + 1))
for i in range(0, Q, 4096):
    blk = np.asarray(P[i:i + 4096], dtype=np.float32)
    out[i:i + 4096, :S] = blk[:, seen]
    out[i:i + 4096, S] = blk[:, new].sum(1)
out.flush()
pos = np.full(E, S, dtype=np.int64); pos[seen] = np.arange(S)
meta = {"y": pos[m["y"]], "ent_freq": np.append(freq[seen], 0)}
for k in ("ts", "t_index"):
    if k in m: meta[k] = m[k]
if "last_seen" in m:
    ls = m["last_seen"]
    meta["last_seen"] = np.concatenate([ls[:, seen], np.full((ls.shape[0], 1), -1, ls.dtype)], 1)
np.savez_compressed(pre + "_open_meta.npz", **meta)
print(f"[open] {pre}: {E} -> {S} known + NEW; share of queries answered by NEW = {(meta['y'] == S).mean():.3f}")
