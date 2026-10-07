"""add_time.py PREFIX NAME [DATA_DIR]
Attach per-query timestamps + entity recency to saved scores, after checking that the query
order matches the dataset's test file (plain, or with inverse queries appended)."""
import sys, os, numpy as np
from cp_dump import last_seen_matrix


def load(d, f):
    return np.loadtxt(os.path.join(d, f), dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)


def find_dir(name):
    for root in ["data", "../data", "../CENET/data", "../RE-Net/data", "../TiRGN/data"]:
        for cand in [name, name.upper(), name.lower()]:
            d = os.path.join(root, cand)
            if os.path.exists(os.path.join(d, "test.txt")):
                return d
    return None


prefix, name = sys.argv[1], sys.argv[2]
d = sys.argv[3] if len(sys.argv) > 3 else find_dir(name)
if d is None:
    sys.exit(f"[add_time] no data dir with test.txt for {name}; pass it as 3rd arg")
tr, va, te = load(d, "train.txt"), load(d, "valid.txt"), load(d, "test.txt")
m = dict(np.load(prefix + "_meta.npz"))
y = m["y"]
E = np.load(prefix + "_probs.npy", mmap_mode="r").shape[1]
if len(y) == len(te) and (y == te[:, 2]).all():
    ts, mode = te[:, 3], "plain"
elif len(y) == 2 * len(te) and (y == np.concatenate([te[:, 2], te[:, 0]])).all():
    ts, mode = np.concatenate([te[:, 3], te[:, 3]]), "with-inverse"
else:
    k = min(len(y), len(te))
    sys.exit(f"[add_time] order mismatch for {name} ({d}): |y|={len(y)} |test|={len(te)} "
             f"match={np.mean(y[:k] == te[:k, 2]):.3f}")
m["ts"] = ts
m["t_index"] = np.unique(ts)
m["last_seen"] = last_seen_matrix(np.concatenate([tr, va, te]), m["t_index"], E).astype(np.int32)
np.savez_compressed(prefix + "_meta.npz", **m)
print(f"[add_time] {name}: {mode}, {len(m['t_index'])} test timestamps, data={d}")
