"""
cp_dump.py - save a scorer's test-set outputs once, so every conformal experiment
runs on saved scores (no GPU, no model reload).

Usage inside an existing script (e.g. cp_full.py), right before `run(name, probs, y, obj_count)`:

    from cp_dump import save_scores
    save_scores(f"scores/{name}", probs, y,
                ent_freq=[obj_count.get(e, 0) for e in range(probs.shape[1])],
                ts=test_timestamps,          # optional: per-query timestamp, same order as y
                all_facts=all_quads)         # optional: [N,4] (s, r, o, t) of train+valid+test

`ts` enables the chronological split and per-window drift analysis.
`ts` + `all_facts` also enable recency features (frequency x recency Mondrian, feature CP).
"""
import os
import numpy as np


def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def last_seen_matrix(all_facts, t_index, num_entities):
    """last_seen[u, e] = latest timestamp < t_index[u] at which entity e appeared
    as subject or object (-1 if never). Uses only strictly earlier facts."""
    f = np.asarray(all_facts)
    s, o, t = f[:, 0].astype(np.int64), f[:, 2].astype(np.int64), f[:, 3].astype(np.int64)
    order = np.argsort(t, kind="stable")
    s, o, t = s[order], o[order], t[order]
    last = np.full(num_entities, -1, dtype=np.int64)
    out = np.empty((len(t_index), num_entities), dtype=np.int64)
    ptr = 0
    for u, tu in enumerate(t_index):
        end = np.searchsorted(t, tu, side="left")      # facts with t < tu
        if end > ptr:
            np.maximum.at(last, s[ptr:end], t[ptr:end])
            np.maximum.at(last, o[ptr:end], t[ptr:end])
            ptr = end
        out[u] = last
    return out


def save_scores(prefix, probs, y, ent_freq, ts=None, all_facts=None):
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    p = _to_numpy(probs).astype(np.float16)
    np.save(prefix + "_probs.npy", p)
    meta = {"y": _to_numpy(y).astype(np.int64),
            "ent_freq": np.asarray(ent_freq, dtype=np.int64)}
    if ts is not None:
        ts = _to_numpy(ts).astype(np.int64)
        t_index = np.unique(ts)
        meta["ts"] = ts
        meta["t_index"] = t_index
        if all_facts is not None:
            meta["last_seen"] = last_seen_matrix(all_facts, t_index, p.shape[1])
    np.savez_compressed(prefix + "_meta.npz", **meta)
    print(f"[cp_dump] saved {prefix}: probs {p.shape} fp16, keys {list(meta)}")
