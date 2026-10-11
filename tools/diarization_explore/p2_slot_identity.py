"""Is Nemotron's arrival slot an identity signal independent of VBx? File-level slot -> track majority, per row."""
import pickle, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_features as pf
from collections import Counter
rows = pickle.loads((pf.CACHE / "rows.pkl").read_bytes())["rows"]
order = sorted(rows, key=lambda i: i.startswith("onetonton"))
res = {}
for ident in order:
    stable = pf.reference(ident)[0]
    data = np.load(pf.CACHE / f"{ident}.npz")
    frames = data["native"].shape[0]
    maps, purity, _, _ = pf.slot_maps(data, frames)
    prob = pf.resample(data["probs"], float(data["frame_seconds"]), frames)
    # file-level majority track per slot (frame-weighted by where the slot's per-frame map points)
    major = {}
    for j in range(maps.shape[0]):
        m = maps[j][(maps[j] >= 0) & (prob[:, j] > 0.5)]
        major[j] = Counter(m.tolist()).most_common(1)[0][0] if len(m) else -1
    for r in rows[ident]:
        row = stable[r["index"]]
        lo = int(float(row["start"]) / pf.STEP); hi = min(max(lo + 1, int(np.ceil(float(row["end"]) / pf.STEP))), frames)
        mass = prob[lo:hi].sum(axis=0)
        j = int(mass.argmax())
        res[(ident, r["index"])] = dict(slot=j, slot_track=major[j], slot_purity=float(purity[j]), mass=float(mass[j]))
    print(ident, flush=True)
pickle.dump(res, open(pf.CACHE / "slot_identity.pkl", "wb"))
