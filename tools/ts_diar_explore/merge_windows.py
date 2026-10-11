"""Cross-window speaker merging for the windowed arms (owner 2026-09-24: "少量代表性 sample 试合窗的工程方法").

Within a 5 min window MOSS separates speakers better than every reference (window-level B3 0.952 vs
Nemotron-3+vbx 0.934, DiariZen 0.919); stitched over an episode it falls to 0.60-0.92 because the windows'
labels do not join up. Every method here keeps each window's partition and only decides which window-local
labels ("units") are the same person:

  greedy       overlap co-speaking time, greedy one-to-one per adjacent pair (what `episode.py` does)
  hung         the same evidence, Hungarian assignment, a link needs >= LINK_MIN s of co-speaking
  emb@T        WeSpeaker embedding of each unit's clean speech, DiariZen's PLDA space, constrained
               average-linkage AHC at cosine distance T; units of one window never merge
  hung+emb@T   `hung` links as must-links (initial groups), then the same AHC over the groups
  anchor       each unit -> the reference timeline's track it co-speaks with longest (identity borrowed
               from Nemotron-3+vbx; the unit's partition is kept); `anchor-dz` borrows DiariZen's instead
  anchor-m     the same, but matched one-to-one per window (Hungarian), unmatched units joined by voice

Embedding machinery is `over8.Embedder` / `plda_space` / `constrained_merge` unchanged. ⚠ T is swept on the
same few episodes it is reported on -- read the sweep as "is there a regime where it helps", not a calibration.

    PYTHONPATH=<worktree>/src C:/Users/Carl/miniconda3/envs/asr/python.exe -m tools.ts_diar_explore.merge_windows \
        --arms moss mai2-diar g35t --items hina-72 hina-70 hina-64 hina-67 [--span A B]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

from . import arms as A
from . import linking
from . import score as S
from .common import TMP, WT, vocal_track, words_to_segments
from .episode import stitch

sys.path.insert(0, str(WT / "tools" / "diarization_explore"))
import over8  # noqa: E402

LINK_MIN = 2.0
THRESHOLDS = (0.3, 0.4, 0.5, 0.6)
MIN_CLEAN = 2.0


def units(parts) -> dict[str, list[tuple[float, float]]]:
    """'w<k>:<label>' -> its speech intervals (segments; word-timed arms merged by speaker and pause)."""
    out = defaultdict(list)
    for k, (_, _, w) in enumerate(parts):
        segs = w["segments"] if not w.get("words") else words_to_segments(w["words"])
        for s in segs:
            if s.get("speaker") is not None:
                out[f"w{k}:{s['speaker']}"].append((s["start"], s["end"]))
    return out


def adjacent_links(parts, *, hungarian: bool) -> list[tuple[str, str]]:
    """Links between window-local labels of adjacent windows, from co-speaking seconds in their overlap -- on
    the same units `episode.stitch` links (words, else segments), so `greedy` here is the stitched arm."""
    links = []
    for k in range(1, len(parts)):
        lo, hi = parts[k][0], parts[k - 1][1]
        prev, cur = ({f"w{j}:{n}": iv for n, iv in linking.by_label(S.tokens(parts[j][2]), lo, hi).items()}
                     for j in (k - 1, k))
        links += linking.link(prev, cur, hungarian=hungarian, min_link=LINK_MIN if hungarian else 0.0)
    return links


def groups_from_links(names, links) -> dict[str, int]:
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in links:
        parent[find(b)] = find(a)
    roots = sorted({find(n) for n in names})
    return {n: roots.index(find(n)) for n in names}


def embeddings(item: str, u, embedder) -> dict[str, tuple[np.ndarray, float]]:
    """unit -> (PLDA-space mean embedding, clean seconds); units with too little clean speech are absent."""
    import soundfile as sf
    wave, sr = sf.read(str(vocal_track(item)), dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    frames = int(len(wave) / sr * 100) + 1
    to_plda = over8.plda_space()
    by_window = defaultdict(list)
    for n in u:
        by_window[n.split(":")[0]].append(n)
    out = {}
    for names in by_window.values():
        masks = {}
        for n in names:
            m = np.zeros(frames, dtype=np.float32)
            for a, b in u[n]:
                m[int(a * 100):int(b * 100)] = 1
            masks[n] = m
        total = sum(masks.values())
        for n in names:
            clean = (masks[n] * (total == 1)).astype(np.float32)  # single-speaker frames only
            if clean.sum() * 0.01 < MIN_CLEAN:
                continue
            raw = embedder.raw(wave, clean)
            if raw is not None:
                out[n] = (to_plda(raw).mean(axis=0), clean.sum() * 0.01)
    return out


def ahc(names, emb, threshold, initial: dict[str, int] | None = None) -> dict[str, int]:
    """Constrained AHC over units (or over initial groups of units); unembeddable units stay alone."""
    init = initial or {n: i for i, n in enumerate(names)}
    groups = defaultdict(list)
    for n in names:
        groups[init[n]].append(n)
    keys = sorted(groups)
    usable = [g for g in keys if any(n in emb for n in groups[g])]
    if len(usable) < 2:
        return init
    cents, weights = [], []
    for g in usable:
        vs = [(emb[n][0], emb[n][1]) for n in groups[g] if n in emb]
        w = np.array([x[1] for x in vs])
        cents.append(np.average(np.stack([x[0] for x in vs]), axis=0, weights=w))
        weights.append(w.sum())
    windows = [{n.split(":")[0] for n in groups[g]} for g in usable]
    cannot = np.array([[bool(a & b) and i != j for j, b in enumerate(windows)] for i, a in enumerate(windows)])
    merged = over8.constrained_merge(np.stack(cents), np.array(weights), cannot, threshold)
    out, fresh = {}, max(merged) + 1
    for g in keys:
        gid = merged[usable.index(g)] if g in usable else None
        if gid is None:
            gid, fresh = fresh, fresh + 1
        for n in groups[g]:
            out[n] = int(gid)
    return out


def reference_tracks(item: str, reference: str) -> dict[int, list[tuple[float, float]]]:
    by = defaultdict(list)
    for s in A.load(item, reference)["segments"]:
        by[int(s["speaker"][1:])].append((s["start"], s["end"]))
    return by


def anchor(u, item: str, reference: str = "n3vbx") -> dict[str, int]:
    by = reference_tracks(item, reference)
    out, fresh = {}, 1000
    for n, iv in u.items():
        scores = {k: linking.co_seconds(iv, v) for k, v in by.items()}
        best = max(scores, key=scores.get) if scores else None
        if best is None or scores[best] <= 0:
            out[n], fresh = fresh, fresh + 1
        else:
            out[n] = int(best)
    return out


def anchor_matched(u, item: str, emb, reference: str = "n3vbx", threshold: float = 0.4) -> dict[str, int]:
    """Per window, units <-> reference tracks one-to-one (Hungarian on co-speaking time), so two people MOSS
    separates inside a window are never folded onto one reference track; a unit left without a track joins
    the nearest group by voice (constrained AHC), or stays alone."""
    from scipy.optimize import linear_sum_assignment
    by = reference_tracks(item, reference)
    keys = sorted(by)
    init, fresh = {}, 1000
    windows = defaultdict(list)
    for n in u:
        windows[n.split(":")[0]].append(n)
    for names in windows.values():
        m = np.array([[linking.co_seconds(u[n], by[k]) for k in keys] for n in names])
        rows, cols = linear_sum_assignment(-m)
        matched = {names[r]: keys[c] for r, c in zip(rows, cols) if m[r, c] > 0}
        for n in names:
            if n in matched:
                init[n] = int(matched[n])
            else:
                init[n], fresh = fresh, fresh + 1
    return ahc(sorted(u), emb, threshold, initial=init)


def relabel(parts, mapping: dict[str, int]) -> dict:
    merged = stitch(parts, do_link=False)
    key = "words" if merged["words"] else "segments"
    for x in merged[key]:
        if x["speaker"] is not None:
            x["speaker"] = f"G{mapping.get(x['speaker'], x['speaker'])}"
    if merged["words"]:
        merged["segments"] = words_to_segments(merged["words"])
    return merged


def probes(parts, u, ref) -> dict[str, float]:
    """Reference-guided probes, not methods (marked † in the table): B-cubed of the windows' partition when
    identity is decided with the answer in hand. Any grouping of window-local labels is allowed, two labels of
    one window included (as `anchor` also allows). Every candidate is scored the way `S.speaker` scores
    `relabel(parts, grouping)` -- a group's overlap with a line is summed before the line's winner is chosen --
    and the best grouping found is re-scored through `relabel` + `S.speaker`, so each value is achievable.

      majority†  each unit -> the person it overlaps longest
      search†    steepest ascent over groupings (move one unit to another group or a new one), maximising the
                 B-cubed itself, from `majority` and from all-separate: what some grouping reaches, not the
                 most any grouping can reach
    """
    names = sorted(u)
    lines, overlaps = S.line_overlaps(ref, relabel(parts, {}), 0.0, 1e9)
    col = {f"G{n}": j for j, n in enumerate(names)}
    m_ov = np.zeros((len(lines), len(names)))  # seconds each unit overlaps each line
    for i, ov in enumerate(overlaps):
        for lab, sec in ov.items():
            m_ov[i, col[lab]] = sec
    person = np.unique([line["speaker"] for line in lines], return_inverse=True)[1]
    heard = m_ov.sum(1) > 0

    def score(assign: np.ndarray) -> float:
        """B-cubed of a grouping (unit j -> group assign[j]): group seconds summed per line, then the winner."""
        by_group = m_ov @ np.eye(len(names))[assign].reshape(len(names), -1)
        label = np.where(heard, by_group.argmax(1), len(names) + np.arange(len(lines)))  # unheard: singletons
        pair = np.unique(person * (2 * len(names) + len(lines)) + label, return_inverse=True)[1]
        both, lc, pc = np.bincount(pair)[pair], np.bincount(label)[label], np.bincount(person)[person]
        prec, rec = (both / lc).mean(), (both / pc).mean()
        return 2 * prec * rec / (prec + rec)

    people = {p: k for k, p in enumerate(sorted({r["speaker"] for r in ref if r["speaker"]}))}
    maj = np.arange(len(names))
    for j, n in enumerate(names):
        c = defaultdict(float)
        for r in ref:
            if r["speaker"]:
                c[r["speaker"]] += linking.co_seconds(u[n], [(r["start"], r["end"])])
        if c and max(c.values()) > 0:
            maj[j] = len(names) + people[max(c, key=c.get)]  # group ids past the units' own
    maj = np.unique(maj, return_inverse=True)[1]  # compact ids for the one-hot
    best, best_s = maj, score(maj)
    for start in (maj, np.arange(len(names))):
        m, cur = start.copy(), score(start)
        while True:
            top, top_s = None, cur
            for j in range(len(names)):
                for g in set(m.tolist()) | {len(names)}:  # an existing group, or a group of its own
                    if g == m[j]:
                        continue
                    trial = m.copy()
                    trial[j] = g
                    trial = np.unique(trial, return_inverse=True)[1]
                    s = score(trial)
                    if s > top_s + 1e-12:
                        top, top_s = trial, s
            if top is None:
                break
            m, cur = top, top_s
        if cur > best_s:
            best, best_s = m, cur
    check = lambda a: S.speaker(ref, relabel(parts, {n: int(g) for n, g in zip(names, a)}), 0.0, 1e9)["b3_f1"]  # noqa: E731
    return {"majority†": check(maj), "search†": check(best)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["moss", "mai2-diar", "g35t"],
                    choices=sorted(n for n, x in A.ARMS.items() if x.windowed))
    ap.add_argument("--items", nargs="+", default=["hina-72", "hina-70", "hina-64", "hina-67"])
    ap.add_argument("--span", nargs=2, type=float, metavar=("A", "B"),
                    help="the `episode.py --span` run of each arm; only reference lines in [A, B) are scored")
    ap.add_argument("--probes", action="store_true", help="also the reference-guided probes (majority†, search†)")
    a = ap.parse_args()
    span = tuple(a.span) if a.span else None
    embedder = over8.Embedder()
    rows = defaultdict(dict)
    try:
        for item in a.items:
            ref = S.reference(item)
            for arm in a.arms:
                parts = A.windows(item, arm, span=span)
                if not parts:
                    continue
                u = units(parts)
                names = sorted(u)
                emb = embeddings(item, u, embedder)
                hung = groups_from_links(names, adjacent_links(parts, hungarian=True))
                methods = {"greedy": groups_from_links(names, adjacent_links(parts, hungarian=False)),
                           "hung": hung, "anchor": anchor(u, item), "anchor-dz": anchor(u, item, "diarizen"),
                           "anchor-m": anchor_matched(u, item, emb), "anchor-m-dz": anchor_matched(u, item, emb, "diarizen")}
                for t in THRESHOLDS:
                    methods[f"emb@{t}"] = ahc(names, emb, t)
                    methods[f"hung+emb@{t}"] = ahc(names, emb, t, initial=hung)
                for name, mapping in methods.items():
                    s = S.speaker(ref, relabel(parts, mapping), *(span or (0.0, 1e9)))
                    rows[(arm, name)][item] = (s["owner"], s["b3_f1"], s["labels"], s["persons"])
                if a.probes:
                    lo, hi = span or (0.0, 1e9)
                    for name, v in probes(parts, u, [r for r in ref if lo <= (r["start"] + r["end"]) / 2 < hi]).items():
                        rows[(arm, name)][item] = (float("nan"), v, 0, 0)
                print(f"{item} {arm}: {len(names)} units, {len(emb)} embedded", flush=True)
    finally:
        embedder.close()
    items = a.items
    print(f"\n{'arm':<16} {'method':<14}" + "".join(f"{i:>18}" for i in items) + f"{'mean B3':>9}")
    for (arm, name), per in rows.items():
        cells = "".join((f"{per[i][0] * 100:5.1f}/{per[i][1]:.3f}({per[i][2]}/{per[i][3]})" if i in per else "—").rjust(18)
                        for i in items)
        mean = np.mean([per[i][1] for i in per])
        print(f"{arm:<16} {name:<14}{cells}{mean:9.3f}")
    (TMP / f"merge_windows-{'-'.join(a.items)}.json").write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in rows.items()}, indent=1),
                                            encoding="utf-8")


if __name__ == "__main__":
    main()
