"""What the rows a timeline leaves unlabelled are, per backend (owner 2026-10-04: 复核新后端的弃权行).

The edit cost charges an unlabelled row nothing (owner 2026-09-20), on a premise measured then on
Sortformer and DiariZen only: those rows are short interjections like 「はい」 that a proofreader
does not care about. Nemotron-3 + vbx leaves 11-16 such rows per file against DiariZen's ~1, and
its "cheaper than DiariZen" reading came entirely from them. This checks the premise on the new
default: are its abstained rows still short interjections, and where do they come from?

    python tools/diarization_explore/abstain_profile.py run        # production nemotron3 timelines, GPU
    python tools/diarization_explore/abstain_profile.py profile    # offline

`run` calls the production stage (`run_diarization`, default backend) on the same vocal tracks the
W2 / W2b rulers read, and caches each timeline under the main checkout's `tmp/diarization-abstain/`.
`profile` scores those and the cached DiariZen arm with the W2 helpers (same grid, same `assign_all`,
same `edit_cost`), and for every abstained row prints why it is empty (no activity at all vs some
activity under the 20 % share), its length and the ASR text.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nemotron3_probe as n3  # noqa: E402

sys.path.insert(0, str(n3.WT / "src"))

OUT = n3.MAIN / "tmp" / "diarization-abstain"
#: ⚠ hina first: `corpus_keys("onetonton")` runs `accuracy_cross.configure()`, which rebinds
#: `refined_multi.sources` to the onetonton files for the rest of the process.
CORPORA = ("hina", "onetonton")
WORDY_CHARS = 8


#: The arms `profile` compares: production, and Nemotron-3 with a lower entry threshold (findings 2.6:
#: 0.4 / 0.3 was nearly free on AliMeeting and VoxConverse). `enter`/`leave` reach every
#: `segments_from_activity` call that is not given parameters; only the first one, on Nemotron's own
#: probabilities in `rasterised`, sees anything but 0/1, so that is the one this moves.
VARIANTS = {"nemotron3+vbx": None, "nemotron3+vbx e0.4": (0.4, 0.3)}


def timeline_path(ident, variant="nemotron3+vbx"):
    thresholds = VARIANTS[variant]
    suffix = "" if thresholds is None else f"-e{thresholds[0]}-l{thresholds[1]}"
    return OUT / ident / f"nemotron3{suffix}-diar.json"


def _lower_entry(thresholds):
    from finesub.speech.diarization import segments

    original = segments.segments_from_activity

    def patched(activity, *, frame_seconds=segments.FRAME_SECONDS, parameters=None):
        if parameters is None:
            parameters = segments.SmoothingParameters(enter=thresholds[0], leave=thresholds[1])
        return original(activity, frame_seconds=frame_seconds, parameters=parameters)

    segments.segments_from_activity = patched


def run(variant="nemotron3+vbx", corpora=CORPORA):
    import nemotron3_cost as nc
    import vram_guard
    vram_guard.install()
    from finesub.speech.diarization.stage import run_diarization, write_timeline

    if VARIANTS[variant] is not None:
        _lower_entry(VARIANTS[variant])
    for corpus in corpora:
        for key in nc.corpus_keys(corpus):
            ident = key if corpus == "hina" else f"onetonton-{key:02d}"
            path = timeline_path(ident, variant)
            if path.exists():
                continue
            source = nc.source(ident)
            if not source.exists():
                print("skip", ident, "(no vocal track)", flush=True)
                continue
            result = run_diarization(source, device="cuda")
            path.parent.mkdir(parents=True, exist_ok=True)
            write_timeline(result, path)
            print(ident, result.backend, len({t.track for t in result.tracks}), "tracks", flush=True)


def _arms(corpus, key):
    """`nemotron3_cost.load_episode` without its exploration-path Nemotron arm, which is not on disk any more."""
    import backend_cost as bc
    import relabel_eval as ev

    ident = key if corpus == "hina" else f"onetonton-{key:02d}"
    if corpus == "hina":
        loaded = bc.hina_episode(key)
        if loaded is None:
            print("skip", ident, "(no cached DiariZen run)")
            return None
        (stable, _, grid, people, _), cached, starts = loaded
    else:
        data = ev.episode(key)
        cached = bc.V2_OUT / ident / "diarizen.npz"
        if data is None or not cached.exists():
            print("skip", ident, "(no cached DiariZen run)")
            return None
        stable, _, grid, people, _ = data
        starts = bc.refined_starts(key)
    if not timeline_path(ident).exists():
        print("skip", ident, "(no production nemotron3 timeline)")
        return None
    arms = {"diarizen": np.load(cached)["probs"][: len(grid)]}
    for variant in VARIANTS:
        if timeline_path(ident, variant).exists():
            tracks = json.loads(timeline_path(ident, variant).read_text(encoding="utf-8"))["tracks"]
            arms[variant] = bc.timeline_probs(tracks, len(grid))[: len(grid)]
    return ident, stable, grid, people, starts, arms


def _abstained(probs, stable, grid, people):
    """Rows the reference covers that this timeline labels with nothing, with why and how long."""
    import refined_multi as rm
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity

    tracks = segments_from_activity(probs.tolist(), frame_seconds=rm.ev.FRAME)
    active = probs[: len(grid)].max(axis=1) >= 0.5
    rows = []
    for index, (row, speaker) in enumerate(zip(stable, assign_all(stable, tracks))):
        labelled = {**row, "speaker": speaker.to_dict()}
        cover, frames = rm.ev.span_cover(labelled, grid)
        if not cover.any():
            continue
        a = int(row["start"] / rm.ev.FRAME)
        b = max(a + 1, int(np.ceil(row["end"] / rm.ev.FRAME)))
        rows.append(dict(index=index, empty=not rm.ev.tracks_of(labelled),
                         seconds=float(row["end"] - row["start"]),
                         activity=float(active[a:b].mean()) if b > a else 0.0,
                         people=int((cover > 0).sum()), who=people[int(cover.argmax())],
                         text=str(row.get("text", "")).strip()))
    return rows


def profile():
    import cost_metrics as cm
    import nemotron3_cost as nc

    records = []
    for corpus in CORPORA:
        for key in nc.corpus_keys(corpus):
            loaded = _arms(corpus, key)
            if loaded is None:
                continue
            ident, stable, grid, people, starts, arms = loaded
            per_arm = {}
            for arm, probs in arms.items():
                rows = _abstained(probs, stable, grid, people)
                cost = {flag: cm.edit_cost(probs, stable, grid, people, starts, price_unlabelled=flag)
                        for flag in (False, True)}
                per_arm[arm] = dict(rows=rows, edit=cost[False]["total"], priced=cost[True]["total"],
                                    abstained=cost[False]["abstained"],
                                    fixed=float(__import__("accuracy_lab").score(probs, stable, grid, people)["fixed"]),
                                    lead10=int(cm.leading_counts(probs, stable, grid)[10]))
            records.append(dict(corpus=corpus, ident=ident, arms=per_arm))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "profile.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    report(records)


def report(records):
    def summary(rows, label):
        if not rows:
            print(f"  {label:<34} n=0")
            return
        seconds = np.array([r["seconds"] for r in rows])
        chars = np.array([len(re.sub(r"[\W_]", "", r["text"])) for r in rows])
        silent = np.mean([r["activity"] == 0 for r in rows])
        multi = np.mean([r["people"] >= 2 for r in rows])
        # A rough "carries content" bar: 「はい」「えぇー」「本当に」 sit under it, a clause sits over it.
        wordy = np.mean(chars >= WORDY_CHARS)
        print(f"  {label:<34} n={len(rows):<5} sec median {np.median(seconds):.2f} P90 {np.percentile(seconds, 90):.2f}"
              f"  chars median {np.median(chars):.0f} P90 {np.percentile(chars, 90):.0f}  >={WORDY_CHARS} chars {wordy:.0%}"
              f"  no-activity {silent:.0%}  >=2 ref people {multi:.0%}")

    for corpus in CORPORA:
        group = [r for r in records if r["corpus"] == corpus]
        if not group:
            continue
        arms = [a for a in ("diarizen", *VARIANTS) if all(a in r["arms"] for r in group)]
        print(f"\n== {corpus}: {len(group)} files")
        for arm in arms:
            edit = np.mean([r["arms"][arm]["edit"] for r in group])
            priced = np.mean([r["arms"][arm]["priced"] for r in group])
            print(f"{arm}: edit {edit:.1f}  priced {priced:.1f}  abstained/file "
                  f"{np.mean([r['arms'][arm]['abstained'] for r in group]):.1f}")
        from accuracy_report import bootstrap_mean_ci
        for arm in arms[2:]:
            parts = []
            for key in ("priced", "fixed", "abstained", "lead10"):
                diffs = [r["arms"][arm].get(key, np.nan) - r["arms"]["nemotron3+vbx"].get(key, np.nan) for r in group]
                low, high = bootstrap_mean_ci(diffs)
                parts.append(f"{key} {np.mean(diffs):+.3f} [{low:+.3f}, {high:+.3f}]")
            print(f"  {arm} minus production, paired: " + "  ".join(parts))
        everything = [row for r in group for row in r["arms"]["diarizen"]["rows"]]
        summary(everything, "all reference-covered rows")
        for arm in arms:
            summary([row for r in group for row in r["arms"][arm]["rows"] if row["empty"]], f"{arm} abstained")
        only = []
        for r in group:
            dz = {row["index"] for row in r["arms"]["diarizen"]["rows"] if row["empty"]}
            only += [dict(row, ident=r["ident"]) for row in r["arms"]["nemotron3+vbx"]["rows"]
                     if row["empty"] and row["index"] not in dz]
        summary(only, "nemotron3+vbx only (DiariZen labels)")
        from collections import Counter
        by = Counter(row["who"] for row in only)
        print("    by reference person:", ", ".join(f"{w} {c}" for w, c in by.most_common(8)))
        rng = np.random.default_rng(20261004)
        for row in rng.permutation(only)[:25] if only else []:
            print(f"    {row['ident']:<14}{row['seconds']:5.2f}s act {row['activity']:.2f} ppl {row['people']}  {row['text'][:40]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "profile", "report"))
    parser.add_argument("--variant", default="nemotron3+vbx", choices=sorted(VARIANTS))
    parser.add_argument("--corpus", nargs="+", default=list(CORPORA), choices=CORPORA)
    args = parser.parse_args()
    if args.command == "run":
        run(args.variant, [c for c in CORPORA if c in args.corpus])
    elif args.command == "profile":
        profile()
    else:
        report(json.loads((OUT / "profile.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
