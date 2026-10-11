"""Plan 5o.4: scorer v2 -- content that does not depend on how the output is split into lines.

The 5h.14-5h.20 scorer aligned every output line on its own and only accepts blocks of >=2
characters, so fragmenting a line read as "missed" and "made up" (plan 5o.1 M1). Here, per
window:

1. all output lines, in the order given, are joined into one string and aligned once
   (`s4_breakdown.align` itself unchanged) -- each reference character's status comes from it;
2. what the first pass left over is aligned again against the same references: matches
   there are *repeats*, not "nothing near";
3. a reference character's attribution is the best over every copy of it (first-pass
   source + repeats) -- strict > multi incl. own > unlabelled > wrong; a speaker with no
   matched track is counted apart (M15);
4. every output character gets exactly one kind: the best copy of some reference character
   (own / multi / someone else's / unlabelled when right, near-miss when wrong), another
   copy (repeat), or nothing near.

Reads caches only (`DIAR_EXPLORE_DATA` / `DIAR_EXPLORE_WORK`, `DIAR_S4_CORPUS=ami` for AMI).

  --score          both groups, every arm, the 5o.4 verdicts
  --boot N         bootstrap resamples (default 2000)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

import numpy as np

import s4_breakdown as b
import s4_headroom as ref
import s4_live as live
import s4_llm as llm
import s4_projection as proj
import s4_resegment as rsg
import s4_timecut as tc

CORRECT = ("right", "homophone")
HEARD = ("right", "homophone", "misheard")

#: (arm, session, run) -> chunks whose provider call never came back, so that arm scores a
#: zero there that reads exactly like "it wrote nothing". Printed with the verdicts.
SILENT_GAPS: dict[tuple[str, str, int], int] = {}
ATTR_RANK = {"strict": 4, "multi": 3, "unlabelled": 2, "no track": 2, "wrong": 1}


def attribution(own, tracks):
    if own is None:
        return "no track"
    if not tracks:
        return "unlabelled"
    if own in tracks:
        return "strict" if len(tracks) == 1 else "multi"
    return "wrong"


# ---- one window -----------------------------------------------------------------------------

def align_with_repeats(h, refs, line_of=None):
    """`s4_breakdown.align` on a joined string: one more pass, and gaps kept to their line.

    Pass 1 claims matching blocks exactly as `align` does -- a block may run across a line
    break, which is what lets fragments join up. Pass 2 looks for blocks again among what
    pass 1 left -- a second copy of text already claimed is a *repeat*. Only then are the
    gaps paired ("misheard"), never with a repeat, and only with characters on the line of
    the block the gap hangs off (`line_of[j]`, a line id per character): unrestricted, the gap after one person's
    last block runs into the next person's line and turns their words into a "misheard"
    copy of this sentence on the wrong line. Both found by the contract tests and a trial
    run (plan 5o.4); per line this is exactly `align`'s gap pairing.

    Returns rid -> per position the copies [(status, output index)]: the first-pass one
    (block or gap pair), then any repeats.
    """
    same_line = (lambda a, b_: True) if line_of is None else (lambda a, b_: line_of[a] == line_of[b_])  # noqa: E731
    order = sorted(refs, key=lambda row: -sum(n for _, _, n in b.blocks(row["norm"], h, set())))
    claimed: dict[int, str] = {}
    first, second = {}, {}
    for row in order:
        first[row["id"]] = found = b.blocks(row["norm"], h, set(claimed))
        for _, j, n in found:
            claimed.update((j + t, row["id"]) for t in range(n))
    taken = dict(claimed)
    for row in order:
        second[row["id"]] = found = b.blocks(row["norm"], h, set(taken))
        for _, j, n in found:
            taken.update((j + t, row["id"]) for t in range(n))
    free = lambda j: 0 <= j < len(h) and j not in taken  # noqa: E731
    out = {}
    for row in order:
        r = row["norm"]
        status, source = ["miss"] * len(r), [None] * len(r)
        found = first[row["id"]]
        for i, j, n in found:
            for t in range(n):
                status[i + t] = "right" if r[i + t] == h[j + t] else "homophone"
                source[i + t] = j + t

        def pair(r_gap, h_gap):
            for ri, hj in zip(r_gap, h_gap):
                same = r[ri] == h[hj]
                sound = not same and b.lazy_pinyin is not None and b.keys(r[ri]) == b.keys(h[hj])
                status[ri] = "right" if same else "homophone" if sound else "misheard"
                source[ri] = hj

        if found:  # the gap pairing of `align`, but for `free` and the line of the block
            first_i, first_j, _ = found[0]
            run, j = [], first_j - 1
            while free(j) and same_line(j, first_j) and len(run) < first_i:
                run.append(j)
                j -= 1
            pair(list(range(first_i - 1, -1, -1)), run)
            for (i1, j1, n1), (i2, j2, _) in zip(found, found[1:]):
                pair(list(range(i1 + n1, i2)), [j for j in range(j1 + n1, j2)
                                                  if free(j) and (same_line(j, j1 + n1 - 1) or same_line(j, j2))])
            last_i, last_j, last_n = found[-1]
            run, j = [], last_j + last_n
            while free(j) and same_line(j, last_j + last_n - 1) and len(run) < len(r) - last_i - last_n:
                run.append(j)
                j += 1
            pair(list(range(last_i + last_n, len(r))), run)
        copies = [[(value, j)] if j is not None else [] for value, j in zip(status, source)]
        for i, j, n in second[row["id"]]:
            for t in range(n):
                copies[i + t].append(("right" if r[i + t] == h[j + t] else "homophone", j + t))
        out[row["id"]] = copies
    return out


def window(lines, refs, owner_of):
    """Score one window's output.

    lines     [(tracks, text)] in the order given
    refs      nearby reference rows (`id`, `norm`, `spk`)
    owner_of  reference id -> that speaker's track, or None

    Returns (content, copies, kinds, line_of): content / copies map (rid, pos) to
    [(status, attribution, line index, output index)] -- `content` from the one joined
    alignment only, `copies` adding the per-label alignments. Statuses, and which output
    characters are spoken for at all (kinds -> repeat / nothing near / decoy), come from
    `content`; only the attribution label comes from `copies`. kinds[j] is output character
    j's kind and line_of[j] its line.
    """
    h = "".join(text for _, text in lines)
    line_of = [index for index, (_, text) in enumerate(lines) for _ in text]
    starts = [sum(len(text) for _, text in lines[:index]) for index in range(len(lines))]

    # Gaps stay within a run of consecutive lines carrying the same labels, so cutting a
    # line into pieces (same labels, same order) changes nothing at all.
    run_of_line, run = [], -1
    for index, (tracks, _) in enumerate(lines):
        run += index == 0 or tracks != lines[index - 1][0]
        run_of_line.append(run)

    def collect(text, index_map, into):
        runs_here = [run_of_line[line_of[index_map[local]]] for local in range(len(text))]
        for rid, positions in align_with_repeats(text, refs, runs_here).items():
            for pos, found in enumerate(positions):
                for value, local in found:
                    if value in HEARD:
                        j = index_map[local]
                        into[(rid, pos)].append((value, attribution(owner_of(rid), lines[line_of[j]][0]), line_of[j], j))

    # Content: one alignment of everything, so the line grid cannot matter.
    content: dict[tuple[str, int], list[tuple[str, str, int, int]]] = defaultdict(list)
    collect(h, range(len(h)), content)
    # Attribution also asks each label's own lines, joined: a copy on the speaker's own line
    # that the joined string only gap-paired, or whose block another line's copy took
    # first, is still a copy (found in the trial run, plan 5o.4).
    copies = defaultdict(list, {key: list(value) for key, value in content.items()})
    groups = defaultdict(list)
    for index, (tracks, _) in enumerate(lines):
        groups[tracks].append(index)
    if len(groups) > 1:
        for members in groups.values():
            text = "".join(lines[index][1] for index in members)
            index_map = [starts[index] + t for index in members for t in range(len(lines[index][1]))]
            collect(text, index_map, copies)
    # The one copy that stands for each reference character. Which output characters are
    # spoken for at all -- and so which are repeats and which correspond to nothing -- comes
    # from the joined alignment ALONE, because that is the half that has to be independent of
    # the line grid (5o.4: "内容与诱饵仍只取全体拼接的那一次"; 6 纪律 5). Only the attribution
    # *label* of a character that the joined alignment already matched may come from `copies`.
    # Reading `sources`/`primary` off `copies` made the decoy rate a function of how many
    # distinct label groups the window happened to hold (audit 5q, B).
    best_attr = {key: max(options, key=lambda o: ATTR_RANK[o[1]])[1] for key, options in copies.items()}
    primary: dict[int, tuple[str, str, str]] = {}
    for (rid, pos), options in content.items():
        value, attr, _, j = max(options, key=lambda o: (ATTR_RANK[o[1]], b.RANK[o[0]]))
        primary.setdefault(j, (rid, value, best_attr.get((rid, pos), attr)))
    sources = {j for options in content.values() for _, _, _, j in options}
    kinds = []
    for j in range(len(h)):
        if j in primary:
            _, value, attr = primary[j]
            kinds.append(("near-miss" if value == "misheard" else attr, primary[j][0]))
        elif j in sources:
            kinds.append(("repeat", None))
        else:
            kinds.append(("nothing near", None))
    return content, copies, kinds, line_of


# ---- one arm --------------------------------------------------------------------------------

class Loaded:
    """Everything an arm is scored against, per session: rows, windows, chunks."""

    def __init__(self, name):
        self.session = session = b.Session(name)
        self.name = name
        self.by_id = {row["id"]: row for row in session.refs}
        self.chunk_of = {}
        for index, chunk in enumerate(llm.chunks(name)):
            for span in chunk["spans"]:
                self.chunk_of[(span["start"], span["end"])] = f"{name}:{index}"
        self.kind_of = {(s["start"], s["end"]): s["kind"] for s in session.spans}
        self.pairs = [row for _, row in b.pair_members(session)]
        real = [s for s in session.spans if s["kind"] == "real"]
        self.touched = [row for row in session.refs if any(s["start"] < row["end"] + ref.PAD and s["end"] > row["start"] - ref.PAD for s in real)]
        self.track_speakers = defaultdict(set)
        for speaker, track in session.speaker_track.items():
            self.track_speakers[track].add(speaker)

    def owner_of(self, rid):
        return self.session.speaker_track.get(self.by_id[rid]["spk"])


def score_run(loaded, lines_for, pad):
    """Per-chunk counters for one run of one arm. `lines_for(loaded) -> {(s, e): [(tracks, text)]}`."""
    chunks = defaultdict(Counter)
    best_status = {}  # (session, rid, pos) -> best status over windows, for retention
    for item in loaded:
        session = item.session
        lines_by_window = lines_for(item)
        scored = {}
        for key, lines in lines_by_window.items():
            lines = [(tuple(tracks), text) for tracks, text in lines if text]
            if not lines:
                continue
            near = session.near(key[0], key[1], b.NEAR)
            content, copies, kinds, line_of = window(lines, near, item.owner_of)
            scored[key] = (content, copies)
            chunk = chunks[item.chunk_of[key]]
            prefix = "decoy " if item.kind_of[key] == "decoy" else ""
            for kind, _ in kinds:
                chunk[f"{prefix}written"] += 1
                chunk[f"{prefix}hyp {kind}"] += 1
            if not prefix:
                # labels a person never got a character of: one deletion each
                owners = defaultdict(set)
                for j, (kind, rid) in enumerate(kinds):
                    if rid is not None:
                        owners[line_of[j]].add(item.by_id[rid]["spk"])
                for index, (tracks, _) in enumerate(lines):
                    if len(tracks) > 1:
                        chunk["edit drop label"] += sum(1 for k in tracks if not (item.track_speakers[k] & owners[index]))
        for key, kind in item.kind_of.items():  # every real window, whether or not this arm wrote in it
            if kind == "real":
                chunks[item.chunk_of[key]]["seconds"] += (key[1] - key[0]) + 2 * pad
        for group, rows in (("pair", item.pairs), ("all", item.touched)):
            for row in rows:
                spans = [s for s in b.touching(session, row)]
                if not spans:
                    continue
                chunk = chunks[item.chunk_of[(spans[0]["start"], spans[0]["end"])]]
                own_none = item.owner_of(row["id"]) is None
                for pos in range(len(row["norm"])):
                    windows = [scored[(s["start"], s["end"])] for s in spans if (s["start"], s["end"]) in scored]
                    heard = [o for content, _ in windows for o in content.get((row["id"], pos), ())]
                    status = max((o[0] for o in heard), key=b.RANK.get, default="miss")
                    options = [o for _, copies in windows for o in copies.get((row["id"], pos), ())]
                    attr = max((o[1] for o in options), key=ATTR_RANK.get, default=None)
                    joined = max((o[1] for o in heard), key=ATTR_RANK.get, default=None)
                    chunk[f"{group} total"] += 1
                    chunk[f"{group} {status}"] += 1
                    if status in HEARD:
                        chunk[f"{group} attr {'no track' if own_none else attr}"] += 1
                        # the same axis from the joined alignment alone (5o.4: reported beside it)
                        chunk[f"{group} joined {'no track' if own_none else joined}"] += 1
                    if group == "pair":
                        best_status[(item.name, row["id"], pos)] = status
                    else:
                        chunk["edit add"] += status == "miss"
                        chunk["edit fix"] += status == "misheard"
                        chunk["edit move"] += status in HEARD and attr == "wrong" and not own_none
    return chunks, best_status


def merged(chunks, keys):
    total = Counter()
    for key in keys:
        total.update(chunks.get(key, {}))
    return total


def metrics(c):
    total, heard = c["pair total"], c["pair right"] + c["pair homophone"] + c["pair misheard"]
    out = {
        "correct": (c["pair right"] + c["pair homophone"]) / max(total, 1),
        "heard": heard / max(total, 1),
        "strict": c["pair attr strict"] / max(heard, 1),
        "wrong": c["pair attr wrong"] / max(heard, 1),
        "wrong_abs": c["pair attr wrong"] / max(total, 1),
        "wrong_joined": c["pair joined wrong"] / max(heard, 1),
        "strict_abs": c["pair attr strict"] / max(total, 1),
        "decoy": c["decoy hyp nothing near"] / max(c["decoy written"], 1),
    }
    written = max(c["written"], 1)
    for kind in ("strict", "multi", "wrong", "unlabelled", "no track", "near-miss", "repeat", "nothing near"):
        out[f"hyp {kind}"] = c[f"hyp {kind}"] / written
    minutes = max(c["seconds"], 1e-9) / 60
    out["edit add"] = c["edit add"] / minutes
    out["edit fix"] = c["edit fix"] / minutes
    out["edit move"] = c["edit move"] / minutes
    out["edit delete"] = (c["hyp repeat"] + c["hyp nothing near"]) / minutes
    out["edit drop label"] = c["edit drop label"] / minutes
    all_total = max(c["all total"], 1)
    out["all correct"] = (c["all right"] + c["all homophone"]) / all_total
    out["all wrong_abs"] = c["all attr wrong"] / all_total
    # of the pair characters A0 got right, how many this arm still gets right
    out["retention"] = c["ret kept"] / max(c["ret base"], 1)
    return out


class Arm:
    """One arm = one or more runs; its metrics are the mean over runs (as pre-registered)."""

    def __init__(self, label, runs):
        self.label = label
        self.runs = runs  # [(chunks, best_status)]

    def point(self, keys):
        return self.mean([merged(chunks, keys) for chunks, _ in self.runs])

    @staticmethod
    def mean(counters):
        values = [metrics(c) for c in counters]
        return {k: sum(v[k] for v in values) / len(values) for k in values[0]}


def interval(values):
    values = sorted(values)
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


# ---- arms -----------------------------------------------------------------------------------

def narrow_arms(loaded):
    """The +-0.3 s group (5h.15 / 5h.16 / 5h.18): production, per-track text, the model runs."""
    def a0(item):
        return {key: [(t, x) for t, x, _ in lines] for key, lines in rsg.arm_lines(item.session, _span_align(item), "A0").items()}

    def a1(item):
        return {key: [(t, x) for t, x, _ in lines] for key, lines in rsg.arm_lines(item.session, _span_align(item), "A1").items()}

    arms = [("A0 production", [a0]), ("A1 per-track (anchored)", [a1])]
    for variant, label in (("v1", "A2 v1"), ("v2", "A2 v2 (5h.16)"), ("v2t", "T5 (5h.18)"),
                           ("v2-text", "A2 v2 text only (5o.9 E1)"), ("v2t-text", "T5 text only (5o.9 E1)")):
        prefix = rsg.VARIANTS[variant][0]
        for model in ("lite", "agy"):
            runs = []
            for run in range(1, 6):  # 5o.9 E1 takes v2 / T5 to five runs
                def lines_for(item, prefix=prefix, model=model, run=run, label=label):
                    rsg.PREFIX = prefix
                    rows, failures, _ = rsg.model_rows(item.name, model, run)
                    # A chunk the provider never answered for has no lines, which scores
                    # exactly like "this arm wrote nothing there". The 5o.9 guard aborts on
                    # three *consecutive* failures; the outage that prompted it was scattered
                    # (5 of 17 windows), so it slips through -- say so rather than swallow it
                    # (audit 5q, C).
                    if failures:
                        SILENT_GAPS[(f"{label} {model}", item.name, run)] = failures
                    return {key: [(tuple(r["tracks"]), b.normalize(r["text"])) for r in items] for key, items in (rows or {}).items()}
                if all((proj.ALI / item.name / f"{prefix}-{model}-run{run}.json").exists() for item in loaded):
                    runs.append(lines_for)
            if runs:
                arms.append((f"{label} {model}", runs))
    return arms


_SPAN_ALIGN = {}


def _span_align(item):
    if item.name not in _SPAN_ALIGN:
        _SPAN_ALIGN[item.name] = b.collect(item.session)[0]
    return _SPAN_ALIGN[item.name]


def wide_arms(loaded):
    """The +-1 s group (5h.17 / 5h.19 / 5h.20), plus T0 where it was decoded."""
    cache = {}

    def results(item, model):
        if (item.name, model) not in cache:
            path = proj.ALI / item.name / f"timecut_{model}.json"
            layout = json.loads((proj.ALI / item.name / "timecut_layout.json").read_text(encoding="utf-8"))
            cache[(item.name, model)] = (json.loads(path.read_text(encoding="utf-8")) if path.exists() else None, layout)
        return cache[(item.name, model)]

    def simple(fn):
        return lambda item: {key: [(t, x) for t, x, _ in lines] for key, lines in fn(item).items()}

    def active(item):
        tracks = item.session.tracks
        return proj.activity(tracks, int(max(t["end"] for t in tracks) / proj.FRAME) + 2)

    arms = [("A0-W production", [simple(lambda item: tc.production_lines(item.session))]),
            ("T4 words by time (fixed)", [simple(lambda item: tc.word_lines(item.session, active(item)))])]
    for model, tag in (("whisper", "W"), ("qwen", "Q")):
        if not all(results(item, model)[0] is not None for item in loaded):
            continue
        def variant(name, model=model, **options):  # bind `model` now, not when the arm runs
            return simple(lambda item: tc.variant_lines(item.session, *results(item, model), name, **options))
        arms += [(f"T1-{tag}", [variant("T1")]),
                 (f"T2-{tag} (fallback, no de-leak)", [variant("T2", fallback=True, deleak=False)]),
                 (f"T2-{tag} (5h.17: no fallback, de-leak)", [variant("T2")]),
                 (f"T6-{tag}", [variant("T6")]),
                 (f"T7-{tag}", [variant("T7")])]
        if tag == "Q":
            arms.append(("T1-Q placebo de-leak", [variant("T1", placebo=True)]))
        if all(any(k.endswith("/whole/x") for k in results(item, model)[0]) for item in loaded):
            arms.append((f"T0-{tag} whole window (content only)", [variant("T0")]))
    return arms


# ---- report ---------------------------------------------------------------------------------

def run_group(title, loaded, arms, pad, boots, rng):
    print(f"\n==== {title} ====", flush=True)
    keys = sorted({c for item in loaded for c in item.chunk_of.values()})
    chunk_of_row = {}
    for item in loaded:
        for row in item.pairs:
            spans = b.touching(item.session, row)
            if spans:
                for pos in range(len(row["norm"])):
                    chunk_of_row[(item.name, row["id"], pos)] = item.chunk_of[(spans[0]["start"], spans[0]["end"])]
    scored = []
    for label, fns in arms:
        print(f"  scoring {label} ({len(fns)} run{'s' if len(fns) > 1 else ''})", flush=True)
        scored.append(Arm(label, [score_run(loaded, fn, pad) for fn in fns]))
    base = scored[0]
    base_best = base.runs[0][1]
    for arm in scored:  # retention as per-chunk counts, so a resample only adds them up
        for chunks, best in arm.runs:
            for key, value in base_best.items():
                if value in CORRECT:
                    chunks[chunk_of_row[key]]["ret base"] += 1
                    chunks[chunk_of_row[key]]["ret kept"] += best.get(key) in CORRECT
    # Cluster bootstrap over the 120 s chunks: weights[s, c] = how often chunk c is drawn in resample s.
    weights = np.zeros((boots, len(keys)))
    for s in range(boots):
        for _ in keys:
            weights[s, rng.randrange(len(keys))] += 1

    def boot(arm):
        per_run = []
        for chunks, _ in arm.runs:
            names = sorted({n for c in chunks.values() for n in c})
            table = np.array([[chunks.get(k, {}).get(n, 0) for n in names] for k in keys], dtype=float)
            per_run.append([Counter(dict(zip(names, row))) for row in weights @ table])
        return [arm.mean([run[s] for run in per_run]) for s in range(boots)]

    base_boot = boot(base)
    base_point = base.point(keys)
    report, all_draws = {}, {}
    for arm in scored:
        point = arm.point(keys)
        draws = all_draws[arm.label] = boot(arm) if arm is not base else base_boot
        ci = {k: interval([d[k] for d in draws]) for k in ("correct", "strict", "wrong", "wrong_abs", "decoy", "retention")}
        diff_correct = interval([d["correct"] - e["correct"] for d, e in zip(draws, base_boot)])
        diff_strict = interval([d["strict"] - e["strict"] for d, e in zip(draws, base_boot)])
        checks = [
            ("1 正确召回 >= A0 - 2 pp", point["correct"] >= base_point["correct"] - 0.02, diff_correct, -0.02, "diff"),
            ("2 严格归对 >= A0 + 20 pp", point["strict"] >= base_point["strict"] + 0.20, diff_strict, 0.20, "diff"),
            ("3 归错人 <= 10%", point["wrong"] <= 0.10, ci["wrong"], 0.10, "abs"),
            ("4 诱饵(按字) <= 10%", point["decoy"] <= 0.10, ci["decoy"], 0.10, "abs"),
        ]
        replace = ("5 保留率 >= 95%", point["retention"] >= 0.95, ci["retention"], 0.95, "abs")
        candidate = all(ok for _, ok, *_ in checks)
        verdict = ("可替换" if candidate and replace[1] else "可作候选") if candidate else "不做"
        straddle = [name for name, _, (lo, hi), line, _ in checks + [replace] if lo < line < hi]
        report[arm.label] = {"point": point, "ci": ci, "diff_correct": diff_correct, "verdict": verdict, "straddle": straddle, "runs": len(arm.runs)}
        print(f"\n  {arm.label}  (runs {len(arm.runs)})")
        print(f"    correct {point['correct']:6.1%} [{ci['correct'][0]:.1%}, {ci['correct'][1]:.1%}]  (vs A0 {diff_correct[0]:+.1%}..{diff_correct[1]:+.1%})"
              f"  heard {point['heard']:6.1%}  retention {point['retention']:6.1%} [{ci['retention'][0]:.1%}, {ci['retention'][1]:.1%}]")
        print(f"    strict {point['strict']:6.1%} [{ci['strict'][0]:.1%}, {ci['strict'][1]:.1%}]  wrong {point['wrong']:6.1%} [{ci['wrong'][0]:.1%}, {ci['wrong'][1]:.1%}]"
              f"  wrong/ref {point['wrong_abs']:6.2%} [{ci['wrong_abs'][0]:.2%}, {ci['wrong_abs'][1]:.2%}]  strict/ref {point['strict_abs']:6.1%}"
              f"  (wrong, joined alignment only: {point['wrong_joined']:.1%})")
        print("    written: " + "  ".join(f"{k[4:]} {point[k]:.1%}" for k in point if k.startswith("hyp ")))
        print(f"    decoy (chars nothing near) {point['decoy']:6.1%} [{ci['decoy'][0]:.1%}, {ci['decoy'][1]:.1%}]")
        print("    edits per window-minute: " + "  ".join(f"{k[5:]} {point[k]:.1f}" for k in point if k.startswith("edit ")))
        print(f"    all refs in windows: correct {point['all correct']:.1%}  wrong/ref {point['all wrong_abs']:.2%}")
        for name, ok, (lo, hi), line, kind in checks + [replace]:
            print(f"    {'PASS' if ok else 'FAIL'}  {name}   [{lo:+.1%}, {hi:+.1%}]{'  <- 区间跨线' if lo < line < hi else ''}")
        print(f"    => {verdict}{'（区间跨线：' + '、'.join(n.split()[0] for n in straddle) + '）' if straddle else ''}")
        gaps = {key: n for key, n in SILENT_GAPS.items() if key[0] == arm.label}
        if gaps:
            report[arm.label]["silent_gaps"] = {f"{s} run{r}": n for (_, s, r), n in gaps.items()}
            print("    ⚠ 无应答的片段（该臂在那里被记成零分，不是「写错了」）："
                  + "  ".join(f"{s} run{r}: {n}" for (_, s, r), n in sorted(gaps.items())))
    if "T1-W" in all_draws and "T1-Q" in all_draws:  # 5h.20's question, paired over the same resamples
        gap = report["T1-W"]["point"]["wrong"] - report["T1-Q"]["point"]["wrong"]
        lo, hi = interval([w["wrong"] - q["wrong"] for w, q in zip(all_draws["T1-W"], all_draws["T1-Q"])])
        report["T1-W minus T1-Q wrong"] = {"point": gap, "ci": (lo, hi)}
        print(f"\n  5h.20: wrong(T1-W) - wrong(T1-Q) = {gap:+.1%} [{lo:+.1%}, {hi:+.1%}]  (Qwen's lead counts if >= 4 pp)")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--only", choices=("narrow", "wide"), default=None)
    parser.add_argument("--out", default=None, help="write the numbers as JSON here")
    arguments = parser.parse_args()
    if not arguments.score:
        parser.print_help()
        return
    import subprocess

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"corpus {ref.CORPUS}  HEAD {head}  boot {arguments.boot}", flush=True)
    loaded = [Loaded(name) for name in ref.SESSIONS]
    rng = random.Random(b.SEED)
    out = {"corpus": ref.CORPUS, "head": head}
    if arguments.only in (None, "narrow") and not ref.ENGLISH:
        out["narrow"] = run_group("+-0.3 s (5h.15 / 5h.16 / 5h.18)", loaded, narrow_arms(loaded), live.SPAN_PAD, arguments.boot, rng)
    if arguments.only in (None, "wide"):
        out["wide"] = run_group("W = span +-1 s (5h.17 / 5h.19 / 5h.20)", loaded, wide_arms(loaded), tc.W_PAD, arguments.boot, rng)
    if arguments.out:
        with open(arguments.out, "w", encoding="utf-8") as handle:
            json.dump(out, handle, ensure_ascii=False, indent=1, default=list)


if __name__ == "__main__":
    main()
