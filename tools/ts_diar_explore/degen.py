"""Where open MOSS degenerated, found with signals a pipeline has at run time, and how it reads everywhere else.

Detector (no reference involved): cut the run into 30 s bins and hold MOSS's text against a second ASR on the
same audio -- production's Whisper words (AliMeeting: mai2, there is no production Chinese arm). A bin where the
checker heard >= MIN_CHARS characters is degenerate when MOSS has less than half or more than twice as many
characters (dropped, stopped, looping), or when their character bigrams share less than MIN_SIM (the text is
somebody else's: timestamps jumped). On clean speech the two agree closely (bigram share 0.8-1.0); jumped bins
sit at 0.02-0.17, dropped ones at 0 characters -- the thresholds are not tuned into a narrow gap.

Readings: reference lines whose midpoint falls in a clean bin, scored for the MOSS run and the other arms on the
same lines (text CER and speaker B-cubed from `score.py`). Pro is shown only where it is clean on nearly all of MOSS's clean bins, and then on the bins both keep.
A bin is flagged when the two ASRs disagree, whichever of them is wrong (on hina-71 it is Whisper, which
translated and looped over a Chinese stretch that MOSS got right) -- either way a pipeline would fall back there.

    python -m tools.ts_diar_explore.degen            # every open-MOSS run
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from . import arms as A
from . import score as S
from . import timing as T
from .report import reference

BIN = 30.0
MIN_CHARS = 20
MIN_SIM = 0.5
PRO_SHARE = 0.95
#: (item, arm, span) of every open-MOSS run: whole-episode and 15 min single passes, then the 5 min window arm.
RUNS = [
    ("hina-72", "moss-single", None), ("hina-70", "moss-single", None), ("hina-71", "moss-single", None),
    ("hina-73", "moss-single", None), ("hina-68", "moss-single", (0, 3600)),
    ("hina-73", "moss-single", (1320, 2220)), ("hina-68", "moss-single", (1920, 2820)),
    ("hina-67", "moss-single", (1260, 2160)), ("hina-64", "moss-single", (900, 1800)),
    ("hina-53", "moss-single", (60, 960)),
    ("ali-R8001_M8004", "moss-single", None), ("ali-R8009_M8018", "moss-single", None),
    ("ali-R8007_M8010", "moss-single", None), ("gold-BV1cqLR6hEp3", "moss-single", None),
    *[(i, "moss", None) for i in ("hina-72", "hina-70", "hina-71", "hina-64", "hina-67", "hina-73", "hina-68",
                                  "ali-R8001_M8004", "ali-R8009_M8018", "ali-R8007_M8010", "gold-BV1cqLR6hEp3")],
]


def _chars_by_bin(units: list[dict], lo: float, n: int) -> list[str]:
    """Each unit's normalised text spread over the bins it spans, in proportion to time."""
    out = [""] * n
    for u in units:
        t = S.norm(u["text"])
        a, b = u["start"], max(u["end"], u["start"] + 0.01)
        k0, k1 = int((a - lo) // BIN), int((b - lo - 1e-6) // BIN)
        if not t or k1 < 0 or k0 >= n:
            continue
        pos = 0
        for k in range(k0, k1 + 1):
            share = len(t) - pos if k == k1 else round(len(t) * (min(b, lo + (k + 1) * BIN) - max(a, lo + k * BIN)) / (b - a))
            if 0 <= k < n:
                out[k] += t[pos:pos + share]
            pos += share
    return out


def _bigram_share(x: str, y: str) -> float:
    bx, by = Counter(x[i:i + 2] for i in range(len(x) - 1)), Counter(y[i:i + 2] for i in range(len(y) - 1))
    return sum((bx & by).values()) / max(1, min(sum(bx.values()), sum(by.values())))


def checker(item: str) -> list[dict]:
    return A.load(item, "mai2-diar" if item.startswith("ali-") else "prod")["words"]


def clean_bins(arm: dict, check: list[dict], lo: float, hi: float) -> list[tuple[float, float, bool]]:
    """(start, end, clean) per bin. Bins the checker heard almost nothing in count as clean: nothing to compare."""
    n = int(np.ceil((hi - lo) / BIN))
    m, c = _chars_by_bin(arm["segments"], lo, n), _chars_by_bin(check, lo, n)
    out = []
    for k, (x, y) in enumerate(zip(m, c)):
        bad = len(y) >= MIN_CHARS and (len(x) < 0.5 * len(y) or len(x) > 2 * len(y) or _bigram_share(x, y) < MIN_SIM)
        bad |= len(x) > 2 * max(len(y), MIN_CHARS)  # text where the checker heard nothing: a loop in silence
        out.append((lo + k * BIN, min(hi, lo + (k + 1) * BIN), not bad))
    return out


def _in(spans: list[tuple[float, float]], x: dict) -> bool:
    mid = (x["start"] + x["end"]) / 2
    return any(a <= mid < b for a, b in spans)


def _reference(item: str) -> list[dict]:
    """The survey reference; the single-speaker gold clip has none, so its human word cues stand in (text only)."""
    if item == "gold-BV1cqLR6hEp3":
        return [{**w, "speaker": None} for w in T.srt(T.FIXED)]
    return reference(item)


def _pro(item: str, span: tuple[float, float] | None) -> dict | None:
    for sp in ([span] if span else []) + [None, (0, 3600)]:
        if A.exists(item, "mosspro", span=sp) if sp else A.exists(item, "mosspro"):
            return A.load(item, "mosspro", span=sp) if sp else A.load(item, "mosspro")
    return None


def main() -> None:
    print(f"bins {BIN:.0f} s; degenerate when the checker has >= {MIN_CHARS} chars and MOSS has <0.5x / >2x of them "
          f"or bigram share < {MIN_SIM}")
    print(f"{'run':<34} {'clean':>6}  {'CER: moss / prod / mai2 / Pro':<32} {'B3: moss / DiariZen / n3vbx / mai2 / Pro'}")
    rows = []
    for item, name, span in RUNS:
        if not A.exists(item, name, span=span) if span else not A.exists(item, name):
            continue
        arm = A.load(item, name, span=span) if span else A.load(item, name)
        check = checker(item)
        lo, hi = span or (0.0, max(w["end"] for w in check))
        bins = clean_bins(arm, check, lo, hi)
        spans = [(a, b) for a, b, ok in bins if ok]
        pro = _pro(item, span)
        if pro is not None:  # Pro only when it is clean nearly everywhere MOSS is; then on the bins both keep
            pro_ok = {a for a, _, ok in clean_bins(pro, check, lo, hi) if ok}
            both = [(a, b) for a, b in spans if a in pro_ok]
            if len(both) >= PRO_SHARE * len(spans):
                spans = both
            else:
                pro = None
        heard = [len(S.norm(w["text"])) for w in check if lo <= w["start"] < hi]
        kept = sum(len(S.norm(w["text"])) for w in check if lo <= w["start"] < hi and _in(spans, w)) / max(1, sum(heard))
        ref = [r for r in _reference(item) if lo <= r["start"] < hi and _in(spans, r)]
        arms = {"moss": arm, "DiariZen": A.load(item, "diarizen"), "n3vbx": A.load(item, "n3vbx"),
                "prod": A.load(item, "prod") if A.exists(item, "prod") else None,
                "mai2": A.load(item, "mai2-diar") if A.exists(item, "mai2-diar") else None, "Pro": pro}
        cer, b3 = {}, {}
        for k, x in arms.items():
            if x is None:
                continue
            if k in ("moss", "prod", "mai2", "Pro") and any(S.norm(r["text"]) for r in ref):
                c = S.cer([r for r in ref if S.norm(r["text"])], {"words": [u for u in S.tokens(x) if _in(spans, u)]}, 0, 1e9)
                cer[k] = c["cer"] if c["ref_chars"] else None
            if k != "prod" and any(r["speaker"] for r in ref):
                s = S.speaker(ref, x, 0, 1e9)
                b3[k] = s.get("b3_f1")
        f = lambda d, ks: " / ".join(f"{d[k]:.3f}" if d.get(k) is not None else "  —  " for k in ks)  # noqa: E731
        label = f"{item} {name}" + (f" {span[0]}-{span[1]}" if span else "")
        print(f"{label:<34} {kept:6.1%}  {f(cer, ['moss', 'prod', 'mai2', 'Pro']):<32} {f(b3, ['moss', 'DiariZen', 'n3vbx', 'mai2', 'Pro'])}")
        rows.append((label, kept, cer, b3))


if __name__ == "__main__":
    main()
