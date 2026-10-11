"""Acceptance 2 of the marks' production port (`docs/plans/diar-marks-plan.md`, 「验收（预注册）」): on another genre
(CALLHOME Japanese, two-person telephone calls), how many lines does the product path flag, and how many errors does
it catch? Descriptive; the thresholds are not refitted here.

Rules written before the run:
* rerun the product path (`run_diarization` with its sidecar, then `apply_to_stable`); whether the new timeline equals
  step 1's cached one is reported, not a gate;
* a line's person: the channel with more counted 20 ms frames in it, a frame counting for a channel when that channel's
  silero VAD says voiced and its level is no more than 20 dB below the other channel's; no counted frame -> no truth;
* a track's owner: the majority by line seconds; ordinary lines are those on tracks at least 0.8 pure; back-channel
  lines (<= 4 characters, punctuation not counted) are read separately;
* stop and show the owner when the flagged share of ordinary non-back-channel lines is outside 10-45%, or the errors
  caught are under 50%; errors caught is read only with at least 30 wrong lines.

    python tools/diarization_explore/marks_callhome_check.py     # GPU; run `hw status` first
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marks_step1 as ms  # noqa: E402
import p2_features as pf  # noqa: E402

OUT = ms.CALLHOME_OUT / "marks"
PURE = 0.8
GUARD_DB = 20.0
BAND = (0.10, 0.45)
CAUGHT = 0.50
MIN_WRONG = 30


def counted_frames(call):
    """(per-channel counted 20 ms frames, frame count) on the cut the pipeline ran on."""
    import librosa
    import soundfile
    import torch
    from silero_vad import load_silero_vad

    import edit_judge_truth as et
    import s4_callhome as sc

    vad = load_silero_vad()
    begin = max(0.0, min(s for _, s, _, _ in sc.utterances(et.CALLHOME_CHA / f"{call}.cha")) - sc.EDGE)
    signal, rate = soundfile.read(str(et.CALLHOME_CHA.parent / "mp3" / f"{call}.mp3"), dtype="float32", always_2d=True)
    chans = [librosa.resample(signal[int(begin * rate):, ch], orig_sr=rate, target_sr=16000) for ch in range(2)]
    mono, _ = soundfile.read(str(et.CALLHOME_CHA.parent / "audio" / f"{call}.wav"), dtype="float32")
    a, b = (chans[0] + chans[1])[:30 * 16000], mono[:30 * 16000]
    best = max(range(-80, 81), key=lambda k: float(np.dot(a[max(0, k):len(a) + min(0, k)], b[max(0, -k):len(b) - max(0, k)])))
    assert abs(best) <= 2, (call, "channels and the pipeline's mono are not aligned", best)
    n = int(min(len(c) for c in chans) / 16000 / pf.STEP)
    voiced, level = [], []
    for c in chans:
        x = torch.from_numpy(c[:len(c) // 512 * 512])
        probs = vad.audio_forward(x, 16000).squeeze(0).numpy()
        vad.reset_states()
        at = (np.arange(n) * pf.STEP / 0.032).astype(int).clip(0, len(probs) - 1)
        voiced.append(probs[at] >= 0.5)
        hop = int(pf.STEP * 16000)
        level.append(10 * np.log10((c[:n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-10))
    counted = [voiced[ch] & (level[ch] >= level[1 - ch] - GUARD_DB) for ch in range(2)]
    return counted, n


def main():
    import edit_judge_truth as et
    from finesub.speech.diarization import marks
    from finesub.speech.diarization.evidence import evidence_path
    from finesub.speech.diarization.stage import apply_to_stable, run_diarization, write_timeline

    OUT.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(Counter)
    for ident in et.CALLHOME:
        call = ident.split("-", 1)[1]
        run = et.CALLHOME_RUNS / call
        diar = OUT / f"{call}-diar.json"
        if not diar.exists():
            write_timeline(run_diarization(run / f"{call}-vocal.ogg", device="cuda", evidence_path=evidence_path(diar)), diar)
        cached = json.loads((ms.CALLHOME_OUT / f"{call}-diar.json").read_text(encoding="utf-8"))["tracks"]
        same = json.loads(diar.read_text(encoding="utf-8"))["tracks"] == cached
        stable = OUT / f"{call}-stable.json"
        shutil.copyfile(run / f"{call}-stable.json", stable)
        apply_to_stable(stable, diar)
        payload = json.loads(stable.read_text(encoding="utf-8"))
        assert payload["metadata"]["speaker"]["marks"] is not None, call
        counted, n = counted_frames(call)

        lines = []
        seconds = defaultdict(Counter)
        for row in payload["segments"]:
            speaker = row.get("speaker") or {}
            lo, hi = int(float(row["start"]) / pf.STEP), min(n, int(np.ceil(float(row["end"]) / pf.STEP)))
            if hi <= lo or speaker.get("track") is None:
                continue
            frames = [int(c[lo:hi].sum()) for c in counted]
            if not any(frames) or frames[0] == frames[1]:
                continue
            person = int(np.argmax(frames))
            track = int(speaker["track"])
            seconds[track][person] += float(row["end"]) - float(row["start"])
            backchannel = marks.characters(str(row.get("text", ""))) <= marks.BACKCHANNEL_CHARS
            lines.append((track, person, backchannel, (speaker.get("mark") or {}).get("tier") == "low"))
        owner = {k: c.most_common(1)[0][0] for k, c in seconds.items()}
        purity = {k: c.most_common(1)[0][1] / sum(c.values()) for k, c in seconds.items()}
        for track, person, backchannel, low in lines:
            kind = ("back-channel" if backchannel else "ordinary") if purity[track] >= PURE else "merged"
            wrong = owner[track] != person
            totals[kind].update(lines=1, flagged=low, wrong=wrong, caught=wrong and low)
        print(call, f"timeline identical to step 1's: {same}", dict(totals["ordinary"]), flush=True)

    for kind, c in totals.items():
        caught = f"caught {c['caught'] / c['wrong']:.1%} of {c['wrong']} wrong" if c["wrong"] >= MIN_WRONG else \
            f"{c['wrong']} wrong (< {MIN_WRONG}: not read)"
        print(f"{kind}: {c['lines']} lines, flagged {c['flagged'] / max(1, c['lines']):.1%}, {caught}")
    o = totals["ordinary"]
    share = o["flagged"] / max(1, o["lines"])
    caught = o["caught"] / o["wrong"] if o["wrong"] >= MIN_WRONG else None
    stop = not BAND[0] <= share <= BAND[1] or (caught is not None and caught < CAUGHT)
    print(f"VERDICT: {'stop and show the owner' if stop else 'within the band'} (flagged {share:.1%}, "
          f"caught {'not read' if caught is None else f'{caught:.1%}'})")


if __name__ == "__main__":
    main()
