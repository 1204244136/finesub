"""Gemini 3.5 Transcribe (35t) as a diarizer on the 魔裁 story clips (owner 2026-10-05), scored like the backends.

35t is called as the external-model survey calls it (`tools/ts_diar_explore`: `/interactions`, verbatim, word
timestamps, speaker diarization, ja-JP, fed production's vocal track), in 540 s windows with 60 s overlap (its
output cap) -- 3 windows per 20-min clip, 9 calls. Window labels are linked across the overlap by co-speaking
seconds (`episode.stitch`). Its words become a diar.json timeline (one segment per word, track = linked speaker) next
to a copy of production's stable.json in `out/manosaba-g35t/`, so `manosaba_truth.py --runs out/manosaba
out/manosaba-g35t` scores it on the same ASR rows with production's row assignment (`assign_all`).

Calls go through `g35t.py` (free keys, ledger, quota day); each window's output is cached and reused.

    python tools/diarization_explore/manosaba_g35t.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import manosaba_truth as mt  # noqa: E402
from tools.ts_diar_explore import episode  # noqa: E402
from tools.ts_diar_explore.common import clip, probe_duration  # noqa: E402

WINDOW, OVERLAP = 540.0, 60.0
CACHE = mt.n3.MAIN / "tmp" / "diarization-retrack" / "manosaba" / "g35t"
RUNS = mt.n3.MAIN / "out" / "manosaba-g35t"


def window(clip_name, src, start, end):
    """One window's parsed output, called once and cached."""
    out = CACHE / clip_name / f"win-{start:.0f}-{end - start:.0f}.json"
    if not out.exists():
        audio = clip(src, CACHE / clip_name / f"vocal-{start:.0f}-{end - start:.0f}.wav", start, end - start)
        subprocess.run([sys.executable, "-m", "tools.ts_diar_explore.g35t", str(audio), str(out), "--offset", str(start)],
                       check=True, encoding="utf-8", errors="replace", cwd=str(HERE.parents[1]))
    return json.loads(out.read_text(encoding="utf-8"))


def run(clip_name):
    src = mt.RUNS / f"{clip_name}-vocal.ogg"
    parts = []
    for start, end in episode.windows(0.0, probe_duration(src), WINDOW, OVERLAP):
        result = window(clip_name, src, start, end)
        if result["words"] and all(w["speaker"] is None for w in result["words"]):
            # 35t sometimes answers a window with words but no speaker field at all (manosaba-c 0-540 s); the same
            # audio answers the same way, so split it in two overlapping halves instead of retrying
            half = (end - start + OVERLAP) / 2
            print(f"{clip_name}: {start:.0f}-{end:.0f} s came back without speakers; splitting it", flush=True)
            parts += [(s, e, window(clip_name, src, s, e)) for s, e in ((start, start + half), (end - half, end))]
        else:
            parts.append((start, end, result))
    words = [w for w in episode.stitch(parts, do_link=True)["words"] if w["speaker"] is not None and w["end"] > w["start"]]
    ids = {label: i for i, label in enumerate(dict.fromkeys(w["speaker"] for w in words))}
    RUNS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(mt.RUNS / f"{clip_name}-stable.json", RUNS / f"{clip_name}-stable.json")
    tracks = [{"track": ids[w["speaker"]], "start": w["start"], "end": w["end"]} for w in words]
    (RUNS / f"{clip_name}-diar.json").write_text(json.dumps({"metadata": {"backend": "g35t", "labels": ids}, "tracks": tracks},
                                                            ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{clip_name}: {len(parts)} windows, {len(words)} words, {len(ids)} linked speakers")


def main():
    for clip_name in mt.CLIPS:
        run(clip_name)


if __name__ == "__main__":
    main()
