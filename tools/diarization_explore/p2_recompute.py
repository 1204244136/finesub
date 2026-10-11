"""P2 step 1, GPU part: everything production computes on the way to a timeline, kept (owner 2026-10-05).

For every file of `p2_split.json` (dev 43 + held-out 15), or the `--only` ones (the 魔裁 clips `manosaba-a/b/c` too):
  1. the production timeline from the production entry `run_diarization` (cached in tmp/diarization-abstain/ when
     an earlier run made it; otherwise run once) -- the A/A reference;
  2. one pass of the production steps with nothing thrown away: Nemotron-3 frame probabilities, `local`
     (= `windows(rasterised(probs))`), WeSpeaker embeddings, the VBx clustering (`hard`, `soft`,
     `centroid_of_track`) -- saved to tmp/diarization-p2/<file>.npz;
  3. A/A: the timeline rebuilt from that pass (`assemble` -> `segments_from_activity`) must equal the
     production timeline segment by segment; the verdict goes to tmp/diarization-p2/aa.json.

    python tools/diarization_explore/p2_recompute.py [--only hina-01 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nemotron3_probe as n3  # noqa: E402

sys.path.insert(0, str(n3.WT / "src"))

OUT = n3.MAIN / "tmp" / "diarization-p2"
PRODUCTION = n3.MAIN / "tmp" / "diarization-abstain"
SPLIT = json.loads((HERE / "p2_split.json").read_text(encoding="utf-8"))


MANOSABA = n3.MAIN / "out" / "manosaba"  # the 魔裁 story clips (`manosaba_truth.py`); production ran there


def vocal(ident):
    if ident.startswith("manosaba-"):
        return MANOSABA / f"{ident}-vocal.ogg"
    if ident.startswith("onetonton-"):
        return n3.MAIN / "out" / ident / f"{ident}-vocal.ogg"
    return n3.MAIN / "out" / "refined-2609" / ident / f"{ident}-vocal.ogg"


def production_timeline(ident):
    from finesub.speech.diarization.stage import run_diarization, write_timeline
    path = MANOSABA / f"{ident}-diar.json" if ident.startswith("manosaba-") else PRODUCTION / ident / "nemotron3-diar.json"
    if not path.exists():
        result = run_diarization(vocal(ident), device="cuda")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_timeline(result, path)
    return json.loads(path.read_text(encoding="utf-8"))["tracks"]


def one_pass(ident, handle):
    import soundfile
    import torch

    from finesub.speech.diarization import diarizen_assemble, diarizen_cluster, diarizen_embed
    from finesub.speech.diarization import nemotron3_backend as nb
    from finesub.speech.diarization import nemotron3_model
    from finesub.speech.preprocessing.audio import TARGET_SR, resample_if_needed

    signal, rate = soundfile.read(str(vocal(ident)), dtype="float32", always_2d=True)
    waveform, _ = resample_if_needed(torch.from_numpy(signal.mean(axis=1)).unsqueeze(0), rate, TARGET_SR)
    probs = nb.frame_activity(waveform, handle=handle, device="cuda")
    probs = probs.detach().cpu().numpy() if hasattr(probs, "detach") else np.asarray(probs)
    local = nb.windows(nb.rasterised(probs, samples=int(waveform.shape[-1]), source_step=nemotron3_model.FRAME_SECONDS))
    embeddings = diarizen_embed.embed(handle.embedding, waveform, local, device="cuda", only_active=True)
    clustered = diarizen_cluster.cluster(embeddings, local, handle.plda_dir)
    native = diarizen_assemble.assemble(local, clustered.hard).astype(np.float32)
    np.savez_compressed(OUT / f"{ident}.npz", probs=probs, frame_seconds=nemotron3_model.FRAME_SECONDS,
                        local=local, embeddings=embeddings, hard=clustered.hard, soft=clustered.soft,
                        centroid_of_track=clustered.centroid_of_track, native=native)
    return native


def same_timeline(native, reference):
    from finesub.speech.diarization.nemotron3_backend import NATIVE_STEP
    from finesub.speech.diarization.segments import segments_from_activity
    ours = [(t.track, round(t.start, 6), round(t.end, 6)) for t in segments_from_activity(native.tolist(), frame_seconds=NATIVE_STEP)]
    theirs = [(int(t["track"]), round(float(t["start"]), 6), round(float(t["end"]), 6)) for t in reference]
    return ours == theirs, len(ours), len(theirs)


def machine_free():
    """Global rule: no heavy GPU work while another project's formal timing runs (`hw status` exit 3)."""
    import shutil
    import subprocess
    # ⚠ Windows Python does not resolve `hw` to `hw.cmd` through PATHEXT; a bare name is FileNotFoundError.
    hw = shutil.which("hw")
    if hw is None:
        raise SystemExit("hw not found on PATH -- cannot check the machine, not running")
    status = subprocess.run([hw, "status", "--hold", "5m"], capture_output=True, text=True)
    print(status.stdout.strip().splitlines()[0] if status.stdout.strip() else "hw status: no output", flush=True)
    if status.returncode == 1:
        raise SystemExit(f"hw status failed: {status.stderr.strip()[:200]}")
    return status.returncode == 0


def main():
    if not machine_free():
        raise SystemExit("hw status: busy (another project's formal timing) -- run later")
    import vram_guard
    vram_guard.install()
    from finesub.speech.diarization.stage import resolve_backend

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    verdicts_path = OUT / "aa.json"
    verdicts = json.loads(verdicts_path.read_text(encoding="utf-8")) if verdicts_path.exists() else {}
    handle = resolve_backend(None).ensure(None, "cuda")
    for ident in args.only or (SPLIT["dev"] + SPLIT["heldout"]):
        if ident in verdicts and (OUT / f"{ident}.npz").exists():
            continue
        reference = production_timeline(ident)
        native = one_pass(ident, handle)
        same, ours, theirs = same_timeline(native, reference)
        verdicts[ident] = {"identical": same, "segments": ours, "production_segments": theirs}
        verdicts_path.write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
        print(ident, "A/A", "ok" if same else "DIFFERENT", ours, theirs, flush=True)


if __name__ == "__main__":
    main()
