"""Sortformer diarization runner for asr-playground speaker-clustering worktree.

Usage:
    python run_diarize.py <audio.wav> [--model nvidia/diar_streaming_sortformer_4spk-v2.1]
                              [--out out.json] [--smoke] [--cpu]

Runs end-to-end diarization (16kHz mono wav expected), writes results as JSON
and prints a compact summary: speaker count, turn stats, top-10 segments.
"""
import argparse
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="16kHz mono wav path")
    p.add_argument("--model", default="nvidia/diar_streaming_sortformer_4spk-v2.1",
                   help="HF model id or local .nemo path")
    p.add_argument("--out", default=None, help="output JSON path (default: <audio>.sortformer.json)")
    p.add_argument("--smoke", action="store_true", help="only print first segments + structure")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    return p.parse_args()

def load_model(model_id, use_cuda):
    t0 = time.time()
    if model_id.endswith(".nemo"):
        from nemo.collections.asr.models import SortformerEncLabelModel
        model = SortformerEncLabelModel.restore_from(restore_path=model_id, map_location="cuda" if use_cuda else "cpu", strict=False)
    else:
        from nemo.collections.asr.models import SortformerEncLabelModel
        model = SortformerEncLabelModel.from_pretrained(model_id)
    model.eval()
    if use_cuda:
        model = model.cuda()
    # Recommended high-quality streaming config (30.4s latency) from the model card.
    try:
        model.sortformer_modules.chunk_len = 340
        model.sortformer_modules.chunk_right_context = 40
        model.sortformer_modules.fifo_len = 40
        model.sortformer_modules.spkcache_update_period = 300
    except Exception as e:
        print(f"[warn] streaming param set failed: {e}", file=sys.stderr)
    print(f"[load] {time.time()-t0:.1f}s  model={model_id}")
    return model

def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available() and not args.cpu
    print(f"[info] cuda={torch.cuda.is_available()} device=" +
          (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"))
    model = load_model(args.model, use_cuda)

    t0 = time.time()
    out = model.diarize(audio=[args.audio], batch_size=1)
    elapsed = time.time() - t0
    print(f"[diarize] {elapsed:.1f}s")

    segs = out[0]
    # Normalize each segment into a dict (NeMo returns "start end label" strings)
    norm = []
    for s in segs:
        if isinstance(s, dict):
            label = s.get("label", s.get("speaker", "?"))
            start = s.get("start", s.get("begin_time", 0.0))
            end = s.get("end", s.get("end_time", 0.0))
        elif isinstance(s, str):
            parts = s.split()
            start, end, label = float(parts[0]), float(parts[1]), parts[2]
        else:
            # assume tuple/list (start, end, label)
            start, end, label = float(s[0]), float(s[1]), s[2]
        norm.append({"start": round(float(start), 3), "end": round(float(end), 3), "label": str(label)})
    norm.sort(key=lambda x: x["start"])

    if args.smoke:
        print("--- first 15 segments ---")
        for s in norm[:15]:
            print(s)
        return

    # Summary
    n_spk = sorted({s["label"] for s in norm})
    total = sum(s["end"] - s["start"] for s in norm)
    per_spk = {}
    for s in norm:
        per_spk.setdefault(s["label"], 0.0)
        per_spk[s["label"]] += s["end"] - s["start"]
    print(f"[summary] segments={len(norm)} speakers={len(n_spk)} total_speech={total:.1f}s")
    for lbl in n_spk:
        turns = sum(1 for s in norm if s["label"] == lbl)
        print(f"  {lbl}: {turns} turns, {per_spk[lbl]:.1f}s")

    out_path = args.out or args.audio.replace(".wav", ".sortformer.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "audio": args.audio,
                   "diarize_sec": round(elapsed, 2), "segments": norm}, f, ensure_ascii=False, indent=1)
    print(f"[out] {out_path}")

if __name__ == "__main__":
    main()
