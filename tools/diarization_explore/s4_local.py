"""Plan 5h.8, the GPU arms: the same calls as s4_live, answered by Qwen3-ASR and Whisper.

Same jobs (s4_live.jobs_for), same scorer (s4_live.score with another result file).
Both models are deterministic, so L0's second copy is the first one again -- the
text-volume control that Live gets for free is weaker here; the anchor-side split
(drowned speaker's own track vs the other one) carries the mechanism check.

  Qwen3-ASR   no timestamps: the anchor's own transcript is stripped from the front,
              exactly as for Live
  Whisper     word timestamps: anchored results keep only the words that start after
              the anchor and the gap (the stored A/<k> is then empty, so nothing more
              is stripped)
"""

from __future__ import annotations

import argparse
import json

import s4_headroom as ref
import s4_live as live
import s4_projection as proj

BATCH = 48


def run_qwen():
    from finesub.speech.verification.qwen_referee import QwenReferee

    referee = QwenReferee(device="cuda")
    try:
        for session in ref.SESSIONS:
            path = proj.ALI / session / "qwen.json"
            done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            jobs = [(j, a) for j, a in live.jobs_for(session) if not j.endswith("/L0/1") and j not in done]
            for start in range(0, len(jobs), BATCH):
                batch = jobs[start: start + BATCH]
                texts = referee.transcribe_batch([audio for _, audio in batch])
                for (job_id, _), (text, _) in zip(batch, texts):
                    done[job_id] = text
                path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
            for job_id in list(done):  # deterministic: the "second" plain run is the first again
                if job_id.endswith("/L0/0"):
                    done[job_id[:-1] + "1"] = done[job_id]
            path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
            print(session, "qwen jobs", len(done), flush=True)
    finally:
        referee.close()


def run_whisper():
    from finesub.speech.recognition.fw_refine_backend import RefinedWhisperModel
    from finesub.speech.recognition.transcribe import REFINE_SEC

    model = RefinedWhisperModel("large-v3-turbo", device="cuda", compute_type="float16", refine_sec=REFINE_SEC)
    for session in ref.SESSIONS:
        path = proj.ALI / session / "whisper.json"
        done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        jobs = live.jobs_for(session)
        anchor_seconds = {j.replace("/A/", "/L1/"): len(a) / live.SR for j, a in jobs if "/A/" in j}
        for job_id, audio in jobs:
            if job_id in done or job_id.endswith("/L0/1"):
                continue
            words = proj.decode(model, audio, 0.0)
            if "/L1/" in job_id:
                cut = anchor_seconds[job_id] + live.GAP - 0.15
                done[job_id] = "".join(w["word"] for w in words if w["start"] >= cut)
            elif "/A/" in job_id:
                done[job_id] = ""  # already cut by timestamp; nothing left to strip
            else:
                done[job_id] = "".join(w["word"] for w in words)
        for job_id in list(done):
            if job_id.endswith("/L0/0"):
                done[job_id[:-1] + "1"] = done[job_id]
        path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        print(session, "whisper jobs", len(done), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--whisper", action="store_true")
    parser.add_argument("--score", action="store_true")
    arguments = parser.parse_args()
    if arguments.qwen:
        run_qwen()
    if arguments.whisper:
        run_whisper()
    if arguments.score:
        for result in ("qwen.json", "whisper.json"):
            if all((proj.ALI / s / result).exists() for s in ref.SESSIONS):
                live.score(result)


if __name__ == "__main__":
    main()
