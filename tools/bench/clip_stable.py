"""Cut a stable.json to its first N seconds, for the session-mode A/B (``run_session_ab``).

Keeps the segments that end by the cut, clamps the VAD intervals, drops the
pause hints past it, and records the cut under ``metadata.clip``. Only a cut
from 0 is supported: ``score_session_ab`` aligns against the refined subtitle
with ``--clip-end`` only.

    python -m tools.bench.clip_stable out/x/x-stable.json out/session-ab/clips/x-25m-stable.json 1500
"""
import json, sys
from pathlib import Path
src, dst, end = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
d = json.loads(src.read_text(encoding="utf-8"))
d["segments"] = [s for s in d["segments"] if float(s["end"]) <= end]
vt = d.get("vad_timeline") or {}
iv = []
for it in vt.get("intervals") or []:
    a, b = (it["start"], it["end"]) if isinstance(it, dict) else (it[0], it[1])
    if a < end:
        b = min(b, end)
        iv.append({**it, "end": b} if isinstance(it, dict) else [a, b, *it[2:]])
vt["intervals"] = iv
ph = vt.get("pause_hints")
if isinstance(ph, dict):
    vt["pause_hints"] = {k: [x for x in v if float(x) < end] if isinstance(v, list) else v for k, v in ph.items()}
elif isinstance(ph, list):
    def pos(x):
        if isinstance(x, dict):
            return float(x.get("time", x.get("start", x.get("at", 0))))
        return float(x[0] if isinstance(x, (list, tuple)) else x)
    vt["pause_hints"] = [x for x in ph if pos(x) < end]
if not d["segments"]:
    raise SystemExit(f"no segment ends by {end} s")
d["metadata"] = {**(d.get("metadata") or {}), "clip": {"source": str(src), "start": 0.0, "end": end}}
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
print(dst, len(d["segments"]), "segments, last end", d["segments"][-1]["end"], "intervals", len(iv))
