"""Run the task-flow acceptance arms (plan deleted; `git show 3a3dce18:docs/plans/llm-task-execution-plan.md` §11).

One "run" = one full ``python -m finesub.llm.correction_translation --execute``
over a frozen stable.json (text profile, retrieval=local, knowledge=none,
fast off), in its own artifact directory, with the arm's flow switch and
model pins. Arms are interleaved (bench-discipline: never all of A then all
of B) and a finished run is skipped, so the script can be re-invoked to
continue. Every run writes ``run.json`` (arm, command, env, wall clock, exit
code) next to its outputs; ``score_flow_ab.py`` reads those.

    python -m tools.bench.run_flow_ab --stable <p2-stable.json> --out out/flow-ab --n 3 --arms batched-flash agentic-flash

Model pins (owner 2026-09-09): the capable arms run on agy 3.6 Flash (the
weakest Gemini agy offers -- there is no 3.5 on agy); the lite arm puts the
correction cell on the free API 3.5 Flash Lite. The legacy-flow arms of the
§11 acceptance (L-*) no longer exist: stage D deleted that flow; their
readings stay in `docs/bench-baselines.md` 二十八.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]

AGY_FLASH = "local-agy-media-gemini-3_6-flash"
API_LITE = "gemini-free-3_5-flash-lite"

#: arm -> (strategy pin, model pins). The legacy flow is gone (stage D), so
#: an arm now names a strategy: `batched` (the request-turn shape, an agent
#: standing in for the request-bound API path) or `agentic` (tool session).
ARMS: Dict[str, tuple[str, Dict[str, str]]] = {
    "batched-flash": ("batched", {"research": AGY_FLASH, "correction-text": AGY_FLASH}),
    "agentic-flash": ("agentic", {"research": AGY_FLASH, "correction-text": AGY_FLASH}),
    "batched-lite": ("batched", {"research": AGY_FLASH, "correction-text": API_LITE}),
}


def _pins_for_media(pins: Dict[str, str], media: str) -> Dict[str, str]:
    """The text task groups become their multimodal twins under an audio/video profile."""

    if media == "text":
        return dict(pins)
    return {group.replace("-text", "-mm"): target for group, target in pins.items()}


def _command(
    stable: Path,
    out_srt: Path,
    artifact_dir: Path,
    pins: Dict[str, str],
    *,
    media: str,
    audio: Path | None,
    max_window_subtitle_tokens: int | None,
) -> List[str]:
    cmd = [
        sys.executable, "-m", "finesub.llm.correction_translation", str(stable),
        "--execute", "--fast", "off", "--media", media, "--retrieval", "local", "--knowledge", "none",
        "--task-artifact-dir", str(artifact_dir), "-o", str(out_srt),
    ]
    if audio is not None:
        cmd += ["--audio", str(audio)]
    for group, target in pins.items():
        cmd += ["--llm-model", f"{group}={target}"]
    return cmd


def run_one(
    *,
    arm: str,
    index: int,
    stable: Path,
    out_root: Path,
    force: bool = False,
    media: str = "text",
    audio: Path | None = None,
    max_window_subtitle_tokens: int | None = None,
) -> Dict[str, object]:
    flow, pins = ARMS[arm]
    pins = _pins_for_media(pins, media)
    run_dir = out_root / arm / f"run{index:02d}"
    meta_path = run_dir / "run.json"
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("exit_code") == 0:
            return {"skipped": True, **meta}
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_dir / "artifacts"
    out_srt = run_dir / f"{stable.stem.replace('-stable', '')}.srt"
    cmd = _command(
        stable, out_srt, artifact_dir, pins,
        media=media, audio=audio, max_window_subtitle_tokens=max_window_subtitle_tokens,
    )
    env = dict(os.environ)
    env["FINESUB_TASK_STRATEGY"] = flow
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if max_window_subtitle_tokens is not None:
        # The planner cap has no CLI flag -- it is `[chunking]` in config.toml
        # (CLAUDE.md: option defaults live in the backend). A run of these
        # arms has no config.toml in effect otherwise (the checkout carries
        # none), so a file holding only this section is defaults + the cap.
        config = out_root / "config.toml"
        config.write_text(
            f"[chunking]\nmax_window_subtitle_tokens = {int(max_window_subtitle_tokens)}\n",
            encoding="utf-8",
        )
        env["FINESUB_CONFIG_FILE"] = str(config)
    started = time.time()
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
    meta = {
        "arm": arm,
        "index": index,
        "flow": flow,
        "media": media,
        "audio": str(audio) if audio else "",
        "max_window_subtitle_tokens": max_window_subtitle_tokens,
        "pins": pins,
        "command": cmd,
        "started": started,
        "wall_seconds": round(time.time() - started, 1),
        "exit_code": proc.returncode,
        "repo": str(REPO),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stable", required=True)
    parser.add_argument("--out", default="out/flow-ab")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--media", default="text", help="text|audio|video")
    parser.add_argument("--audio", help="source audio for --media audio")
    parser.add_argument(
        "--max-window-subtitle-tokens", type=int, default=None,
        help="planner cap (production default 10000); lower it to force several windows on a short clip",
    )
    args = parser.parse_args(argv)
    stable = Path(args.stable).resolve()
    out_root = Path(args.out).resolve()
    audio = Path(args.audio).resolve() if args.audio else None
    for index in range(1, args.n + 1):
        # Interleave: run index i of every arm before index i+1 of any arm.
        for arm in args.arms:
            meta = run_one(
                arm=arm, index=index, stable=stable, out_root=out_root, force=args.force,
                media=args.media, audio=audio, max_window_subtitle_tokens=args.max_window_subtitle_tokens,
            )
            state = "skipped" if meta.get("skipped") else f"exit={meta['exit_code']} {meta['wall_seconds']}s"
            print(f"{arm} run{index:02d}: {state}", flush=True)  # product output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
