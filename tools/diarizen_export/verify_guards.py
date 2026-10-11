"""Inject each bug this round's tests claim to catch, and confirm the suite turns red.

⚠ A test that has never failed is a test whose failure mode is unverified. This round added
guards for six behaviours; each one gets its bug put back, the affected file run, and the source
restored -- so "it passes" becomes "it would have caught this".
"""
import subprocess
import sys
from pathlib import Path

PY = r"C:\Users\Carl\miniconda3\envs\asr\python.exe"

CASES = [
    ("evidence: Sortformer must report no evidence",
     "src/finesub/speech/diarization/sortformer_backend.py",
     "        return BackendOutput(runner.diarize(waveform))",
     "        return BackendOutput(runner.diarize(waveform), object())",
     "test/test_diarization_evidence.py"),
    ("samples: closeness must go through the mapping",
     "src/finesub/speech/diarization/samples.py",
     "    mapping = getattr(evidence, \"centroid_of_track\", None)",
     "    mapping = None  # BUG: index soft by track id",
     "test/test_diarization_evidence.py"),
    ("samples: a shared stretch is not exclusive",
     "src/finesub/speech/diarization/samples.py",
     "    alone = (data[:, track] > 0) & (data.sum(axis=1) == 1)",
     "    alone = data[:, track] > 0",
     "test/test_diarization_evidence.py"),
    ("turns: overlap must never be called a handover",
     "tools/diarizen_export/turns.py",
     "    if simultaneous > max_simultaneous:\n        return None  # overlap, not a handover",
     "    if False:\n        return None  # overlap, not a handover",
     "tools/diarizen_export/test_turns.py"),
    ("turns: the cut must be an interior word boundary",
     "tools/diarizen_export/turns.py",
     "    edges = [edge for edge in edges if start < edge < end]",
     "    edges = [edge for edge in edges if start <= edge <= end]",
     "tools/diarizen_export/test_turns.py"),
    ("evidence: the sidecar must stay removable, not become a record",
     "src/finesub_bootstrap/artifacts.py",
     '    "-diar-evidence.npz",\n)',
     ")",
     "test/test_diarization_evidence.py"),
    # --- 2026-09-20 audit round -----------------------------------------------------------
    ("hints: the evidence must survive the timeline round trip",
     "src/finesub/speech/diarization/stage.py",
     "            single_speaker_log_bayes=None if evidence is None else float(evidence),",
     "            single_speaker_log_bayes=None,  # BUG: drop it on read",
     "test/test_diarization_stage.py"),
    ("provenance: stable must name the backend that made the timeline",
     "src/finesub/speech/diarization/stage.py",
     "            model=model_for(result.backend),",
     "            model=MODEL_NAME,  # BUG: hardcode Sortformer",
     "test/test_diarization_stage.py"),
    ("assembly: an all-masked file is an empty timeline, not an exception",
     "src/finesub/speech/diarization/diarizen_assemble.py",
     "    if clusters < 1:",
     "    if False:  # BUG: let the negative dimension through",
     "test/test_diarizen_port.py"),
    ("samples: the scored window is the delivered window",
     "src/finesub/speech/diarization/samples.py",
     "    if end - start <= seconds:\n        return [(start, end)]",
     "    if True:\n        return [(start, end)]  # BUG: score whole runs again",
     "test/test_diarization_evidence.py"),
    ("cost: a row nobody labelled is not free",
     "tools/diarization_explore/refined_multi.py",
     "        if not tr and not price_unlabelled:",
     "        if not tr:  # BUG: abstention is a discount again",
     "tools/diarization_explore/test_cost_metrics.py"),
    ("contract: role labels must not pass as people truth",
     "tools/diarization_explore/eval_contract.py",
     '"hina", TruthKind.ROLES,',
     '"hina", TruthKind.PEOPLE,  # BUG: call roles people',
     "tools/diarization_explore/test_eval_contract.py"),
    ("A3: diarizen's declared grid must match the array it returns",
     "src/finesub/speech/diarization/diarizen_backend.py",
     "    frame_seconds=NATIVE_STEP,",
     "    frame_seconds=0.08,  # BUG: declare the old grid, return the native one",
     "test/test_diarization_backends.py"),
    ("crossfit: the held-out fold's truth must not reach the training mapping",
     "tools/diarization_explore/cost_metrics.py",
     "        usable = ~held_out[lo:hi]",
     "        usable = np.ones(hi - lo, dtype=bool)  # BUG: read the whole segment again",
     "tools/diarization_explore/test_cost_metrics.py"),
]

failures = []
for name, path, good, bad, tests in CASES:
    source = Path(path)
    original = source.read_text(encoding="utf-8")
    if original.count(good) != 1:
        failures.append(f"{name}: anchor not found ({original.count(good)} matches)")
        continue
    source.write_text(original.replace(good, bad), encoding="utf-8")
    try:
        result = subprocess.run([PY, "-m", "pytest", "-q", "-x", tests],
                                capture_output=True, text=True)
        caught = result.returncode != 0
    finally:
        source.write_text(original, encoding="utf-8")
    print(f"{'RED  ' if caught else '⚠ GREEN'}  {name}")
    if not caught:
        failures.append(f"{name}: the bug went unnoticed")

print()
if failures:
    print("PROBLEMS:")
    for line in failures:
        print("  " + line)
    sys.exit(1)
print(f"all {len(CASES)} injected bugs were caught")
