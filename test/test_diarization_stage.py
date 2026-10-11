"""The diarization stage's file-level behaviour, without loading a model.

The split this file guards: the expensive half is cached because it cannot be
reproduced, and the cheap half rewrites stable.json and *can* be redone. Getting
that backwards would either recompute a non-reproducible answer (losing labels a
human already corrected downstream) or cache a join that should follow the data.
"""

from __future__ import annotations

import json

import pytest

from finesub.speech.diarization.segments import TrackSegment
from finesub.speech.diarization.stage import (
    ARTIFACT_ENV,
    BACKENDS,
    DiarizationResult,
    DiarizationUnavailable,
    apply_to_stable,
    is_annotated,
    labels_for,
    read_timeline,
    resolve_artifacts,
    write_timeline,
)

TIMELINE = DiarizationResult(
    tracks=(
        TrackSegment(track=0, start=0.0, end=2.0),
        TrackSegment(track=1, start=2.0, end=4.0),
    ),
    frame_seconds=0.08,
    overlap=0.0,
)


def stable_payload():
    return {
        "metadata": {"asr": {"model": "large-v3-turbo"}},
        "segments": [
            {"start": 0.0, "end": 1.8, "text": "first"},
            {"start": 2.1, "end": 3.9, "text": "second"},
            {"start": 9.0, "end": 9.5, "text": "after the timeline ends"},
        ],
    }


def test_missing_artifacts_name_the_way_out(tmp_path, monkeypatch):
    monkeypatch.delenv(ARTIFACT_ENV, raising=False)
    with pytest.raises(DiarizationUnavailable, match="export_sortformer"):
        resolve_artifacts(tmp_path / "nowhere")


def test_the_lossless_vocal_delivery_is_brought_to_16k_mono(tmp_path, monkeypatch):
    # `resolve_vocal_audio` falls back to the separator's `.flac`, which keeps the
    # model's 44.1 kHz stereo. The stage used to refuse anything but 16 kHz, so
    # `--diarize` could not run on a task that only had the lossless delivery.
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    soundfile = pytest.importorskip("soundfile")
    import numpy as np

    from finesub.speech.diarization import runner as runner_module
    from finesub.speech.diarization.stage import run_diarization

    seconds, rate = 2.0, 44100
    tone = np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, int(seconds * rate))).astype(np.float32) * 0.1
    flac = tmp_path / "clip-vocal.flac"
    soundfile.write(str(flac), np.stack([tone, tone], axis=1), rate)
    (tmp_path / "sortformer_core.ts").write_bytes(b"")
    seen = {}

    class FakeRunner:
        """⚠ Stands in for the whole protocol, not just `diarize`.

        The real runner holds about 300 MB of GPU modules and is closed by the caller, so a double
        that only answers `diarize` stops the caller at the `with`. Asserting `closed` here is what
        keeps the release from being dropped later without a test noticing.
        """

        def __init__(self, directory, device="cuda", geometry=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exception):
            self.close()

        def close(self):
            seen["closed"] = True

        def diarize(self, waveform):
            seen["shape"] = tuple(waveform.shape)
            return torch.zeros((int(waveform.shape[-1] / 1280), 4))

    monkeypatch.setattr(runner_module, "SortformerRunner", FakeRunner)
    run_diarization(flac, artifacts=tmp_path, device="cpu", backend="sortformer")
    assert seen["shape"][0] == 1, "mono, one waveform"
    assert abs(seen["shape"][1] - int(seconds * 16000)) <= 2, "resampled to 16 kHz"
    assert seen.get("closed"), "the runner's GPU modules are handed back, not left to the allocator"


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A machine whose card and model files a test can switch off one at a time."""
    from finesub.speech.runtime import device as device_module

    (tmp_path / "sortformer_core.ts").write_bytes(b"")
    monkeypatch.setenv(ARTIFACT_ENV, str(tmp_path))
    state = {"cuda": True}
    monkeypatch.setattr(device_module, "cuda_usable", lambda: state["cuda"])
    return state


def _paths(tmp_path, *, timeline: bool):
    """The three members of `PipelinePaths` the stage touches."""

    class Paths:
        stable_json = tmp_path / "clip-stable.json"
        diar_json = tmp_path / "clip-diar.json"

        @staticmethod
        def resolve_vocal_audio():
            raise AssertionError("labelling from a cached timeline must not reach the model")

    Paths.stable_json.write_text(json.dumps(stable_payload()), encoding="utf-8")
    if timeline:
        write_timeline(TIMELINE, Paths.diar_json)
    return Paths


def test_labels_from_a_cached_timeline_need_neither_the_card_nor_the_model(tmp_path, machine):
    # Applying a timeline is JSON to JSON: a machine that cannot diarize can still
    # write the labels an earlier run's timeline carries.
    from finesub.stages import _run_diarization_stage

    machine["cuda"] = False
    paths = _paths(tmp_path, timeline=True)
    _run_diarization_stage(paths, diarize="acoustics", device=None, gpu_tier="entry", announce=lambda *a, **k: None)
    assert is_annotated(paths.stable_json)


def test_on_gets_a_stage_line():
    from finesub.stages import PIPELINE_STAGE_ORDER, _planned_stages

    everything = max(PIPELINE_STAGE_ORDER.values())
    assert "diarize" in _planned_stages(everything, diarize="acoustics")
    assert "diarize" not in _planned_stages(everything, diarize="off")


def four_busy_tracks(seconds=25.0):
    return DiarizationResult(
        tracks=tuple(TrackSegment(track=k, start=k * seconds, end=(k + 1) * seconds) for k in range(4))
        + (TrackSegment(track=1, start=10.0, end=15.0),),
        frame_seconds=0.08,
        overlap=0.0,
    )


def test_the_timeline_carries_the_acoustic_hints(tmp_path):
    path = write_timeline(four_busy_tracks(), tmp_path / "clip-diar.json")
    hints = json.loads(path.read_text(encoding="utf-8"))["metadata"]["hints"]
    # s1's extra 5 s sit inside s0's 25 s; s0 (25 s) is the shorter of the two (s1: 30 s)
    assert hints["co_activity"]["s0-s1"] == pytest.approx(5 / 25, abs=1e-4)
    assert "not_same" not in hints, "plan 5q: co-activity is reported as a number, never as a verdict"
    # 30 / 25 / 25 / 25 s of track over 100 s of speech
    assert hints["fourth_track_share"] == pytest.approx(0.25)
    assert "capacity_warning" not in hints, "plan 5o.5: no threshold told >=5 speakers from four"


def test_a_short_track_inside_a_long_one_reports_its_share_and_no_verdict(tmp_path):
    # The share is relative to the shorter track, so a 5 s track made of moments the model
    # held one voice on two slots reads 1.0. That is why it stays a number (plan 5q).
    result = DiarizationResult(
        tracks=(TrackSegment(0, 0.0, 100.0), TrackSegment(1, 40.0, 45.0)), frame_seconds=0.08, overlap=0.0
    )
    assert result.hints["co_activity"] == {"s0-s1": 1.0}
    assert set(result.hints) == {"co_activity", "fourth_track_share"}


def test_fewer_than_four_tracks_hold_no_fourth_share(tmp_path):
    assert TIMELINE.hints["fourth_track_share"] == 0.0


def test_stable_gets_the_same_hints(tmp_path):
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    timeline = write_timeline(TIMELINE, tmp_path / "clip-diar.json")
    apply_to_stable(stable, timeline)
    hints = json.loads(stable.read_text(encoding="utf-8"))["metadata"]["speaker"]["hints"]
    assert hints == {"co_activity": {"s0-s1": 0.0}, "fourth_track_share": 0.0}


def test_stable_and_the_timeline_carry_identical_hints(tmp_path):
    # Plan 5o.1 K2: the timeline is written at millisecond precision; hints derived
    # from the finer numbers of the run would differ from those of a later read.
    #
    # ⚠ The fixture carries EVIDENCE, and that is the point of it. This test existed with an
    # evidence-free result, so it stayed green through the whole window in which
    # `from_payload` dropped `single_speaker_log_bayes` and `samples` -- while asserting, by its
    # own name, the invariant that was broken. A test whose fixture cannot express the failure
    # is not covering it.
    fine = DiarizationResult(
        tracks=(TrackSegment(0, 0.0, 30.00049), TrackSegment(1, 9.00049, 42.0)),
        frame_seconds=0.08,
        overlap=0.0,
        backend="diarizen",
        single_speaker_log_bayes=63.14159,
        log_bayes_embeddings=20,
        samples=({"track": 0, "start": 1.0, "end": 6.0, "closeness": 0.2},),
    )
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    timeline = write_timeline(fine, tmp_path / "clip-diar.json")
    apply_to_stable(stable, timeline)
    written = json.loads(timeline.read_text(encoding="utf-8"))["metadata"]["hints"]
    assert "single_speaker_log_bayes" in written and "samples" in written, "the fixture must bite"
    assert json.loads(stable.read_text(encoding="utf-8"))["metadata"]["speaker"]["hints"] == written
    assert fine.hints == written, "what the run holds is what the file says"
    # And a third time, because the product path is run -> write -> read -> apply: a value that
    # survives the write but not the read is dropped on the FIRST run, not only on resume.
    assert read_timeline(timeline).hints == written


def test_stable_records_the_backend_that_actually_made_the_timeline(tmp_path):
    """⚠ `-stable.json` is a RECORD artifact: a wrong model name there is never cleaned up.

    `to_payload` resolved the model through the backend registry while `apply_to_stable`
    hardcoded Sortformer's checkpoint, so every DiariZen run wrote the wrong provenance into the
    one file that is kept forever.
    """
    result = DiarizationResult(tracks=(TrackSegment(0, 0.0, 10.0),), frame_seconds=0.08,
                               overlap=0.0, backend="diarizen")
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    apply_to_stable(stable, write_timeline(result, tmp_path / "clip-diar.json"))
    speaker = json.loads(stable.read_text(encoding="utf-8"))["metadata"]["speaker"]
    assert speaker["backend"] == "diarizen"
    assert speaker["model"] == BACKENDS["diarizen"].model
    assert "sortformer" not in speaker["model"].lower()


def test_the_environment_variable_is_honoured(tmp_path, monkeypatch):
    (tmp_path / "sortformer_core.ts").write_bytes(b"")
    monkeypatch.setenv(ARTIFACT_ENV, str(tmp_path))
    assert resolve_artifacts() == tmp_path


def test_an_explicit_path_wins_over_the_environment(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    (explicit / "sortformer_core.ts").write_bytes(b"")
    other = tmp_path / "other"
    other.mkdir()
    (other / "sortformer_core.ts").write_bytes(b"")
    monkeypatch.setenv(ARTIFACT_ENV, str(other))
    assert resolve_artifacts(explicit) == explicit


def test_the_timeline_round_trips(tmp_path):
    path = write_timeline(TIMELINE, tmp_path / "x-diar.json", artifacts_revision="rev1")
    again = read_timeline(path)
    assert again.tracks == TIMELINE.tracks
    assert again.frame_seconds == pytest.approx(TIMELINE.frame_seconds)


def test_the_timeline_records_that_it_is_not_reproducible(tmp_path):
    path = write_timeline(TIMELINE, tmp_path / "x-diar.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["reproducible"] is False


def test_labels_follow_the_timeline_and_stop_where_it_stops():
    speakers = labels_for(stable_payload()["segments"], TIMELINE)
    assert [speaker.track for speaker in speakers] == [0, 1, None]


def test_apply_to_stable_writes_labels_and_keeps_the_rest(tmp_path):
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    timeline = write_timeline(TIMELINE, tmp_path / "x-diar.json")

    apply_to_stable(stable, timeline)

    payload = json.loads(stable.read_text(encoding="utf-8"))
    assert payload["segments"][0]["speaker"]["track"] == 0
    assert "speaker" not in payload["segments"][2], "no track there, so no label"
    assert payload["metadata"]["asr"]["model"] == "large-v3-turbo", "other metadata survives"
    assert payload["metadata"]["speaker"]["track_count"] == 2


def test_applying_twice_is_idempotent(tmp_path):
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    timeline = write_timeline(TIMELINE, tmp_path / "x-diar.json")

    apply_to_stable(stable, timeline)
    first = stable.read_text(encoding="utf-8")
    apply_to_stable(stable, timeline)
    assert stable.read_text(encoding="utf-8") == first, "the cheap half may always be redone"


def test_no_temporary_file_is_left_behind(tmp_path):
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    apply_to_stable(stable, write_timeline(TIMELINE, tmp_path / "x-diar.json"))
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".")] == []


def test_is_annotated_is_false_for_a_plain_stable(tmp_path):
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    assert is_annotated(stable) is False, "the file exists long before anyone diarizes it"


def test_is_annotated_becomes_true_after_applying(tmp_path):
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    apply_to_stable(stable, write_timeline(TIMELINE, tmp_path / "x-diar.json"))
    assert is_annotated(stable) is True


def test_is_annotated_survives_a_corrupt_file(tmp_path):
    stable = tmp_path / "x-stable.json"
    stable.write_text("{not json", encoding="utf-8")
    assert is_annotated(stable) is False


def test_is_annotated_is_false_when_the_file_is_absent(tmp_path):
    assert is_annotated(tmp_path / "missing.json") is False


@pytest.fixture
def warnings_said(monkeypatch):
    from finesub import reporting
    import finesub.stages as stages_module

    said = []

    class Recorder(reporting.NullReporter):
        def warning(self, code, message, *, impact="", action=""):
            said.append(code)

    monkeypatch.setattr(reporting, "current_reporter", lambda: Recorder())
    monkeypatch.setattr(stages_module, "current_reporter", lambda: Recorder())
    return said


def test_diarization_follows_a_usable_card(monkeypatch, warnings_said):
    from finesub.speech.runtime import device as device_module
    from finesub.stages import _diarization_device

    monkeypatch.setattr(device_module, "cuda_unusable_reason", lambda: None)
    assert _diarization_device(None, "entry") == "cuda"
    assert "diarization-on-cpu" not in warnings_said


@pytest.mark.parametrize(
    "device, tier", [("cpu", "entry"), (None, "cpu")], ids=["--device cpu", "--gpu-tier cpu"]
)
def test_diarization_runs_on_cpu_when_the_run_does_and_says_how_slow(
    monkeypatch, warnings_said, device, tier
):
    """Both backends run on CPU now, so a run without a GPU still gets labels -- and is told
    the cost, because an hour of audio then takes about fifty minutes."""
    from finesub.speech.runtime import device as device_module
    from finesub.stages import _diarization_device

    monkeypatch.setattr(device_module, "cuda_unusable_reason", lambda: None)
    assert _diarization_device(device, tier) == "cpu"
    assert "diarization-on-cpu" in warnings_said
