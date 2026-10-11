"""Scorer contracts: abstention, permutation, fragmentation and fusion alignment."""
import numpy as np
import pytest

import accuracy_lab as lab


def fixture_data():
    truth = np.zeros((200, 2), dtype=bool)
    truth[:100, 0] = True
    truth[100:, 1] = True
    stable = [{"start": 0.8, "end": 7.2}, {"start": 8.8, "end": 15.2}]
    p = np.zeros((200, 4), dtype=np.float32)
    p[:100, 0], p[100:, 1] = 0.99, 0.99
    return p, stable, truth


def test_abstention_stays_in_fixed_denominator():
    p, stable, truth = fixture_data()
    p[100:] = 0
    s = lab.score(p, stable, truth, ["a", "b"])
    assert s["strict"] == 1
    assert s["fixed"] == pytest.approx(0.5)
    assert s["coverage"] == pytest.approx(0.5)


def test_a_handful_of_files_does_not_get_a_confidence_interval():
    # Widened from n < 2 to n < 5 (audit 2026-09-18, section 2.9). The old assertion here --
    # bootstrap_mean_ci([0.1, 0.3]) == [0.1, 0.3] -- is exactly the degenerate case it was meant to
    # exclude: two files give three distinct resample means, so the quantiles are an artefact of
    # the grid rather than 95% coverage. Report per-file values below five.
    from accuracy_report import MIN_FILES_FOR_INTERVAL, bootstrap_mean_ci
    assert MIN_FILES_FOR_INTERVAL == 5
    assert bootstrap_mean_ci([0.2]) is None
    assert bootstrap_mean_ci([0.1, 0.3]) is None
    assert bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4]) is None
    assert bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4, 0.5]) is not None


def test_speaker_permutation_does_not_change_accuracy():
    p, stable, truth = fixture_data()
    a = lab.score(p, stable, truth, ["a", "b"])
    b = lab.score(p[:, [2, 1, 3, 0]], stable, truth, ["a", "b"])
    assert a["fixed"] == b["fixed"] == 1


def test_align_uses_activity_and_not_labels():
    p, _, _ = fixture_data()
    np.testing.assert_array_equal(lab.align(p, p[:, [2, 1, 3, 0]]), p)


def test_crossfit_does_not_name_a_track_seen_only_in_scored_fold():
    p, stable, truth = fixture_data()
    s = lab.score(p, stable, truth, ["a", "b"])
    assert s["fixed"] == 1
    assert s["crossfit"] == 0


def test_unlabelled_everywhere_scores_zero_not_nan():
    p, stable, truth = fixture_data()
    s = lab.score(np.zeros_like(p), stable, truth, ["a", "b"])
    assert s["fixed"] == s["strict"] == s["coverage"] == 0


def test_identity_score_penalizes_fragmentation_even_when_purity_is_perfect():
    from accuracy_identity import bcubed
    one_identity = bcubed([0, 0, 0, 0], ["a"]*4, [1]*4)
    fragmented = bcubed([0, 1, 2, 3], ["a"]*4, [1]*4)
    assert one_identity["f1"] == 1
    assert fragmented["precision"] == 1
    assert fragmented["recall"] == 0.25
    assert fragmented["f1"] == pytest.approx(0.4)


def test_identity_score_penalizes_merging_distinct_people():
    from accuracy_identity import bcubed
    merged = bcubed([0, 0], ["a", "b"], [1, 1])
    assert merged["precision"] == 0.5
    assert merged["recall"] == 1


def test_replaying_cache_selection_preserves_vectors_predictions_and_silence():
    torch = pytest.importorskip("torch")
    from accuracy_trace import SelectionTrace
    cache = torch.arange(10, dtype=torch.float32).reshape(1, 2, 5)
    new = cache + 10
    history = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    predictions = history.flip(1)
    silence = torch.full((1, 5), -1.0)

    def selector(indices):
        def update(cache, history, has, silence, n, embs, predictions, current):
            selected = torch.cat([cache, embs], dim=1)[:, indices]
            probs = torch.cat([history, predictions], dim=1)[:, indices]
            return (torch.cat([selected, silence[:, None]], dim=1),
                    torch.cat([probs, torch.zeros((1, 1, 2))], dim=1), True, silence, n)
        return update

    args = (cache, history, True, silence, torch.tensor([1]), new, predictions, history)
    baseline = SelectionTrace(selector([2, 0]))
    expected = baseline(*args)
    assert baseline.rows[0]["indices"] == [2, 0, -1]
    replay = SelectionTrace(selector([3, 1]), baseline.rows, freeze=True)
    actual = replay(*args)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_hybrid_preserves_more_than_four_global_identities():
    from accuracy_hybrid import local_project
    sf = np.zeros((300, 4), dtype=np.float32)
    community = np.zeros((300, 6), dtype=np.float32)
    for i in range(6):
        sf[i*50:(i+1)*50, 0] = 1
        community[i*50:(i+1)*50, i] = 1
    mixed = local_project(sf, community, window=50)
    np.testing.assert_array_equal(mixed.argmax(axis=1), community.argmax(axis=1))
    assert mixed.shape[1] == 6


def test_export_preserves_text_and_rejects_stale_audio_or_overwrite(tmp_path, monkeypatch):
    import json
    import accuracy_export
    monkeypatch.setattr(lab, "WT", tmp_path)
    monkeypatch.setattr(lab, "OUT", tmp_path / "cache")
    audio = tmp_path / "audio.ogg"
    audio.write_bytes(b"test audio identity")
    monkeypatch.setattr(lab, "source", lambda _: audio)
    run = tmp_path / "out" / "refined-2609" / "test"
    run.mkdir(parents=True)
    probs, stable, _ = fixture_data()
    stable[0]["text"], stable[1]["text"] = "first", "second"
    original = {"segments": stable}
    (run / "test-stable.json").write_text(json.dumps(original))
    cache = lab.OUT / "test"
    cache.mkdir(parents=True)
    np.savez_compressed(cache / "community.npz", probs=probs)
    (cache / "community.json").write_text(json.dumps({"identity": {"audio": lab.digest(audio)}}))
    monkeypatch.setattr(lab.rm, "sources", lambda: pytest.fail("Export must not read gold"))
    destination = accuracy_export.export("test", "community", tmp_path / "preview")
    result = json.loads((destination / "test-stable.json").read_text(encoding="utf-8"))
    assert [{k: v for k, v in s.items() if k != "speaker"} for s in result["segments"]] == stable
    assert result["metadata"]["speaker"]["model"] == "pyannote/speaker-diarization-community-1"
    with pytest.raises(FileExistsError):
        accuracy_export.export("test", "community", destination)
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source audio identity"):
        accuracy_export.export("test", "community", tmp_path / "stale")


def test_independent_merge_respects_cannot_links_after_transitive_merge():
    from accuracy_merge import merge_identities
    activity = np.zeros((300, 3), dtype=np.float32)
    for i in range(3):
        activity[i*100:(i+1)*100, i] = 1
    activity[40:70, 2] = 1
    centers = np.array([10, 20, 85, 125, 150, 175, 225, 250, 275])
    embeddings = np.tile([1.0, 0.0], (len(centers), 1))
    merged, evidence = merge_identities(activity, embeddings, centers, 0.1)
    assert evidence["groups"] == [[0, 1], [2]]
    assert (merged[40:70].sum(axis=1) == 2).all()
    # Identical embeddings without enough clean evidence do not license merging.
    _, sparse = merge_identities(activity, embeddings[:1], centers[:1], 1.0)
    assert sparse["groups"] == [[0], [1], [2]]


def test_historical_arms_do_not_follow_the_shipping_default():
    """`base` means the geometry the recorded readings used, not whatever ships today.

    The owner moved `chunk_right_context` 1 -> 80 on 2026-09-20. Arms that inherited the dataclass
    defaults would have followed it and kept their names, so a re-run of `base` would have compared
    against a configuration no recorded reading used (Codex audit 2026-09-21). The three places
    that name these arms must agree, and must not be today's default.
    """
    import sys
    from pathlib import Path
    from dataclasses import asdict
    from finesub.speech.diarization.chunking import SortformerGeometry
    import accuracy_v2
    sys.path.insert(0, str(Path(__file__).parent))
    import candidate_run

    assert lab.BASE["chunk_right_context"] == 1
    for arm in ("base", "cache2", "ema", "refresh", "no_latest", "micro", "split"):
        assert lab.arm_geometry(arm)["chunk_right_context"] == 1, arm
    assert lab.HIGH["chunk_right_context"] == 40
    assert accuracy_v2.BASE == lab.BASE and accuracy_v2.HIGH == lab.HIGH
    assert candidate_run.SORTFORMER_ARMS == {"sortformer-base": lab.BASE, "sortformer-high": lab.HIGH}
    shipping = asdict(SortformerGeometry())
    assert any(shipping[key] != value for key, value in lab.BASE.items()), \
        "if the default ever returns to the old geometry, this test's premise needs rereading"
    for key in lab.BASE:
        assert key in shipping, f"{key} is not a SortformerGeometry field"


def test_candidate_run_records_the_geometry_it_ran():
    """Inference and the metadata both read `sortformer_parameters`; each arm records its own right."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import candidate_run
    from finesub.speech.diarization.chunking import SortformerGeometry
    right = {backend: candidate_run.sortformer_parameters(backend)["chunk_right_context"]
             for backend in ("sortformer-base", "sortformer-high", "sortformer-default")}
    assert right == {"sortformer-base": 1, "sortformer-high": 40,
                     "sortformer-default": SortformerGeometry().chunk_right_context}
    source = Path(candidate_run.__file__).read_text(encoding="utf-8")
    assert '"chunk_right_context": 40' not in source, "the record must not carry its own copy of an arm"
