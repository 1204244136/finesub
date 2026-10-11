"""Plan 5r B2: re-decoded overlap candidates shown to the correction round (experiment, off by default)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from finesub.llm import overlap_candidates as oc
from finesub.llm.prompts import _load_prompt_template


def window(*spans):
    return SimpleNamespace(segments=[SimpleNamespace(start=a, end=b) for a, b in spans])


def test_without_the_switch_nothing_is_rendered(monkeypatch):
    monkeypatch.delenv(oc.ENV, raising=False)
    assert oc.candidate_rows(window((0.0, 2.0))) == ""


def test_a_candidate_lists_the_rows_it_overlaps(monkeypatch, tmp_path):
    path = tmp_path / "clip-overlap-candidates.json"
    path.write_text(json.dumps([
        {"start": 1.5, "end": 4.5, "track": 1, "text": "そうそう|ほんとに"},
        {"start": 9.0, "end": 9.5, "track": 0, "text": "outside"},
    ]), encoding="utf-8")
    oc._load.cache_clear()
    monkeypatch.setenv(oc.ENV, str(path))
    rows = oc.candidate_rows(window((0.0, 2.0), (2.0, 4.0), (4.45, 6.0)))
    # 4.45-4.5 shares less than MIN_SHARED, so row 3 is not covered; the pipe is escaped
    assert rows == "s1|1,2|そうそう｜ほんとに"


def test_the_fragment_carries_the_rows_verbatim():
    block = _load_prompt_template("fragment_overlap_candidates_v1.md", candidate_rows="s1|1,2|そうそう")
    assert "<overlap_candidates>\ntrack|covers|text\ns1|1,2|そうそう\n</overlap_candidates>" in block
