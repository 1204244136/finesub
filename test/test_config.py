"""`finesub.config`: locating, parsing and memoizing config.toml.

What a key's value may be is `finesub.settings`' business (`test_settings.py`).
"""

from __future__ import annotations

import pytest

from finesub import config as app_config


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    app_config.clear_config_cache()
    yield
    app_config.clear_config_cache()


def _write(tmp_path, monkeypatch, body: str):
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(path))
    return path


def test_missing_file_reads_as_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(tmp_path / "absent.toml"))

    assert app_config.read_config() == {}
    assert app_config.read_config_with_path() == ({}, None)


def test_the_file_and_its_path_come_back_together(tmp_path, monkeypatch) -> None:
    path = _write(tmp_path, monkeypatch, "[segmentation]\nlength_scale = 0.85\n")

    data, where = app_config.read_config_with_path()

    assert data == {"segmentation": {"length_scale": 0.85}}
    assert where == path


def test_an_edit_is_picked_up_on_the_next_read(tmp_path, monkeypatch) -> None:
    """Mid-run edits take effect: the cache is invalidated by mtime/size."""

    path = _write(tmp_path, monkeypatch, "[vad]\nsilero_assist = true\n")
    assert app_config.read_config()["vad"]["silero_assist"] is True

    path.write_text("[vad]\nsilero_assist = false\n# longer\n", encoding="utf-8")

    assert app_config.read_config()["vad"]["silero_assist"] is False


def test_malformed_toml_names_the_file(tmp_path, monkeypatch) -> None:
    path = _write(tmp_path, monkeypatch, "[segmentation\n")

    with pytest.raises(ValueError, match="Invalid FineSub config TOML"):
        app_config.read_config()
    assert path.exists()
