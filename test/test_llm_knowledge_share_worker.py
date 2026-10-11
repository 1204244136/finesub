"""The mailbox Worker (`deploy/share-worker/src/index.js`) under its Node unit tests.

The Worker is JS, so its tests live beside it and run on Node's built-ins
(`node:test`, `node:sqlite` behind a workerd stand-in; nothing to install).
This file only runs them, so a red JS test fails the default suite. What the
stand-in does and does not cover is in `deploy/share-worker/README.md`; real
workerd is still only exercised by `e2e_check.py` against `wrangler dev`.
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.error
from pathlib import Path

import pytest

from finesub.llm.knowledge.share import client as share_client

WORKER = Path(__file__).resolve().parents[1] / "deploy" / "share-worker"


def _sqlite_flags(node: str) -> list[str] | None:
    """The flags under which this node has `node:sqlite`, or None if it has none.

    Asked rather than read off the version: 22.5-22.12 (and 23.0-23.3) ship it
    behind `--experimental-sqlite`, later releases without, and older ones not
    at all.
    """

    for flags in ([], ["--experimental-sqlite"]):
        try:
            probe = subprocess.run(
                [node, *flags, "-e", "require('node:sqlite')"], capture_output=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if probe.returncode == 0:
            return flags
    return None


def test_the_worker_unit_tests_pass() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; the share Worker's JS tests need Node >= 22.5")
    flags = _sqlite_flags(node)
    if flags is None:
        pytest.skip(f"{node} has no node:sqlite (needs Node >= 22.5); the share Worker's JS tests need it")
    files = sorted(path.relative_to(WORKER).as_posix() for path in (WORKER / "test").glob("*.test.mjs"))
    assert files, "no *.test.mjs under deploy/share-worker/test"
    proc = subprocess.run(
        [node, *flags, "--disable-warning=ExperimentalWarning", "--test", "--test-reporter=spec", *files],
        cwd=WORKER,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=100,
    )
    assert proc.returncode == 0, f"node --test failed:\n{proc.stdout}\n{proc.stderr}"


def test_the_client_sends_the_tag_the_mailbox_compares(monkeypatch) -> None:
    """The Worker answers 304 only to `If-None-Match: "<chain_hash>"` -- the
    exact string its JS test sends. The branch first shipped with no client
    sending the header, so it added an RPC to every pull and saved none
    (reviewer, 2026-09-09); this pins the client half of that contract."""

    sent: list[str | None] = []

    def fake_urlopen(request, timeout=None):  # type: ignore[no-untyped-def]
        sent.append(request.get_header("If-none-match"))
        raise urllib.error.HTTPError(request.full_url, 304, "Not Modified", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(share_client, "_urlopen", fake_urlopen)
    result = share_client.fetch_snapshot("https://srv", known_chain_hash="abc123")
    assert result is share_client.UNCHANGED
    assert sent == ['"abc123"']
