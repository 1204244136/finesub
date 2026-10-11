"""The knowledge write gate asks which base a write lands in, not only whether
this process is a worktree: from a linked worktree the shared bases (the main
checkout's, the installation's) are read, never written by accident; any
other root is the caller's own (architecture review 2026-09-25, candidate 3)."""

from __future__ import annotations

import json
from pathlib import Path

from finesub.llm.knowledge import base as knowledge_base
from finesub.llm.knowledge.base import knowledge_signals_enabled, worktree_blocks_knowledge_write
from finesub.llm.knowledge.node.repo import KnowledgeRepo


def _checkout(path: Path) -> Path:
    (path / "src" / "finesub").mkdir(parents=True)
    (path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    return path


def _main_and_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """A main checkout and a linked worktree of it, laid out the way git does:
    the worktree's ``.git`` is a file pointing at ``<main>/.git/worktrees/<name>``."""

    main = _checkout(tmp_path / "main")
    (main / ".git" / "worktrees" / "slot").mkdir(parents=True)
    worktree = _checkout(tmp_path / "main" / ".worktrees" / "slot")
    (worktree / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'slot'}\n", encoding="utf-8")
    return main, worktree


def _clean_env(monkeypatch) -> None:
    for name in ("FINESUB_ROOT", "FINESUB_KNOWLEDGE_ROOT", "FINESUB_KNOWLEDGE_WRITE", "FINESUB_CHECKOUT_DATA"):
        monkeypatch.delenv(name, raising=False)


def test_a_worktree_may_not_write_the_main_checkout_base(tmp_path, monkeypatch) -> None:
    main, worktree = _main_and_worktree(tmp_path)
    _clean_env(monkeypatch)
    monkeypatch.chdir(worktree)

    assert worktree_blocks_knowledge_write(None)  # the implicit root is the main one
    assert worktree_blocks_knowledge_write(main / "knowledge")  # explicit, still shared
    assert not worktree_blocks_knowledge_write(tmp_path / "scratch" / "knowledge")

    monkeypatch.setenv("FINESUB_KNOWLEDGE_WRITE", "1")
    assert not worktree_blocks_knowledge_write(main / "knowledge")


def test_the_main_checkout_writes_its_own_base(tmp_path, monkeypatch) -> None:
    main, _worktree = _main_and_worktree(tmp_path)
    _clean_env(monkeypatch)
    monkeypatch.chdir(main)
    assert not worktree_blocks_knowledge_write(main / "knowledge")


def _pretend_worktree_sharing(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(knowledge_base, "is_linked_worktree", lambda: True)
    monkeypatch.setattr(knowledge_base, "shared_knowledge_roots", lambda: {root.resolve()})
    monkeypatch.delenv("FINESUB_KNOWLEDGE_WRITE", raising=False)
    monkeypatch.setattr(knowledge_base, "_SIGNAL_SKIP_WARNED", set())


def test_skipped_signals_warn_once_per_root(tmp_path, monkeypatch, reported) -> None:
    _pretend_worktree_sharing(monkeypatch, tmp_path)
    assert knowledge_signals_enabled(tmp_path) is False
    assert knowledge_signals_enabled(tmp_path) is False
    assert [w.code for w in reported.warnings] == ["knowledge-signals-skipped"]
    assert knowledge_signals_enabled(tmp_path / "other") is True


def test_the_correction_run_books_no_signal_into_a_shared_base(tmp_path, monkeypatch, reported) -> None:
    from types import SimpleNamespace

    from finesub.llm.stages.correction import run as run_module

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "原神", "intro": "游戏", "category": "common"})
    raw = SimpleNamespace(chunk_id="w1", segments=[SimpleNamespace(text="原神")], source_ids=["s1"])
    landed = SimpleNamespace(chunk_id="w1", segments=[SimpleNamespace(text="元神")], source_ids=["s1"])
    segment = SimpleNamespace(source_ids=["s1"], corrected_text="原神")

    def events() -> dict[str, int]:
        rows = KnowledgeRepo.open(tmp_path).store.conn.execute(
            "SELECT kind, COUNT(*) FROM events GROUP BY kind"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def book(task: str) -> None:
        run_module._shadow_scan_windows(tmp_path, task, "rev:1", [raw])
        run_module._log_landed_windows(tmp_path, task, "rev:1", [landed], [segment])

    book("open")  # control: with the gate open both kinds are booked
    booked = events()
    assert booked.get("matched", 0) > 0 and booked.get("landed", 0) > 0
    _pretend_worktree_sharing(monkeypatch, tmp_path)
    book("gated")
    assert events() == booked
    assert [w.code for w in reported.warnings] == ["knowledge-signals-skipped"]


def test_agent_tasks_carry_the_gate_to_the_mcp_server(tmp_path, monkeypatch) -> None:
    """The MCP server books `exposed` from its own process, so the harness
    decides and says so in the manifest; absent means on."""

    from finesub.llm.client import _agent_task_inputs
    from finesub.llm.routing.config import CapabilityTier

    KnowledgeRepo.open(tmp_path)

    def metadata() -> dict:
        inputs = _agent_task_inputs(
            [{"role": "system", "content": "p"}, {"role": "user", "content": "u"}],
            validator_spec=None, variant="", capability_tier=CapabilityTier.CAPABLE,
            max_repair_attempts=0, retrieval="none",
            call_kwargs={"task": "knowledge", "kb_tools": "read", "knowledge_root": str(tmp_path),
                         "knowledge_identity": "rev:1"},
        )
        return json.loads(json.dumps(inputs["metadata"]))

    assert "kb_signals" not in metadata()
    _pretend_worktree_sharing(monkeypatch, tmp_path)
    assert metadata()["kb_signals"] == "off"


def test_the_mcp_server_books_no_exposure_when_told_off(tmp_path, monkeypatch) -> None:
    from finesub.llm.agent.agent_mcp_server import KB_TOOL_NAMES, HarnessToolServer

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "原神", "intro": "游戏", "category": "common",
                                         "section_order": ["档案"]})
    monkeypatch.setenv("FINESUB_MCP_KNOWLEDGE_ROOT", str(tmp_path))

    class Ledger:  # `kb_read` books each successful reply into the runtime
        def record_kb_read(self, **_kwargs) -> None:
            pass

    def read(signals: str | None) -> int:
        server = HarnessToolServer(Ledger(), assignment_id="a", worker_id="w", session_id="s",
                                   instance_id="i", tools=list(KB_TOOL_NAMES))
        server._task = {"task_id": "t", "lease_generation": 1}
        metadata = {"knowledge_identity": "rev:1", "kb_tools": "read", "kb_signal_task": "run"}
        if signals:
            metadata["kb_signals"] = signals
        server._manifest = {"metadata": metadata}
        reply = server._call("kb_read", {"entry": "原神"}, rpc_id=1)
        assert reply["isError"] is False
        return KnowledgeRepo.open(tmp_path).store.conn.execute(
            "SELECT COUNT(*) FROM events WHERE kind='exposed'"
        ).fetchone()[0]

    assert read("off") == 0
    assert read(None) > 0
