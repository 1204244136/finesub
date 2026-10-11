"""A line the library moved to another entry moves on pull too.

The pull looked a membership up by (canonical id, parent). A move keeps the
canonical id and changes the parent, so the moved line matched nothing, got a
second membership under its new parent, and stayed under the old one: the
puller ended up with the line in both entries. A line moved locally since the
library last placed it -- by hand, or by a knowledge update merging or splitting
entries -- stays where it was moved (owner 2026-10-02); so does a line under an
entry of the user's own, or under one retired locally.
"""

from __future__ import annotations

from finesub.llm.knowledge.node.store import KnowledgeStore
from finesub.llm.knowledge.share.exchange import chain_hash, content_digest, snapshot_content
from finesub.llm.knowledge.share.server import ShareService
from finesub.llm.knowledge.share.sync import apply_snapshot


def _seal(service: ShareService) -> None:
    """Chain a direct server-store edit the way an accepted verdict would."""

    digest_value = content_digest(snapshot_content(service.store))
    prev = service.store.conn.execute(
        "SELECT chain_hash FROM share_chain ORDER BY rev DESC LIMIT 1"
    ).fetchone()
    service.store.conn.execute(
        "INSERT INTO share_chain(rev, content_digest, chain_hash) VALUES (?, ?, ?)",
        (service.store.current_rev(), digest_value,
         chain_hash(prev["chain_hash"] if prev else "", digest_value)),
    )


def _service(tmp_path) -> ShareService:  # type: ignore[no-untyped-def]
    service = ShareService(tmp_path / "srv", maintainer_token="mt")
    with service.store.begin("import") as txn:
        for sid, surface in (("X", "游戏X"), ("Y", "游戏Y"), ("Z", "游戏Z")):
            txn.create_node(sid, "subject", {"surface": surface, "intro": "", "category": "common",
                                             "section_order": ["角色"]})
        txn.create_node("T", "term", {"surface": "アリス", "zh": "爱丽丝", "desc": "主角"})
        txn.create_membership("M", "X", "T", "角色", 0)
    _seal(service)
    return service


def _local(client: KnowledgeStore, surface: str) -> str:
    return next(n.local_id for n in client.subjects() if n.payload.get("surface") == surface)


def _move_on_server(service: ShareService, parent: str) -> None:
    with service.store.begin("share", note="move") as txn:
        txn.move_membership("M", parent_id=parent)
    _seal(service)


def _parents_of_term(client: KnowledgeStore) -> list[str]:
    term = next(n for n in client.nodes_of_kind("term") if n.payload.get("surface") == "アリス")
    return sorted(m.parent_id for m in client.parents(term.local_id))


def test_a_line_the_library_moved_moves_on_pull(tmp_path) -> None:
    service = _service(tmp_path)
    with KnowledgeStore(tmp_path / "client.sqlite") as client:
        apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-1")
        assert _parents_of_term(client) == [_local(client, "游戏X")]
        _move_on_server(service, "Y")
        report = apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-2")
        assert report.moved_memberships == 1 and report.created_memberships == 0
        assert _parents_of_term(client) == [_local(client, "游戏Y")]
        # the next pull finds the moved row where the library has it
        again = apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-3")
        assert again.moved_memberships == 0 and again.created_memberships == 0


def test_a_line_moved_away_and_back_follows_both_moves(tmp_path) -> None:
    """The return trip finds the CLOSED version under the old entry first; it
    must still move the live row back rather than take that as "removed here"."""

    service = _service(tmp_path)
    with KnowledgeStore(tmp_path / "client.sqlite") as client:
        apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-1")
        _move_on_server(service, "Y")
        apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-2")
        _move_on_server(service, "X")
        report = apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-3")
        assert report.moved_memberships == 1
        assert _parents_of_term(client) == [_local(client, "游戏X")]


def test_a_line_moved_locally_stays_where_it_was_moved(tmp_path) -> None:
    """A knowledge update merged the line into Z (no retirement); the library
    later moves it to Y. Following would undo the merge on every pull."""

    service = _service(tmp_path)
    with KnowledgeStore(tmp_path / "client.sqlite") as client:
        apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-1")
        membership = client.conn.execute(
            "SELECT membership_id FROM membership_versions WHERE valid_to_rev IS NULL"
            " AND parent_id=?", (_local(client, "游戏X"),)).fetchone()[0]
        with client.begin("harness") as txn:
            txn.move_membership(membership, parent_id=_local(client, "游戏Z"))
        # the library still has the old place (push is add-only): no disagreement
        unchanged = apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-2")
        assert unchanged.moved_memberships == 0 and unchanged.created_memberships == 0
        assert unchanged.diverged_moves == 0
        # the library moved it too, somewhere else: counted, local kept
        _move_on_server(service, "Y")
        report = apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-3")
        assert report.moved_memberships == 0 and report.created_memberships == 0
        assert report.diverged_moves == 1
        assert _parents_of_term(client) == [_local(client, "游戏Z")]


def test_a_line_moved_away_and_back_locally_follows_the_library(tmp_path) -> None:
    service = _service(tmp_path)
    with KnowledgeStore(tmp_path / "client.sqlite") as client:
        apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-1")
        x, z = _local(client, "游戏X"), _local(client, "游戏Z")
        membership = client.conn.execute(
            "SELECT membership_id FROM membership_versions WHERE valid_to_rev IS NULL"
            " AND parent_id=?", (x,)).fetchone()[0]
        with client.begin("user") as txn:
            txn.move_membership(membership, parent_id=z)
        with client.begin("user") as txn:
            txn.move_membership(membership, parent_id=x, order_key=3)
        _move_on_server(service, "Y")
        report = apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-2")
        assert report.moved_memberships == 1
        assert _parents_of_term(client) == [_local(client, "游戏Y")]


def test_only_lines_this_remote_resolved_are_moved(tmp_path) -> None:
    """Canonical ids are per corpus and public: another library naming our
    membership's id must not re-parent it."""

    from finesub.llm.knowledge.share.sync import _live_elsewhere

    service = _service(tmp_path)
    with KnowledgeStore(tmp_path / "client.sqlite") as client:
        apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-1")
        term = next(n.local_id for n in client.nodes_of_kind("term"))
        x = _local(client, "游戏X")
        canonical = client.conn.execute(
            "SELECT canonical_membership_id FROM membership_versions WHERE child_id=?"
            " AND valid_to_rev IS NULL", (term,)).fetchone()[0]
        assert canonical
        # this remote resolved both ends: the move is ours to follow
        assert _live_elsewhere(client, canonical, term, {term, x}) is not None
        # another remote that happens to know the line but not its entry
        assert _live_elsewhere(client, canonical, term, {term}) is None
        # ... or neither
        assert _live_elsewhere(client, canonical, term, set()) is None


def test_a_local_line_the_pull_claimed_follows_the_library(tmp_path) -> None:
    """A pull claims an existing local row in place (it stamps the canonical id
    on the user's version); the library's later move still moves it."""

    service = _service(tmp_path)
    with service.store.begin("import") as txn:
        txn.create_node("U", "term", {"surface": "ボブ", "zh": "鲍勃", "desc": "配角"})
    _seal(service)
    with KnowledgeStore(tmp_path / "client.sqlite") as client:
        apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-1")
        bob = next(n.local_id for n in client.nodes_of_kind("term") if n.payload.get("surface") == "ボブ")
        with client.begin("user") as txn:
            txn.create_membership("L", _local(client, "游戏X"), bob, "角色", 1)
        with service.store.begin("share", note="place") as txn:
            txn.create_membership("M2", "X", "U", "角色", 1)
        _seal(service)
        claimed = apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-2")
        assert claimed.created_memberships == 0  # claimed, not duplicated
        with service.store.begin("share", note="move") as txn:
            txn.move_membership("M2", parent_id="Y")
        _seal(service)
        report = apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-3")
        assert report.moved_memberships == 1
        assert sorted(m.parent_id for m in client.parents(bob)) == [_local(client, "游戏Y")]


def test_a_line_kept_under_a_local_entry_is_not_duplicated(tmp_path) -> None:
    service = _service(tmp_path)
    with KnowledgeStore(tmp_path / "client.sqlite") as client:
        apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-1")
        with client.begin("user") as txn:
            txn.create_node("MINE", "subject", {"surface": "我的条目", "intro": "", "category": "common",
                                                "section_order": ["角色"]})
        membership = client.conn.execute(
            "SELECT membership_id FROM membership_versions WHERE valid_to_rev IS NULL"
            " AND parent_id=?", (_local(client, "游戏X"),)).fetchone()[0]
        with client.begin("user") as txn:
            txn.move_membership(membership, parent_id="MINE")
        _move_on_server(service, "Y")
        report = apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-2")
        assert report.moved_memberships == 0 and report.created_memberships == 0
        assert _parents_of_term(client) == ["MINE"]


def test_a_line_merged_out_of_a_retired_entry_stays_put(tmp_path) -> None:
    """A local merge moved the line X -> Y and retired X; the library still has
    it under X. Following that would hang the line under a retired entry,
    where nothing renders it (peer review 2026-10-02)."""

    service = _service(tmp_path)
    with KnowledgeStore(tmp_path / "client.sqlite") as client:
        apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-1")
        x, y = _local(client, "游戏X"), _local(client, "游戏Y")
        membership = client.conn.execute(
            "SELECT membership_id FROM membership_versions WHERE valid_to_rev IS NULL"
            " AND parent_id=?", (x,)).fetchone()[0]
        with client.begin("harness") as txn:
            txn.move_membership(membership, parent_id=y)
            txn.tombstone_node(x)
        report = apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-2")
        assert report.moved_memberships == 0 and report.created_memberships == 0
        assert _parents_of_term(client) == [y]



def test_a_pushed_split_survives_later_pulls(tmp_path) -> None:
    """Split a shared entry, push both entries, pull twice: the lines stay in
    the sub-entry, the parent gets no second copy, and nothing is counted as a
    diverged move -- the library keeping the old place is the add-only push,
    not a move (plan kb-entry-size, contract test)."""

    import json as _json

    from finesub.llm.knowledge.node.proposals import apply_model_proposals
    from finesub.llm.knowledge.node.render import HandleMap
    from finesub.llm.knowledge.node.repo import KnowledgeRepo
    from finesub.llm.knowledge.share.exchange import build_push_bundle, strip_local_maps

    service = _service(tmp_path)
    repo = KnowledgeRepo.open(tmp_path / "client")
    client = repo.store
    apply_snapshot(client, service.snapshot(), remote="srv", task_id="pull-1")
    x = _local(client, "游戏X")
    term = next(n.local_id for n in client.nodes_of_kind("term"))
    with client.begin("user") as txn:  # the user shares the entry
        txn.update_node(x, visibility="shareable")
        txn.update_node(term, visibility="shareable")
    report = apply_model_proposals(
        _json.dumps({"op": "split_entry", "entry": "游戏X", "section": "角色", "intro": "游戏X的角色",
                     "reason": "r"}, ensure_ascii=False),
        repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=HandleMap(),
    )
    assert report.rev is not None and not report.skipped, [r.reason for r in report.skipped]
    child = repo.resolve("游戏X·角色").subject_id

    bundle = build_push_bundle(client, [x, child], idempotency_key="split-1")
    token = service.register()["token"]
    queue_id = service.push(token, strip_local_maps(bundle))["queue_id"]
    item = service.lease("mt")["items"][0]
    reply = service.verdict("mt", queue_id=queue_id, lease_token=item["lease_token"],
                            expected_version=item["verdict_version"], verdict="approve",
                            override="protocol test fixture")
    # what `share status` does once approved: remember the ids the server gave
    for handle, local_id in bundle["handle_map"].items():
        canonical = (reply.get("assigned") or {}).get(handle)
        if canonical:
            client.conn.execute(
                "UPDATE node_versions SET canonical_id=? WHERE local_id=? AND valid_to_rev IS NULL",
                (canonical, local_id),
            )
            client.conn.execute(
                "INSERT OR REPLACE INTO sync_state(remote, canonical_id, local_id, last_server_rev,"
                " last_pulled_payload_hash) VALUES ('srv', ?, ?, NULL, '')",
                (canonical, local_id),
            )
    for task in ("pull-2", "pull-3"):
        report = apply_snapshot(client, service.snapshot(), remote="srv", task_id=task)
        assert report.diverged_moves == 0 and report.moved_memberships == 0
        assert _parents_of_term(client) == [child]
