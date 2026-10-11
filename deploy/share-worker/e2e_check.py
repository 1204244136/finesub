"""End-to-end check for the mailbox deployment, over real HTTP.

The Python unit tests stub the transport, and the Worker's own logic is JS —
so nothing in `python -m pytest` ever runs the two halves against each other.
This script does, against a Worker you point it at:

    # terminal 1
    cd deploy/share-worker
    printf 'MAINTAINER_TOKEN=local-mailbox-token\n' > .dev.vars
    npx wrangler dev --port 8810 --ip 127.0.0.1

    # terminal 2 (repo root)
    python deploy/share-worker/e2e_check.py --remote http://127.0.0.1:8810 \
        --mailbox-token local-mailbox-token

It uses a throwaway local corpus, so it is for a dev Worker and **not** for the
real one: it publishes, and an empty library over a live snapshot leaves every
existing client unable to pull. That is refused rather than trusted to the
reader — a Worker already serving a different corpus stops the run. Pass
`--server-root` to re-run against one this script published to before.

Exits non-zero on the first surprise.

⚠ Run `wrangler dev` from a SHORT path: miniflare's SQLite fails to start under
a deep directory, and the only symptom is a detail-free `internal error`
(`tools/cf_share_poc/README.md` has the full note).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from finesub.llm.knowledge.node.store import KnowledgeStore  # noqa: E402
from finesub.llm.knowledge.share import client  # noqa: E402
from finesub.llm.knowledge.share.exchange import (  # noqa: E402
    build_push_bundle,
    strip_local_maps,
    verify_snapshot,
)
from finesub.llm.knowledge.share.relay import run_relay  # noqa: E402
from finesub.llm.knowledge.share.server import ShareService, contributor_id  # noqa: E402

LOCAL_TOKEN = "e2e-local-service-token"
failures: list[str] = []


def say(name: str, value: object) -> None:
    print(f"  {name}: {value}", flush=True)


def expect(name: str, ok: bool, value: object) -> None:
    say(name, value)
    if not ok:
        failures.append(name)


def seed(store: KnowledgeStore) -> None:
    with store.begin("import") as txn:
        txn.create_node(
            "S",
            "subject",
            {"surface": "原神", "intro": "游戏", "category": "common",
             "native_names": ["原神"], "section_order": ["角色"]},
            visibility="shareable",
        )
        txn.create_node(
            "T", "term",
            {"surface": "スカラマシュ", "zh": "散兵", "alias_text": "", "reading": "", "desc": "角色"},
            visibility="shareable",
        )
        txn.create_membership("M", "S", "T", "角色", 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mailbox-token")
    source.add_argument(
        "--mailbox-token-file",
        help="read it from a file instead, so the secret stays out of shell history",
    )
    parser.add_argument(
        "--server-root",
        default=None,
        help="reuse this local store (and so this corpus) instead of a throwaway one;"
        " required to re-run against a Worker this script already published to",
    )
    args = parser.parse_args(argv)
    remote = args.remote.rstrip("/")
    mailbox = (
        Path(args.mailbox_token_file).read_text(encoding="utf-8").strip()
        if args.mailbox_token_file
        else args.mailbox_token
    )

    tmp = Path(tempfile.mkdtemp(prefix="share-e2e-"))
    root = Path(args.server_root).expanduser() if args.server_root else tmp / "srv"
    service = ShareService(root, maintainer_token=LOCAL_TOKEN)
    print(f"local corpus {service.corpus_id} in {root}", flush=True)
    try:
        return _run(service, remote, mailbox, tmp)
    finally:
        service.close()


def _run(service: ShareService, remote: str, mailbox: str, tmp: Path) -> int:
    try:
        already = client.fetch_identity(remote)
    except client.RemoteError as exc:
        expect("identity before any publish", "503" in str(exc), f"refused ({str(exc)[:60]})")
    else:
        # A Worker already serving somebody else's corpus is the one situation
        # this script must not proceed from: step one is a relay with
        # publish=True, which would put this throwaway library's empty snapshot
        # — new corpus id, chain restarting at rev 1 — over theirs. Every
        # existing client would then hit the impostor gate or the anti-rollback
        # refusal and be unable to pull at all until the maintainer republishes.
        # The docstring's "not against the real one" is enforced here rather
        # than merely stated (reviewer, 2026-09-09).
        if already.get("corpus_id") != service.corpus_id:
            print(
                f"refusing: {remote} already serves corpus {str(already.get('corpus_id'))[:12]}…,"
                " and this run would publish an empty library over it."
                "\n  Point --server-root at the store that owns that corpus to re-run against it.",
                flush=True,
            )
            return 1
        say("identity before any publish", "skipped — this Worker already serves THIS corpus")

    say("relay #1 (publish only)", run_relay(service, remote, maintainer_token=mailbox).lines())
    identity = client.fetch_identity(remote)
    expect(
        "identity after publish",
        identity["corpus_id"] == service.corpus_id,
        identity,
    )

    try:
        token = client.register(remote)
    except client.RemoteError as exc:
        if "today" in str(exc):
            print(
                f"refusing: {exc}"
                "\n  The daily registration window is per Worker and survives restarts"
                " (miniflare keeps state in .wrangler/). Delete that directory, or use"
                " another Worker.",
                flush=True,
            )
            return 1
        raise
    say("contributor token", token[:12] + "…")
    with KnowledgeStore(tmp / "client.sqlite") as store:
        seed(store)
        # Unique per run: the key is the mailbox's dedupe identity, so a
        # fixed one makes the second run against the same Worker a 409
        # ("already used by a different contributor") instead of a check.
        bundle = strip_local_maps(
            build_push_bundle(store, ["S"], idempotency_key=f"e2e-{uuid.uuid4().hex[:12]}")
        )
    first = client.push_bundle(remote, bundle, token=token)
    queue_id = first["queue_id"]
    expect("push", first.get("status") == "pending", first)
    retry = client.push_bundle(remote, bundle, token=token)
    expect(
        "retry reuses the queue item",
        retry["queue_id"] == queue_id and retry.get("duplicate"),
        retry,
    )

    say("relay #2 (collect)", run_relay(service, remote, maintainer_token=mailbox).lines())
    queued = service.peek(LOCAL_TOKEN)["items"]
    # The mailbox relays the token's hash, never the token; the local queue
    # keeps that id as is, which is what `share status` and the digest tally
    # agree on.
    expect(
        "the local queue got it, under the contributor id the mailbox relayed",
        len(queued) == 1 and queued[0]["contributor"] == contributor_id(token),
        [(item["queue_id"], item["contributor"][:12] + "…") for item in queued],
    )

    item = service.lease(LOCAL_TOKEN)["items"][0]
    service.verdict(
        LOCAL_TOKEN,
        queue_id=item["queue_id"],
        lease_token=item["lease_token"],
        expected_version=item["verdict_version"],
        verdict="approve",
        override="e2e check",
    )
    say("relay #3 (carry the verdict back)", run_relay(service, remote, maintainer_token=mailbox).lines())
    status = client.push_status(remote, queue_id, token=token)
    expect(
        "the contributor sees the approval and the canonical ids",
        status["status"] == "approved" and bool(status["assigned"]),
        {"status": status["status"], "assigned": len(status["assigned"])},
    )

    snapshot = client.fetch_snapshot(remote)
    verify_snapshot(snapshot)  # raises if the chain does not hold
    expect(
        "the published snapshot verifies and carries the approved nodes",
        snapshot["corpus_id"] == service.corpus_id and len(snapshot["content"]["nodes"]) >= 2,
        {"server_rev": snapshot["server_rev"], "nodes": len(snapshot["content"]["nodes"])},
    )

    # The conditional pull, over real HTTP. Both halves of it — the Worker's
    # ETag comparison and the client's 304 branch — exist only for each other,
    # and the Python suite stubs `fetch_snapshot` wholesale, so this is the one
    # place where a mismatch between the two strings can ever turn red.
    current = client.fetch_snapshot(remote, known_chain_hash=snapshot["chain_hash"])
    expect(
        "a client that is already current gets 304, not the corpus",
        current is client.UNCHANGED,
        "UNCHANGED" if current is client.UNCHANGED else type(current).__name__,
    )
    stale = client.fetch_snapshot(remote, known_chain_hash="not-the-head")
    expect(
        "...and a stale tag still gets the whole thing",
        isinstance(stale, dict) and stale.get("chain_hash") == snapshot["chain_hash"],
        type(stale).__name__,
    )

    try:
        client.push_status(remote, queue_id, token="not-the-pusher")
        expect("a stranger cannot read that queue item", False, "it answered")
    except client.RemoteError as exc:
        expect("a stranger cannot read that queue item", "403" in str(exc), f"refused ({str(exc)[:40]})")
    try:
        client.fetch_inbox(remote, maintainer_token="wrong")
        expect("the inbox needs the maintainer token", False, "it answered")
    except client.RemoteError as exc:
        expect("the inbox needs the maintainer token", "403" in str(exc), f"refused ({str(exc)[:40]})")

    if failures:
        print("\nFAILED: " + ", ".join(failures), flush=True)
        return 1
    print("\nOK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
