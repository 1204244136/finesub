"""Snapshot signatures (security assessment 2026-09-25, F2): the hash chain
proves the server said something, the signature proves the maintainer did.
Whoever holds the mailbox token can extend the chain; only the key holder can
sign the new head."""

from __future__ import annotations

import copy

import pytest

from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.share import cli as share_cli
from finesub.llm.knowledge.share.exchange import ExchangeError, chain_hash, content_digest
from finesub.llm.knowledge.share.server import ShareService
from finesub.llm.knowledge.share.signing import generate_key_file, load_private_key
from finesub.llm.knowledge.share.sync import apply_snapshot, signing_key_meta


@pytest.fixture
def key(tmp_path):  # type: ignore[no-untyped-def]
    public = generate_key_file(tmp_path / "keys" / "maintainer.key")
    return load_private_key(tmp_path / "keys" / "maintainer.key"), public


def _snapshot(tmp_path, name: str, seed: bytes | None):  # type: ignore[no-untyped-def]
    service = ShareService(tmp_path / name, maintainer_token="m", signing_key=seed)
    try:
        return service.snapshot()
    finally:
        service.close()


def _client(tmp_path) -> KnowledgeRepo:  # type: ignore[no-untyped-def]
    return KnowledgeRepo.open(tmp_path / "client")


def _extended(snapshot: dict) -> dict:
    """What the mailbox token holder can do: append a rev to the chain with
    new content, keeping the old signature (they cannot make a new one)."""

    forged = copy.deepcopy(snapshot)
    forged["content"]["nodes"] = [{
        "canonical_id": "evil", "kind": "subject", "retired": False, "maturity": "normal",
        "payload": {"surface": "伪条目", "intro": "x", "category": "common"},
    }]
    digest_value = content_digest(forged["content"])
    rev = forged["server_rev"] + 1
    link = chain_hash(forged["chain_hash"], digest_value)
    forged["history"].append({"rev": rev, "content_digest": digest_value, "chain_hash": link})
    forged.update(server_rev=rev, content_digest=digest_value, chain_hash=link)
    return forged


def test_keygen_never_overwrites_a_key(tmp_path, key) -> None:
    with pytest.raises(ExchangeError, match="never overwritten"):
        generate_key_file(tmp_path / "keys" / "maintainer.key")


def test_the_first_signed_pull_pins_the_key(tmp_path, key) -> None:
    seed, public = key
    snapshot = _snapshot(tmp_path, "server", seed)
    assert snapshot["signature"]["public_key"] == public
    repo = _client(tmp_path)
    apply_snapshot(repo.store, snapshot, remote=snapshot["corpus_id"])
    assert repo.store.meta(signing_key_meta(snapshot["corpus_id"])) == public


def test_a_token_holder_cannot_extend_a_signed_chain(tmp_path, key) -> None:
    seed, _public = key
    snapshot = _snapshot(tmp_path, "server", seed)
    repo = _client(tmp_path)
    apply_snapshot(repo.store, snapshot, remote=snapshot["corpus_id"])
    with pytest.raises(ExchangeError, match="does not verify"):
        apply_snapshot(repo.store, _extended(snapshot), remote=snapshot["corpus_id"])


def test_a_pinned_library_refuses_unsigned_and_foreign_keys(tmp_path, key) -> None:
    seed, _public = key
    snapshot = _snapshot(tmp_path, "server", seed)
    corpus = snapshot["corpus_id"]
    repo = _client(tmp_path)
    apply_snapshot(repo.store, snapshot, remote=corpus)

    unsigned = _extended(snapshot)
    unsigned.pop("signature")
    with pytest.raises(ExchangeError, match="unsigned"):
        apply_snapshot(repo.store, unsigned, remote=corpus)

    from finesub.llm.knowledge.share.signing import sign_snapshot

    generate_key_file(tmp_path / "other.key")
    other_seed = load_private_key(tmp_path / "other.key")

    resigned = _extended(snapshot)
    resigned["signature"] = sign_snapshot(resigned, other_seed)
    with pytest.raises(ExchangeError, match="different key"):
        apply_snapshot(repo.store, resigned, remote=corpus)
    # a real rotation, said out loud, re-pins
    apply_snapshot(repo.store, resigned, remote=corpus, accept_new_key=True)
    assert repo.store.meta(signing_key_meta(corpus)) == resigned["signature"]["public_key"]


def test_a_key_known_in_advance_admits_nothing_else(tmp_path, key) -> None:
    _seed, public = key
    unsigned = _snapshot(tmp_path, "server", None)
    repo = _client(tmp_path)
    with pytest.raises(ExchangeError, match="not signed"):
        apply_snapshot(repo.store, unsigned, remote=unsigned["corpus_id"], trusted_key=public)


def test_a_library_that_never_signed_stays_pullable(tmp_path) -> None:
    snapshot = _snapshot(tmp_path, "server", None)
    repo = _client(tmp_path)
    apply_snapshot(repo.store, snapshot, remote=snapshot["corpus_id"])
    assert repo.store.meta(signing_key_meta(snapshot["corpus_id"])) is None


def test_the_built_in_mailbox_trusts_only_the_built_in_key(monkeypatch) -> None:
    """The packaged key is a usable Ed25519 public key, and it -- not a first
    sighting -- is what a pull from the built-in address must be signed by."""

    from cryptography.hazmat.primitives.asymmetric import ed25519

    from finesub.llm.knowledge.share import mailbox
    from finesub.llm.knowledge.share.signing import _b64decode

    ed25519.Ed25519PublicKey.from_public_bytes(_b64decode(mailbox.BUILTIN_SIGNING_KEY))
    from types import SimpleNamespace

    # No `[share] signing_key` in whatever config.toml this machine has.
    unset = SimpleNamespace(SHARE_SIGNING_KEY=SimpleNamespace(configured=lambda: None))
    monkeypatch.setattr(mailbox, "settings", unset)
    assert mailbox.trusted_signing_key(mailbox.builtin_mailbox()) == mailbox.BUILTIN_SIGNING_KEY
    assert mailbox.trusted_signing_key("https://someone-else.example") == ""
    # A configured key names the user's own library; the built-in one keeps its key.
    configured = SimpleNamespace(SHARE_SIGNING_KEY=SimpleNamespace(configured=lambda: "their-key"))
    monkeypatch.setattr(mailbox, "settings", configured)
    assert mailbox.trusted_signing_key(mailbox.builtin_mailbox()) == mailbox.BUILTIN_SIGNING_KEY
    assert mailbox.trusted_signing_key("https://someone-else.example") == "their-key"


def test_relay_will_not_publish_without_the_key(tmp_path) -> None:
    args = share_cli.parse_args([
        "relay", "--remote", "https://mailbox.example", "--server-root", str(tmp_path / "srv"),
        "--mailbox-token", "mbx",
    ])
    with pytest.raises(SystemExit, match="signing-key-file"):
        share_cli._cmd_relay(args)
