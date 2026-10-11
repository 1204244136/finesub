"""Pull admission that a hostile snapshot cannot crash (F7), and a transport
that does not hand tokens to the wrong host (F8) -- security assessment
2026-09-25."""

from __future__ import annotations

import urllib.request

import pytest

from finesub.llm.knowledge.share import client as share_client
from finesub.llm.knowledge.share.exchange import (
    SNAPSHOT_SCHEMA_VERSION,
    has_cycle,
    snapshot_content_problems,
)


def _content(**parts):  # type: ignore[no-untyped-def]
    return {"schema": SNAPSHOT_SCHEMA_VERSION, "nodes": [], "items": [], "memberships": [],
            "links": [], "redirects": {}, **parts}


def test_a_ten_thousand_deep_membership_chain_is_admitted_without_recursion() -> None:
    depth = 10_000
    nodes = [{"canonical_id": f"n{i}", "kind": "term", "retired": False, "maturity": "normal",
              "payload": {"surface": f"t{i}", "desc": ""}} for i in range(depth)]
    chain = [{"canonical_id": f"m{i}", "parent": f"n{i}", "child": f"n{i + 1}", "section": "s",
              "order_key": 0, "retired": False} for i in range(depth - 1)]
    assert "memberships contain a cycle" not in snapshot_content_problems(
        _content(nodes=nodes, memberships=chain)
    )
    closing = {"canonical_id": "mx", "parent": f"n{depth - 1}", "child": "n0", "section": "s",
               "order_key": 0, "retired": False}
    assert "memberships contain a cycle" in snapshot_content_problems(
        _content(nodes=nodes, memberships=[*chain, closing])
    )


def test_has_cycle() -> None:
    assert not has_cycle({"a": ["b", "c"], "b": ["c"]})
    assert has_cycle({"a": ["b"], "b": ["c"], "c": ["a"]})
    assert not has_cycle({})


def test_redirect_cycles_are_found_in_one_walk() -> None:
    long_chain = {f"r{i}": f"r{i + 1}" for i in range(20_000)}
    assert not any("redirects" in p for p in snapshot_content_problems(_content(redirects=long_chain)))
    looped = {**long_chain, "r20000": "r0"}
    problems = [p for p in snapshot_content_problems(_content(redirects=looped)) if "redirects" in p]
    assert len(problems) == 1


@pytest.mark.parametrize("url", ["http://share.example/identity", "ftp://share.example/identity"])
def test_a_remote_must_be_https(url: str) -> None:
    with pytest.raises(share_client.RemoteError, match="https"):
        share_client._request(url)


def test_plain_http_is_fine_on_this_machine(monkeypatch) -> None:
    class _Response:
        def read(self, _n: int | None = None) -> bytes:
            return b"{}"

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    monkeypatch.setattr(share_client, "_urlopen", lambda *a, **k: _Response())
    for host in ("127.0.0.1:8787", "localhost:8787", "[::1]:8787"):
        assert share_client._request(f"http://{host}/identity") == {}


def test_a_redirect_is_refused_not_followed() -> None:
    handler = share_client._NoRedirects()
    request = urllib.request.Request("https://share.example/push")
    request.add_header("X-Share-Token", "secret")
    with pytest.raises(share_client.RemoteError, match="redirected"):
        handler.redirect_request(request, None, 302, "Found", {}, "https://elsewhere.example/")
