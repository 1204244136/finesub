"""The hard deadline on an LLM API call holds against a real socket.

An httpx timeout bounds each read, not the request: a server that sends a byte
now and then never trips it. `http.post_within_deadline` is what turns the hard
timeout into a deadline on the whole call, and a fake client cannot show that
it does -- the guarantee lives in what a shut-down socket does to a read that is
blocked in another thread. So most of these drive real requests at a tiny local
server that misbehaves on purpose. Everything is on 127.0.0.1 with an
ephemeral port.

The socket tests are **opt-in** (`pytest --run-network-mock`), like the other
tests that bind sockets (`docs/testing.md`); the two at the top need no socket
and run by default.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time

import httpx
import pytest

from finesub.llm import http as llm_http
from finesub.llm import llm_runtime


class _FailingClient:
    """Stands in for the httpx client: `stream()` raises `error` at once."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def close(self) -> None:
        return None

    def stream(self, *_args, **_kwargs):
        raise self._error


def test_a_read_timeout_at_the_deadline_is_reported_as_the_deadline(monkeypatch) -> None:
    """The per-read timeout and the deadline share one budget, so which fires
    first is scheduling noise -- the exception type must not depend on it."""

    error = httpx.ReadTimeout("timed out")
    monkeypatch.setattr(llm_http, "llm_http_client", lambda **_: _FailingClient(error))

    with pytest.raises(llm_http.DeadlineExceeded):
        llm_http.post_within_deadline("http://x", headers={}, json={}, deadline_seconds=0)


class _Stream:
    """An httpcore network stream as the trace sees it."""

    def __init__(self, sock: object) -> None:
        self.sock = sock
        self.closed = False

    def get_extra_info(self, name: str):
        return self.sock if name == "socket" else None

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("route", ["connection", "proxy", "socks"])
def test_the_cut_reaches_the_tls_socket_not_the_detached_raw_one(monkeypatch, route) -> None:
    """Over HTTPS the TCP socket is detached by `wrap_socket` (fileno -1), so
    shutting it down does nothing -- and on Linux a close alone does not wake
    the blocked read. The TLS socket that replaces it is the one to shut."""

    shut: list[object] = []
    monkeypatch.setattr(llm_http, "_shut", shut.append)
    teardown = llm_http._Teardown(60)
    raw, tls = object(), object()

    teardown.trace(f"{route}.connect_tcp.complete", {"return_value": _Stream(raw)})
    teardown.trace(f"{route}.start_tls.complete", {"return_value": _Stream(tls)})
    teardown.cut()

    assert tls in shut


def test_a_stream_built_after_the_deadline_is_closed_not_leaked() -> None:
    teardown = llm_http._Teardown(0)
    stream = _Stream(None)

    with pytest.raises(llm_http.DeadlineExceeded):
        teardown.trace("connection.connect_tcp.complete", {"return_value": stream})
    assert stream.closed
    # ...and nothing more is sent.
    with pytest.raises(llm_http.DeadlineExceeded):
        teardown.trace("http11.send_request_headers.started", {})


class _BlockingClient:
    """`stream()` blocks until the client is closed -- i.e. until the cut."""

    def __init__(self) -> None:
        self.closed = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def close(self) -> None:
        self.closed.set()

    def stream(self, *_args, **_kwargs):
        self.closed.wait(30)
        raise httpx.ReadError("closed")


def test_ctrl_c_is_not_held_up_by_the_deadline_wait(monkeypatch) -> None:
    """A join with a timeout cannot be interrupted on Windows; waiting in
    slices is what lets Ctrl+C through -- and it must also stop the request."""

    import _thread

    client = _BlockingClient()
    monkeypatch.setattr(llm_http, "llm_http_client", lambda **_: client)
    threading.Timer(0.2, _thread.interrupt_main).start()

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        llm_http.post_within_deadline("http://x", headers={}, json={}, deadline_seconds=30)

    assert time.monotonic() - started < 3
    assert client.closed.is_set(), "the interrupted request is torn down too"


def test_a_genuine_error_before_the_deadline_stays_itself(monkeypatch) -> None:
    """A connect that fails at its own 45 s cap, long before a 46-minute
    deadline, is a connect failure -- not relabelled as the deadline."""

    error = httpx.ConnectTimeout("timed out")
    monkeypatch.setattr(llm_http, "llm_http_client", lambda **_: _FailingClient(error))

    with pytest.raises(httpx.ConnectTimeout) as raised:
        llm_http.post_within_deadline("http://x", headers={}, json={}, deadline_seconds=30)
    assert not isinstance(raised.value, llm_http.DeadlineExceeded)


GEMINI_OK = {
    "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
}


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, behaviour: str) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.behaviour = behaviour
        self.requests = 0
        self.stop = threading.Event()
        self.client_gone = threading.Event()

    @property
    def origin(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def handle_error(self, request, client_address) -> None:
        # The client hanging up mid-response is the point of these tests.
        return


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - http.server's naming
        server: _Server = self.server  # type: ignore[assignment]
        server.requests += 1
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        getattr(self, f"_{server.behaviour}")(server)

    def _trickle(self, server: _Server) -> None:
        """Headers at once, then one byte of body every 50 ms, forever."""

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while not server.stop.is_set():
                self.wfile.write(b"1\r\n \r\n")
                self.wfile.flush()
                server.stop.wait(0.05)
        except OSError:
            server.client_gone.set()

    def _silent(self, server: _Server) -> None:
        """Take the request and never answer."""

        server.stop.wait(10)

    def _late(self, server: _Server) -> None:
        """Answer in full, but only after 0.3 s."""

        server.stop.wait(0.3)
        body = json.dumps(GEMINI_OK).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        return


@pytest.fixture
def serve(monkeypatch):
    # No proxy, whatever the machine or the checkout's config.toml says.
    monkeypatch.setattr(llm_http, "configured_proxy", lambda: llm_http.DIRECT)
    servers: list[_Server] = []

    def start(behaviour: str) -> _Server:
        server = _Server(behaviour)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.stop.set()
        server.shutdown()
        server.server_close()


@pytest.mark.network_mock
def test_a_trickling_response_is_cut_at_the_deadline(serve) -> None:
    server = serve("trickle")

    started = time.monotonic()
    with pytest.raises(llm_http.DeadlineExceeded):
        llm_http.post_within_deadline(
            server.origin, headers={}, json={}, deadline_seconds=0.4
        )
    elapsed = time.monotonic() - started

    # Before the fix this returned whenever the server chose to stop.
    assert 0.35 <= elapsed < 1.5
    # Torn down, not abandoned: the server's next write finds the peer gone.
    assert server.client_gone.wait(2.0)


@pytest.mark.network_mock
def test_a_silent_server_is_cut_at_the_deadline(serve) -> None:
    server = serve("silent")

    started = time.monotonic()
    with pytest.raises(llm_http.DeadlineExceeded):
        llm_http.post_within_deadline(
            server.origin, headers={}, json={}, deadline_seconds=0.4
        )
    assert time.monotonic() - started < 1.5


@pytest.mark.network_mock
def test_a_deadline_during_a_slow_lookup_sends_nothing(serve, monkeypatch) -> None:
    """The caller is back on time, and the request never leaves afterwards.

    A DNS lookup cannot be interrupted, so the worker is still in it when the
    deadline passes; once it comes back it must not connect-and-POST anyway --
    that would start a generation nobody is waiting for.
    """

    server = serve("silent")
    real_getaddrinfo = socket.getaddrinfo

    def slow_getaddrinfo(*args, **kwargs):
        time.sleep(0.4)
        return real_getaddrinfo(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", slow_getaddrinfo)

    started = time.monotonic()
    with pytest.raises(llm_http.DeadlineExceeded):
        llm_http.post_within_deadline(
            server.origin, headers={}, json={}, deadline_seconds=0.1
        )
    assert time.monotonic() - started < 0.35, "the caller does not wait out the lookup"

    time.sleep(0.8)  # well past the lookup finishing
    assert server.requests == 0


def _gemini_at(monkeypatch, server: _Server, *, soft: float, hard: float) -> None:
    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(
        llm_runtime,
        "_read_dotenv",
        lambda: {"GEMINI_FREE": "{free-main:key1}", "GEMINI_BASE_URL": server.origin},
    )
    monkeypatch.setattr(
        llm_runtime, "api_timeouts", lambda max_tokens: llm_runtime.ApiTimeouts(soft, hard)
    )


@pytest.mark.network_mock
def test_two_hard_deadlines_take_the_existing_timeout_abort(serve, monkeypatch) -> None:
    """A cut call is a timeout to everything downstream of it."""

    server = serve("silent")
    _gemini_at(monkeypatch, server, soft=0.1, hard=0.3)
    monkeypatch.setattr(llm_runtime.time, "sleep", lambda _: None)

    with pytest.raises(llm_http.DeadlineExceeded) as raised:
        llm_runtime.chat_complete(
            [{"role": "user", "content": "hi"}],
            provider_tier="GEMINI_FREE",
            model="gemini/gemini-3.1-flash-lite",
            retries=7,
        )

    assert server.requests == 2
    assert getattr(raised.value, "_harness_consecutive_timeout_abort") is True
    attempts = getattr(raised.value, "_harness_api_attempts")
    assert [item["return_code"] for item in attempts] == ["NO_RESPONSE_TIMEOUT"] * 2


@pytest.mark.network_mock
def test_an_answer_between_the_deadlines_is_used(serve, monkeypatch) -> None:
    from finesub.reporting import NullReporter, reporting_to

    warnings: list[str] = []

    class _Warnings(NullReporter):
        def warning(self, code, message, *, impact="", action="") -> None:
            warnings.append(code)

    server = serve("late")
    _gemini_at(monkeypatch, server, soft=0.1, hard=3.0)

    with reporting_to(_Warnings()):
        response = llm_runtime.chat_complete(
            [{"role": "user", "content": "hi"}],
            provider_tier="GEMINI_FREE",
            model="gemini/gemini-3.1-flash-lite",
            retries=0,
        )

    assert warnings == ["llm-api-soft-timeout", "llm-api-late-response"]
    (attempt,) = response["_harness_api_attempts"]
    assert attempt["return_code"] == "200"
    assert attempt["soft_timeout_overrun_sec"] > 0
    assert llm_runtime.extract_message_content(response) == "ok"
