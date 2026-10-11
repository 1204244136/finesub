"""The Codex user's connection settings carried past `--ignore-user-config`."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import tomllib

import pytest

from finesub.llm.agent import codex_connection
from finesub.llm.agent.codex_connection import (
    PROVIDER_TOKEN_ENV,
    CodexConnection,
    CodexConnectionError,
    check_ready,
    connection_from_config,
    local_endpoint,
    read_connection,
    toml_literal,
)
from finesub.llm.agent.local_agent import (
    CodexDriverConfig,
    CodexLocalAgentDriver,
    LocalAgentUnavailableError,
)

from .test_llm_local_agent import _driver

PROXY_CONFIG = """
model_provider = "cliproxyapi"
model = "gpt-6.1-sol"
notify = ["notifier.exe", "turn-ended"]

[mcp_servers.node_repl]
command = "node_repl.exe"

[model_providers.cliproxyapi]
name = "OpenAI"
base_url = "http://127.0.0.1:{port}/v1"
experimental_bearer_token = "sk-proxy-secret"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = true
"""


def _write_config(monkeypatch, tmp_path: Path, text: str) -> Path:
    home = tmp_path / "codex-home"
    home.mkdir(exist_ok=True)
    (home / "config.toml").write_text(text, encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(home))
    return home


@pytest.fixture
def listening_port():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    try:
        yield server.getsockname()[1]
    finally:
        server.close()


@pytest.fixture
def closed_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _overrides(connection: CodexConnection) -> dict[str, object]:
    """Parse the `-c` values back, the way Codex reads them."""

    parsed: dict[str, object] = {}
    for override in connection.overrides:
        key, _, literal = override.partition("=")
        parsed[key] = tomllib.loads(f"v = {literal}")["v"]
    return parsed


# -- reading the config --------------------------------------------------------


def test_missing_config_carries_nothing(tmp_path: Path) -> None:
    assert read_connection(tmp_path / "nowhere") == CodexConnection()


def test_unparsable_config_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("model_provider = [", encoding="utf-8")
    with pytest.raises(CodexConnectionError, match="not valid TOML"):
        read_connection(tmp_path)


def test_only_connection_keys_and_the_named_provider_are_carried() -> None:
    document = tomllib.loads(PROXY_CONFIG.format(port=8317))
    document["model_providers"]["other"] = {"name": "Other", "base_url": "http://x"}

    connection = connection_from_config(document)
    carried = _overrides(connection)

    assert set(carried) == {"model_provider", "model_providers.cliproxyapi"}
    assert carried["model_provider"] == "cliproxyapi"
    assert connection.keys == ("model_provider",)
    assert connection.provider == "cliproxyapi"
    assert connection.base_url == "http://127.0.0.1:8317/v1"


def test_builtin_provider_carries_no_table() -> None:
    connection = connection_from_config(
        {"model_provider": "openai", "openai_base_url": "http://localhost:9000/v1"}
    )

    assert _overrides(connection) == {
        "model_provider": "openai",
        "openai_base_url": "http://localhost:9000/v1",
    }
    assert connection.base_url == "http://localhost:9000/v1"


def test_unset_keys_are_not_passed() -> None:
    assert connection_from_config({"model": "gpt-6"}).overrides == ()


def test_legacy_profile_selector_is_not_followed() -> None:
    """Codex 0.157.1 no longer reads `profile = "..."`; neither does FineSub."""

    document = {
        "model_provider": "openai",
        "profile": "work",
        "profiles": {"work": {"model_provider": "corp"}},
    }

    assert _overrides(connection_from_config(document)) == {"model_provider": "openai"}
    assert connection_from_config({"profile": "missing"}).overrides == ()


def test_a_provider_without_a_table_is_left_to_codex() -> None:
    """A name Codex defines itself (or adds later): carried as is, Codex judges it."""

    connection = connection_from_config({"model_provider": "ghost"})
    assert _overrides(connection) == {"model_provider": "ghost"}
    assert connection.base_url == ""


# -- secrets -----------------------------------------------------------------


def test_bearer_token_moves_to_the_environment() -> None:
    connection = connection_from_config(tomllib.loads(PROXY_CONFIG.format(port=8317)))
    table = _overrides(connection)["model_providers.cliproxyapi"]

    assert "experimental_bearer_token" not in table
    assert table["env_key"] == PROVIDER_TOKEN_ENV
    assert connection.env == {PROVIDER_TOKEN_ENV: "sk-proxy-secret"}
    assert all("sk-proxy-secret" not in override for override in connection.overrides)


def test_env_key_and_header_variables_pass_through() -> None:
    connection = connection_from_config({
        "model_provider": "corp",
        "model_providers": {"corp": {
            "base_url": "https://corp.example/v1",
            "env_key": "CORP_KEY",
            "env_http_headers": {"X-Org": "CORP_ORG"},
        }},
    })

    assert connection.passthrough_env == ("CORP_KEY", "CORP_ORG")
    assert connection.env == {}


@pytest.mark.parametrize("literal", ["http_headers", "query_params"])
def test_literal_headers_and_query_params_are_refused(literal: str) -> None:
    with pytest.raises(CodexConnectionError, match="env_http_headers"):
        connection_from_config({
            "model_provider": "corp",
            "model_providers": {"corp": {"base_url": "https://x", literal: {"k": "v"}}},
        })


def test_token_and_env_key_together_are_refused() -> None:
    with pytest.raises(CodexConnectionError, match="keep one"):
        connection_from_config({
            "model_provider": "corp",
            "model_providers": {"corp": {"experimental_bearer_token": "t", "env_key": "K"}},
        })


# -- TOML literals -------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        'C:\\Users\\a "quoted" path\n\ttab\x01',
        True,
        False,
        42,
        1.5,
        ["a", 'b"c'],
        {"plain": "x", "needs quoting": 1, "nested": {"k": ["v"]}},
    ],
)
def test_literals_round_trip(value: object) -> None:
    assert tomllib.loads(f"v = {toml_literal(value)}")["v"] == value


@pytest.mark.parametrize("value", [[1, 2], float("inf"), None, object()])
def test_unsupported_values_are_refused(value: object) -> None:
    with pytest.raises(CodexConnectionError):
        toml_literal(value)


# -- pre-flight ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:8317/v1", ("127.0.0.1", 8317)),
        ("http://127.5.6.7:9/v1", ("127.5.6.7", 9)),
        ("http://localhost/v1", ("localhost", 80)),
        ("https://localhost/v1", ("localhost", 443)),
        ("http://[::1]:8000/v1", ("::1", 8000)),
        ("https://api.example.com/v1", None),
        ("http://192.168.1.2:8317/v1", None),
        ("", None),
    ],
)
def test_local_endpoint(url: str, expected) -> None:
    assert local_endpoint(url) == expected


def test_ready_when_the_local_provider_answers(listening_port: int) -> None:
    connection = connection_from_config(tomllib.loads(PROXY_CONFIG.format(port=listening_port)))
    check_ready(connection, {PROVIDER_TOKEN_ENV: "sk-proxy-secret"})


def test_local_provider_not_answering_is_refused(closed_port: int) -> None:
    connection = connection_from_config(tomllib.loads(PROXY_CONFIG.format(port=closed_port)))
    with pytest.raises(CodexConnectionError, match="not answering"):
        check_ready(connection, {PROVIDER_TOKEN_ENV: "sk-proxy-secret"})


def test_connect_goes_through_create_connection(monkeypatch) -> None:
    seen = []

    class _Connected:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_connect(address, timeout):
        seen.append((address, timeout))
        return _Connected()

    monkeypatch.setattr(codex_connection.socket, "create_connection", fake_connect)
    check_ready(CodexConnection(base_url="http://localhost/v1"), {})
    check_ready(CodexConnection(base_url="https://api.example.com/v1"), {})

    assert seen == [(("localhost", 80), codex_connection.LOCAL_CONNECT_TIMEOUT_SECONDS)]


def test_missing_provider_variable_is_refused() -> None:
    connection = CodexConnection(passthrough_env=("CORP_KEY",))
    with pytest.raises(CodexConnectionError, match="CORP_KEY"):
        check_ready(connection, {"CORP_KEY": ""})


# -- the driver ----------------------------------------------------------------


def _run_envdump(driver: CodexLocalAgentDriver) -> dict:
    result = driver.run([{"role": "user", "content": "envdump"}], task="general")
    return json.loads(result.content)


def test_driver_carries_the_proxy_and_keeps_the_token_off_argv(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000)

    dump = _run_envdump(driver)
    argv = dump["argv"]

    assert "--ignore-user-config" in argv
    values = [argv[i + 1] for i, item in enumerate(argv) if item == "--config"]
    assert 'model_provider="cliproxyapi"' in values
    assert any(value.startswith("model_providers.cliproxyapi=") for value in values)
    assert all("sk-proxy-secret" not in item for item in argv)
    assert dump["env"][PROVIDER_TOKEN_ENV] == "sk-proxy-secret"
    # Behaviour stays isolated: nothing else from the file is carried.
    assert not any(value.startswith(("model=", "notify", "mcp_servers.node_repl")) for value in values)


def test_driver_refuses_before_spawning_when_the_proxy_is_down(
    tmp_path: Path, monkeypatch, closed_port: int
) -> None:
    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=closed_port))
    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentUnavailableError, match="not answering") as caught:
        driver.run([{"role": "user", "content": "ok"}], task="general")
    attempts = getattr(caught.value, "_harness_execution_attempts")
    assert attempts[0]["return_code"] is None


def test_driver_refuses_a_provider_that_names_a_finesub_credential(
    tmp_path: Path, monkeypatch
) -> None:
    _write_config(monkeypatch, tmp_path, """
model_provider = "corp"
[model_providers.corp]
base_url = "https://corp.example/v1"
env_key = "OPENAI_API_KEY"
""")
    monkeypatch.setenv("OPENAI_API_KEY", "finesub-own-key")
    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentUnavailableError, match="OPENAI_API_KEY.*a variable of its own"):
        driver.run([{"role": "user", "content": "ok"}], task="general")


def test_driver_passes_the_providers_own_variable(tmp_path: Path, monkeypatch) -> None:
    _write_config(monkeypatch, tmp_path, """
model_provider = "corp"
[model_providers.corp]
base_url = "https://corp.example/v1"
env_key = "CORP_KEY"
""")
    monkeypatch.setenv("CORP_KEY", "corp-secret")
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000)

    assert _run_envdump(driver)["env"]["CORP_KEY"] == "corp-secret"


def test_driver_reports_a_broken_config_as_unavailable(tmp_path: Path, monkeypatch) -> None:
    _write_config(monkeypatch, tmp_path, "model_provider = [")
    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentUnavailableError, match="not valid TOML"):
        driver.run([{"role": "user", "content": "ok"}], task="general")


def test_driver_carries_regardless_of_the_unisolated_opt_in(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    template = _driver(tmp_path, monkeypatch)
    opted_in = CodexLocalAgentDriver(CodexDriverConfig(
        command=template.config.command,
        runtime_root=tmp_path / "runtime-opt-in",
        timeout_seconds=5,
        max_result_bytes=1_000_000,
        allow_unisolated_user_config=True,
    ))

    argv = _run_envdump(opted_in)["argv"]

    assert "--ignore-user-config" in argv
    assert 'model_provider="cliproxyapi"' in argv


def test_isolation_record_names_the_provider_never_the_token(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = _driver(tmp_path, monkeypatch)

    result = driver.run([{"role": "user", "content": "ok"}], task="general")
    isolation = result.execution_attempt["isolation"]

    assert isolation["user_config"] == "ignored_except_connection"
    assert isolation["user_connection"] == {
        "carried_keys": ["model_provider"],
        "model_provider": "cliproxyapi",
        "base_url": f"http://127.0.0.1:{listening_port}/v1",
    }
    assert isolation["execpolicy_rules"] == "ignored"
    assert isolation["global_instructions"] == "injected"
    assert "sk-proxy-secret" not in json.dumps(isolation)
    assert "user_rules" not in isolation


def test_proxy_and_config_dir_variables_reach_the_child(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("OPENAI_API_KEY", "finesub-own-key")
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000)

    env = {key.upper(): value for key, value in _run_envdump(driver)["env"].items()}

    assert env["HTTPS_PROXY"] == "http://proxy.example:3128"
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "claude")
    assert "OPENAI_API_KEY" not in env
    assert os.environ["OPENAI_API_KEY"] == "finesub-own-key"


def test_resume_carries_the_connection_too(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    from finesub.llm.agent.local_agent import DriverProbe

    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = _driver(tmp_path, monkeypatch)
    driver._resolved_command = ("codex",)
    probe = DriverProbe(available=True, no_user_config=True, no_user_rules=True)
    driver._call_local.connection = read_connection()
    driver._call_local.error = None
    capsule = type("Capsule", (), {"root": tmp_path})()

    argv = driver._argv(
        capsule, native_search=False, probe=probe,
        session_scope="assignment", conversation_handle="thread-1",
    )

    assert argv[argv.index("exec") + 1] == "resume"
    assert 'model_provider="cliproxyapi"' in argv
    assert argv.index('model_provider="cliproxyapi"') < argv.index("exec")


def test_concurrent_calls_keep_their_own_connection(tmp_path: Path, monkeypatch) -> None:
    """One driver, two threads, two configs: neither argv borrows the other's."""

    import threading

    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, max_parallel=2)
    connections = {
        "a": CodexConnection(overrides=('model_provider="alpha"',)),
        "b": CodexConnection(overrides=('model_provider="beta"',)),
    }
    gate = threading.Barrier(2)

    def read_for_thread(home=None):
        gate.wait(timeout=10)  # both calls hold their config at once
        return connections[threading.current_thread().name]

    monkeypatch.setattr(codex_connection, "read_connection", read_for_thread)
    seen: dict[str, list[str]] = {}

    def call(name: str) -> None:
        seen[name] = _run_envdump(driver)["argv"]

    threads = [threading.Thread(target=call, args=(n,), name=n) for n in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert 'model_provider="alpha"' in seen["a"] and 'model_provider="beta"' not in seen["a"]
    assert 'model_provider="beta"' in seen["b"] and 'model_provider="alpha"' not in seen["b"]


def test_a_provider_failure_is_its_own_error(tmp_path: Path, monkeypatch, closed_port: int) -> None:
    from finesub.llm.agent.local_agent import LocalAgentProviderError

    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=closed_port))
    with pytest.raises(LocalAgentProviderError):
        _driver(tmp_path, monkeypatch).run([{"role": "user", "content": "ok"}], task="general")
