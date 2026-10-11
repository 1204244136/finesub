"""What the shell does with a runtime that disagrees with its lock.

The runtime belongs to FineSub: a mismatch is rebuilt after saying what the
rebuild takes away, unless `[runtime] auto_repair = false` hands that decision
to whoever manages the runtime themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from finesub_bootstrap import runtime_health
from finesub_bootstrap.environment import RuntimeInUse
from finesub_bootstrap.paths import AppPaths
from finesub_bootstrap.runtime_health import RuntimeReport
from finesub_bootstrap.shell import Shell


@dataclass
class FakeRuntime:
    """The slice of `RuntimeEnvironment` the shell consults about health."""

    root: Path
    state: str = "ready"
    report: RuntimeReport | None = None
    busy: bool = False
    installs: list[dict] = field(default_factory=list)
    development_python: None = None

    @property
    def runtime_root(self) -> Path:
        return self.root

    def condition(self) -> str:
        return self.state

    def refuse_if_in_use(self, *, holds_lease: bool = False) -> None:
        if self.busy:
            raise RuntimeInUse("另有 1 个 FineSub 运行正在使用运行环境")

    def check(self, *, verify_hashes: bool = False) -> RuntimeReport | None:
        return self.report

    def health_detail(self) -> str:
        return "缺少必需依赖：torch"

    def status(self, *, force_probe: bool = False):
        class Status:
            detail = "缺少必需依赖：torch"

        return Status()

    failure: Exception | None = None

    def install(
        self,
        *,
        stage=None,
        log=None,
        should_pause=None,
        force: bool = False,
        verify_hashes: bool = False,
        holds_lease: bool = False,
    ) -> None:
        if self.busy:
            raise RuntimeInUse("另有 1 个 FineSub 运行正在使用运行环境")
        if self.failure is not None:
            raise self.failure
        options = {"force": force, "verify_hashes": verify_hashes, "holds_lease": holds_lease}
        self.installs.append(options)
        self.state = "ready"
        self.report = RuntimeReport()


def _shell(tmp_path: Path, runtime: FakeRuntime, *, config: str = "", version: str = "0.5.2") -> Shell:
    paths = AppPaths.for_root(tmp_path / "home")
    paths.user_data.mkdir(parents=True, exist_ok=True)
    if config:
        (paths.user_data / "config.toml").write_text(config, encoding="utf-8")
    return Shell(paths=paths, resources=None, runtime=runtime, version=version)  # type: ignore[arg-type]


def _quarantined() -> RuntimeReport:
    return RuntimeReport(
        missing_files={"silero-vad": ["silero_vad/data/silero_vad.jit"]},
        extras=[("downstream-plugin", "1.0")],
    )


def _ensure(shell: Shell, **options) -> bool:
    arguments = {"explicit": False, "transcription": True}
    arguments.update(options)
    return shell._ensure_runtime(**arguments)


def test_a_mismatch_found_before_a_run_is_rebuilt_and_says_what_goes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = FakeRuntime(tmp_path / "python", report=_quarantined())
    shell = _shell(tmp_path, runtime)

    _ensure(shell)

    assert runtime.installs == [{"force": True, "verify_hashes": False, "holds_lease": True}]
    err = capsys.readouterr().err
    assert "silero_vad.jit" in err
    assert "downstream-plugin 1.0" in err and "FINESUB_EXTRA_PYTHONPATH" in err
    assert "白名单" in err


def test_auto_repair_off_warns_and_lets_the_run_go_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = FakeRuntime(tmp_path / "python", report=_quarantined())
    shell = _shell(tmp_path, runtime, config="[runtime]\nauto_repair = false\n")

    _ensure(shell)

    assert runtime.installs == []
    assert "finesub setup --repair" in capsys.readouterr().err


def test_auto_repair_off_stops_a_runtime_missing_what_every_command_needs(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path / "python", state="damaged")
    shell = _shell(tmp_path, runtime, config="[runtime]\nauto_repair = false\n")

    with pytest.raises(SystemExit) as stopped:
        _ensure(shell)

    assert stopped.value.code == 1
    assert runtime.installs == []


def test_auto_repair_off_still_installs_a_runtime_that_was_never_there(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path / "python", state="absent")
    shell = _shell(tmp_path, runtime, config="[runtime]\nauto_repair = false\n")

    _ensure(shell)

    assert runtime.installs == [{"force": False, "verify_hashes": False, "holds_lease": False}]


def test_setup_rebuilds_whatever_auto_repair_says(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path / "python", report=_quarantined())
    shell = _shell(tmp_path, runtime, config="[runtime]\nauto_repair = false\n")

    _ensure(shell, explicit=True, transcription=False)

    assert runtime.installs[0]["force"] is True


def test_an_upgrade_lists_only_what_was_added_by_hand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = RuntimeReport(
        version_mismatches=[("torch", "2.10.0", "2.11.0")],
        extras=[("downstream-plugin", "1.0")],
    )
    runtime = FakeRuntime(tmp_path / "python", state="outdated", report=report)
    shell = _shell(tmp_path, runtime)

    _ensure(shell)

    err = capsys.readouterr().err
    assert "downstream-plugin 1.0" in err
    assert "torch 2.10.0" not in err


def test_another_run_using_the_runtime_blocks_a_rebuild(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = FakeRuntime(tmp_path / "python", report=_quarantined(), busy=True)
    shell = _shell(tmp_path, runtime)

    assert _ensure(shell) is False  # a quick-check mismatch: the run still goes ahead

    err = capsys.readouterr().err
    assert "FineSub" in err
    # Refused before anything is announced: no "rebuilding..." then "cannot".
    assert "正在按锁重建" not in err
    runtime.state = "damaged"
    with pytest.raises(SystemExit):
        _ensure(shell)  # an instant-tier failure: the run cannot


def test_the_check_before_a_run_follows_the_policy(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path / "python", report=_quarantined())
    runtime.root.mkdir()
    runtime_health.record_success(runtime.root, "0.5.2")

    _ensure(_shell(tmp_path, runtime))  # auto, and this version already ran
    assert runtime.installs == []

    _ensure(_shell(tmp_path, runtime, config='[runtime]\nintegrity_check = "on"\n'))
    assert len(runtime.installs) == 1


def test_a_failed_run_is_followed_by_a_check_and_a_rebuild(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = FakeRuntime(tmp_path / "python", report=_quarantined())
    shell = _shell(tmp_path, runtime)

    shell._after_run(1)

    assert runtime.installs[0]["holds_lease"] is False
    assert "重新运行" in capsys.readouterr().err


def test_a_usage_error_says_nothing_about_the_runtime(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path / "python", report=_quarantined())
    shell = _shell(tmp_path, runtime)

    shell._after_run(2)

    assert runtime.installs == []


def test_integrity_check_off_skips_the_check_after_a_failure(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path / "python", report=_quarantined())
    shell = _shell(tmp_path, runtime, config='[runtime]\nintegrity_check = "off"\n')

    shell._after_run(1)

    assert runtime.installs == []


def test_a_success_is_remembered_for_this_version(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path / "python")
    runtime.root.mkdir()
    shell = _shell(tmp_path, runtime, version="0.5.2")

    shell._after_run(0)

    assert not runtime_health.should_check_before_run("auto", runtime.root, "0.5.2")


def test_a_failure_on_a_legacy_code_page_points_at_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(runtime_health, "ansi_code_page", lambda: 936)
    runtime = FakeRuntime(tmp_path / "python", report=RuntimeReport())
    shell = _shell(tmp_path / "用户", runtime)

    shell._after_run(1)

    assert "FINESUB_HOME" in capsys.readouterr().err


def test_doctor_rejects_what_it_does_not_know(tmp_path: Path) -> None:
    shell = _shell(tmp_path, FakeRuntime(tmp_path / "python"))

    assert shell.doctor(["--repair"]) == 2


def test_the_integrity_line_names_the_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINESUB_HOME", raising=False)

    line = Shell._integrity_line(_quarantined(), verify_hashes=False)

    assert "silero_vad.jit" in line and "finesub setup --repair" in line
    assert Shell._integrity_line(RuntimeReport(), verify_hashes=True).startswith("ok")


def test_setup_repair_rebuilds_and_checks_every_digest(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path / "python", report=RuntimeReport())
    shell = _shell(tmp_path, runtime)
    shell.ensure_ready = lambda **_options: None  # tools and platform are not the subject

    assert shell._repair_runtime() == 0
    assert runtime.installs[0]["force"] is True
    assert runtime.installs[0]["verify_hashes"] is True
    assert runtime.installs[0]["holds_lease"] is False


def test_setup_repair_is_refused_while_another_run_uses_the_runtime(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path / "python", report=RuntimeReport(), busy=True)
    shell = _shell(tmp_path, runtime)
    shell.ensure_ready = lambda **_options: None

    assert shell._repair_runtime() == 1


def test_the_fake_runtime_speaks_the_real_interface() -> None:
    """Otherwise renaming an `install` keyword would leave these tests green."""

    import inspect

    from finesub_bootstrap.environment import RuntimeEnvironment

    for name in ("condition", "check", "status", "install", "refuse_if_in_use", "health_detail"):
        fake = inspect.signature(getattr(FakeRuntime, name)).parameters
        real = inspect.signature(getattr(RuntimeEnvironment, name)).parameters
        assert set(fake) <= set(real), (name, set(fake) - set(real))
    assert isinstance(RuntimeEnvironment.runtime_root, property)


def test_setup_fails_when_the_runtime_was_left_unrepaired(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shell = _shell(tmp_path, FakeRuntime(tmp_path / "python"))
    shell.ensure_ready = lambda **_options: False

    assert shell.setup([]) == 1
    assert "ready" not in capsys.readouterr().out


def test_a_failed_rebuild_keeps_the_run_going_when_the_old_runtime_serves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = FakeRuntime(tmp_path / "python", report=_quarantined())
    runtime.failure = RuntimeError("新装的运行环境里仍有文件缺失")
    shell = _shell(tmp_path, runtime)

    assert _ensure(shell) is False
    err = capsys.readouterr().err
    assert "原来的运行环境保持不变" in err and "finesub setup --repair" in err


def test_a_failed_rebuild_stops_a_run_the_old_runtime_cannot_serve(tmp_path: Path) -> None:
    import subprocess

    runtime = FakeRuntime(tmp_path / "python", state="damaged")
    runtime.failure = subprocess.CalledProcessError(1, ["uv", "pip", "install"])
    shell = _shell(tmp_path, runtime)

    with pytest.raises(SystemExit) as stopped:
        _ensure(shell)
    assert stopped.value.code == 1


def test_damage_found_after_a_failed_run_is_rebuilt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = FakeRuntime(tmp_path / "python", state="damaged")
    shell = _shell(tmp_path, runtime, config='[runtime]\nintegrity_check = "off"\n')

    shell._after_run(1)

    assert runtime.installs == [{"force": True, "verify_hashes": False, "holds_lease": False}]
    assert "重新运行" in capsys.readouterr().err


def test_setup_repair_after_an_upgrade_does_not_list_upgraded_dependencies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = RuntimeReport(version_mismatches=[("torch", "2.10.0", "2.11.0")])
    runtime = FakeRuntime(tmp_path / "python", state="outdated", report=report)
    shell = _shell(tmp_path, runtime)
    shell.ensure_ready = lambda **_options: True

    assert shell._repair_runtime() == 0
    assert "torch 2.10.0" not in capsys.readouterr().err


def test_doctor_fails_when_the_runtime_disagrees_with_its_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = FakeRuntime(tmp_path / "python", report=_quarantined())
    shell = _shell(tmp_path, runtime)

    class Ready:
        state = "ready"
        detail = ""

    monkeypatch.setattr(runtime, "status", lambda **_options: Ready(), raising=False)
    monkeypatch.setattr(shell, "_tool_state", lambda _resource: "ready")
    monkeypatch.setattr(shell, "_tool_report", lambda *_args: "ready")
    monkeypatch.setattr(shell, "_activity_report", lambda: [])
    monkeypatch.setattr(shell, "_download_route_report", lambda: "")
    monkeypatch.setattr(shell, "_env_keys_report", lambda: "")

    assert shell.doctor([]) == 1
    runtime.report = RuntimeReport()
    assert shell.doctor([]) == 0


def test_a_missing_installer_does_not_suggest_trying_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = FakeRuntime(tmp_path / "python", report=_quarantined())
    runtime.failure = FileNotFoundError("uv bootstrap executable was not found: uv.exe")
    shell = _shell(tmp_path, runtime)

    assert _ensure(shell) is False
    err = capsys.readouterr().err
    assert "重新安装 finesub" in err and "稍后" not in err
