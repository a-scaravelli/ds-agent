"""The subprocess sandbox with fake scripts. No network, no Claude call."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from sandbox_helpers import (
    EVAL_NAME, HOLDOUT_NAME, PROTOCOL_NAME, RUNNER_NAME, background_then_exit, chmod_writer,
    cwd_script, env_dumper, env_reader, failer, flooder, home_reader, import_probe, make_policy,
    marker_writer, ok_script, sleeper, write_script, writer,
)

from kaggle_agent.registry import FAILURE_KINDS, ExecutionRecord
from kaggle_agent.sandbox import (
    FAILURE_KIND_BY_STATUS, MINIMAL_KEYS, STATUSES, Sandbox, SandboxConfigError, SandboxPolicy,
    SandboxRequest, SandboxResult, SubprocessSandbox, hash_paths,
)

# Keys the platform or the interpreter adds to every child process (macOS adds
# __CF_USER_TEXT_ENCODING, Python may add LC_CTYPE): not credentials, not set by the sandbox.
PLATFORM_ADDED_KEYS = {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"}

NOT_ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="permissions do not stop root")


def run(tmp_path, script, *args, policy=None, **request):
    policy = policy or make_policy(tmp_path / "repo")
    result = SubprocessSandbox().run(policy, SandboxRequest(entry_point=script, args=args, **request))
    return policy, result


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a killed process nobody has reaped yet is a zombie, not alive; `ps` works on Linux and macOS
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return bool(state) and not state.startswith("Z")
    except (OSError, subprocess.SubprocessError):
        pass
    try:  # no `ps`: fall back to /proc where it exists
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def wait_dead(pid: int, seconds: float = 5) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return not alive(pid)


class FakeSandbox(Sandbox):
    def run(self, policy, request):
        return SandboxResult(status="succeeded", stdout="fake")


def consume(sandbox: Sandbox, policy, request):
    return sandbox.run(policy, request).status


def test_interface_is_small_and_replaceable(tmp_path):
    assert issubclass(SubprocessSandbox, Sandbox)
    assert Sandbox.__abstractmethods__ == frozenset({"run"})
    with pytest.raises(TypeError):
        Sandbox()
    policy = make_policy(tmp_path / "repo")
    request = SandboxRequest(command=("true",))
    assert consume(FakeSandbox(), policy, request) == "succeeded"
    assert consume(SubprocessSandbox(), policy, SandboxRequest(command=(sys.executable, "-c", "pass"))) == "succeeded"


def test_python_entry_point_with_captured_output(tmp_path):
    _, result = run(tmp_path, ok_script(tmp_path / "s"))
    assert result.status == "succeeded" and result.exit_code == 0
    assert "ok" in result.stdout and "warn" in result.stderr
    assert result.failure_kind is None and result.duration_seconds > 0
    assert json.dumps(result.to_dict())


def test_command_and_exclusive_request(tmp_path):
    policy = make_policy(tmp_path / "repo")
    request = SandboxRequest(command=(sys.executable, "-c", "print(6 * 7)"))
    result = SubprocessSandbox().run(policy, request)
    assert result.status == "succeeded" and result.stdout.strip() == "42"
    with pytest.raises(SandboxConfigError):
        SandboxRequest(command=("true",), entry_point=Path("x.py"))
    with pytest.raises(SandboxConfigError):
        SandboxRequest()


def test_timeout_kills_child_and_process_group(tmp_path):
    pid_file = tmp_path / "pid"
    policy = make_policy(tmp_path / "repo", timeout=3)
    started = time.monotonic()
    _, result = run(tmp_path, sleeper(tmp_path / "s"), str(pid_file), policy=policy)
    assert time.monotonic() - started < 10
    assert result.status == "timed_out" and result.failure_kind == "timeout"
    assert "timed out" in result.reason and result.exit_code is None
    assert wait_dead(int(pid_file.read_text()))


def test_leftover_descendants_are_killed_after_a_normal_exit(tmp_path):
    pid_file = tmp_path / "pid"
    _, result = run(tmp_path, background_then_exit(tmp_path / "s"), str(pid_file))
    assert result.status == "succeeded"
    assert wait_dead(int(pid_file.read_text()))


def test_timeout_is_mandatory_and_positive():
    with pytest.raises(TypeError):
        SandboxPolicy()
    for bad in (0, -1, None, "5"):
        with pytest.raises(SandboxConfigError):
            SandboxPolicy(timeout_seconds=bad)


def test_own_working_directory_that_is_empty_and_removed(tmp_path):
    script = cwd_script(tmp_path / "s")
    cwds = []
    for _ in range(2):
        _, result = run(tmp_path, script)
        cwd, count = result.stdout.split()
        assert count == "0" and cwd != os.getcwd()
        assert not Path(cwd).exists()
        assert not Path(cwd).parent.exists()
        cwds.append(cwd)
    assert cwds[0] != cwds[1]


def test_output_is_limited_without_blocking(tmp_path):
    policy = make_policy(tmp_path / "repo", max_output_bytes=1000)
    _, result = run(tmp_path, flooder(tmp_path / "s"), policy=policy)
    assert result.status == "succeeded"
    assert len(result.stdout) == 1000 and len(result.stderr) == 1000
    assert result.stdout_truncated and result.stderr_truncated
    _, small = run(tmp_path, ok_script(tmp_path / "s"), policy=policy)
    assert not small.stdout_truncated and not small.stderr_truncated


def test_child_sees_no_credentials_or_inherited_secrets(tmp_path, monkeypatch):
    names = ["KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "ANTHROPIC_API_KEY",
             "CLAUDE_CODE_OAUTH_TOKEN", "MY_SECRET"]
    for name in names:
        monkeypatch.setenv(name, "leaked-" + name)
    before = dict(os.environ)
    _, result = run(tmp_path, env_reader(tmp_path / "s"), *names)
    assert result.stdout.split() == [f"{n}=ABSENT" for n in names]
    assert "leaked-" not in result.stdout + result.stderr
    _, dump = run(tmp_path, env_dumper(tmp_path / "s"))
    child_env = json.loads(dump.stdout)["env"]
    assert set(child_env) <= set(MINIMAL_KEYS) | PLATFORM_ADDED_KEYS
    assert dict(os.environ) == before


def test_pythonpath_is_not_inherited_and_paths_must_be_explicit(tmp_path, monkeypatch):
    leaky = tmp_path / "leaky"
    leaky.mkdir()
    (leaky / "leaky_mod.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(leaky))
    probe = import_probe(tmp_path / "s")
    _, result = run(tmp_path, probe)
    assert "IMPORT_FAILED" in result.stdout and "PYTHONPATH=ABSENT" in result.stdout
    assert str(leaky) not in result.stdout
    _, explicit = run(tmp_path, probe, str(leaky))
    assert "IMPORTED" in explicit.stdout


def test_sensitive_extra_env_rejected_and_harmless_one_reaches_child(tmp_path):
    for name in ("KAGGLE_KEY", "API_TOKEN", "PYTHONPATH"):
        with pytest.raises(SandboxConfigError):
            SandboxRequest(command=("true",), extra_env={name: "v"})
    _, result = run(tmp_path, env_reader(tmp_path / "s"), "SEED", extra_env={"SEED": "3"})
    assert result.stdout.strip() == "SEED=3"


def test_home_is_redirected_and_credentials_are_not_reachable_through_it(tmp_path, monkeypatch):
    policy = make_policy(tmp_path / "repo")
    fake_home = tmp_path / "repo" / "fakehome"
    monkeypatch.setenv("HOME", str(fake_home))
    _, result = run(tmp_path, home_reader(tmp_path / "s"), policy=policy)
    home = next(l for l in result.stdout.splitlines() if l.startswith("HOME=")).split("=", 1)[1]
    assert home != str(fake_home) and "sandbox-" in home
    assert "KAGGLE_DIR_EXISTS=False" in result.stdout
    assert "SUPER-SECRET" not in result.stdout + result.stderr


def test_child_gets_neither_holdout_nor_registry_nor_credential_paths(tmp_path):
    policy = make_policy(tmp_path / "repo")
    _, result = run(tmp_path, env_dumper(tmp_path / "s"), policy=policy)
    text = result.stdout
    for hidden in policy.hidden_paths:
        assert str(hidden) not in text and hidden.name not in text
    copy_listing = write_script(tmp_path / "s", "listing.py", """
        import os
        base = os.environ["SANDBOX_PROTECTED_DIR"]
        print(sorted(os.path.relpath(os.path.join(r, f), base) for r, _, fs in os.walk(base) for f in fs))
    """)
    _, listing = run(tmp_path, copy_listing, policy=policy)
    assert "holdout" not in listing.stdout and "registry" not in listing.stdout
    assert PROTOCOL_NAME in listing.stdout
    with pytest.raises(SandboxConfigError):
        run(tmp_path, env_reader(tmp_path / "s"), str(policy.hidden_paths[0]), policy=policy)


def test_hashes_before_and_after_are_recorded_and_equal(tmp_path):
    policy, result = run(tmp_path, ok_script(tmp_path / "s"))
    assert set(result.hashes_before) == {p.name for p in policy.protected}
    assert result.hashes_before == result.hashes_after == hash_paths(policy.protected)
    assert result.changed_paths == ()


@NOT_ROOT
def test_writing_to_protected_copy_is_impossible(tmp_path):
    policy, result = run(tmp_path, writer(tmp_path / "s"), "protected:" + PROTOCOL_NAME)
    assert result.status == "crashed" and "PermissionError" in result.stderr
    assert result.hashes_before == result.hashes_after
    assert (tmp_path / "repo" / PROTOCOL_NAME).read_text() == '{"metric": "auc"}\n'
    _, folder = run(tmp_path, writer(tmp_path / "s"), "protected:" + EVAL_NAME + "/metrics.py", policy=policy)
    assert folder.status == "crashed"


def test_modifying_an_original_is_protected_modified_even_with_exit_0(tmp_path):
    repo = tmp_path / "repo"
    policy = make_policy(repo)
    _, result = run(tmp_path, writer(tmp_path / "s"), str(repo / PROTOCOL_NAME), policy=policy)
    assert result.exit_code == 0
    assert result.status == "protected_modified" and result.failure_kind == "code_error"
    assert result.changed_paths == (PROTOCOL_NAME,)
    assert result.hashes_before[PROTOCOL_NAME] != result.hashes_after[PROTOCOL_NAME]
    assert result.status not in ("crashed", "timed_out")


def test_modifying_the_holdout_original_is_detected(tmp_path):
    repo = tmp_path / "repo"
    policy = make_policy(repo)
    joiner = write_script(tmp_path / "s", "joiner.py", """
        import os, sys
        open(os.path.join(*sys.argv[1:]), "a").write("tampered")
    """)
    _, result = run(tmp_path, joiner, str(repo), *HOLDOUT_NAME.split("/"), policy=policy)
    assert result.status == "protected_modified" and HOLDOUT_NAME in result.changed_paths


def test_modified_copy_is_detected(tmp_path):
    _, result = run(tmp_path, chmod_writer(tmp_path / "s"), RUNNER_NAME)
    assert result.status == "protected_modified"
    assert result.changed_paths == ("copy:" + RUNNER_NAME,)
    assert result.hashes_before == result.hashes_after


def test_expected_hash_mismatch_blocks_the_run_before_starting(tmp_path):
    marker = tmp_path / "marker"
    policy = make_policy(tmp_path / "repo")
    expected = dict(hash_paths(policy.protected))
    expected[PROTOCOL_NAME] = "0" * 64
    _, result = run(tmp_path, marker_writer(tmp_path / "s"), str(marker), policy=policy, expected_hashes=expected)
    assert result.status == "protected_modified" and "before the run" in result.reason
    assert not marker.exists() and result.exit_code is None
    assert result.changed_paths == (PROTOCOL_NAME,)
    ok_hashes = hash_paths(policy.protected)
    _, fine = run(tmp_path, marker_writer(tmp_path / "s"), str(marker), policy=policy, expected_hashes=ok_hashes)
    assert fine.status == "succeeded" and marker.exists()


@pytest.mark.parametrize(
    "diff",
    [
        f"diff --git a/{EVAL_NAME}/metrics.py b/{EVAL_NAME}/metrics.py\n--- a/{EVAL_NAME}/metrics.py\n+++ b/{EVAL_NAME}/metrics.py\n",
        f"--- /dev/null\n+++ b/{PROTOCOL_NAME}\n",
        f"rename from bundle/a.py\nrename to {RUNNER_NAME}\n",
        f"--- a/{EVAL_NAME}/metrics.py\n+++ /dev/null\n",
        f"--- a/bundle/..//{EVAL_NAME}/metrics.py\n+++ b/bundle/x.py\n",
        "--- a/../src/other.py\n+++ b/../src/other.py\n",
    ],
)
def test_diff_touching_protected_files_is_rejected_before_running(tmp_path, diff):
    marker = tmp_path / "marker"
    _, result = run(tmp_path, marker_writer(tmp_path / "s"), str(marker), diff=diff)
    assert result.status == "diff_rejected" and result.failure_kind == "code_error"
    assert result.changed_paths and not marker.exists() and result.exit_code is None


def test_diff_on_other_files_is_accepted(tmp_path):
    marker = tmp_path / "marker"
    diff = "--- a/bundle/model.py\n+++ b/bundle/model.py\n--- a/evaluation_notes.md\n+++ b/evaluation_notes.md\n"
    _, result = run(tmp_path, marker_writer(tmp_path / "s"), str(marker), diff=diff)
    assert result.status == "succeeded" and marker.exists()


def test_integrity_failures_are_distinguishable_from_a_plain_crash(tmp_path):
    repo = tmp_path / "repo"
    policy = make_policy(repo)
    _, crash = run(tmp_path, failer(tmp_path / "s"), policy=policy)
    _, tamper = run(tmp_path, writer(tmp_path / "s"), str(repo / RUNNER_NAME), policy=policy)
    _, rejected = run(tmp_path, ok_script(tmp_path / "s"), policy=policy, diff=f"--- a/{RUNNER_NAME}\n+++ b/{RUNNER_NAME}\n")
    results = [crash, tamper, rejected]
    assert {r.failure_kind for r in results} == {"code_error"}
    assert [r.status for r in results] == ["crashed", "protected_modified", "diff_rejected"]
    assert crash.exit_code == 3 and "exit code 3" in crash.reason and "boom" in crash.reason
    retry_candidates = [r for r in results if r.status == "crashed"]
    assert retry_candidates == [crash]


def test_failure_kinds_come_from_the_existing_vocabulary():
    assert FAILURE_KINDS == ("code_error", "hypothesis_falsified", "timeout", "below_threshold")
    assert set(FAILURE_KIND_BY_STATUS) == set(STATUSES)
    for status in STATUSES:
        kind = SandboxResult(status=status).failure_kind
        assert kind is None or kind in FAILURE_KINDS
    assert FAILURE_KIND_BY_STATUS == {
        "succeeded": None, "timed_out": "timeout", "crashed": "code_error",
        "protected_modified": "code_error", "diff_rejected": "code_error",
    }


def test_a_failed_result_can_feed_an_execution_record(tmp_path):
    _, result = run(tmp_path, failer(tmp_path / "s"))
    record = ExecutionRecord(
        id="E-1", attempt_id="T-1", config={}, seed=1, code_hash="c", data_hash="d",
        protocol_hash="p", metrics=(), duration_seconds=result.duration_seconds,
        outcome="failed", failure_kind=result.failure_kind, reason=result.reason,
    )
    assert record.failure_kind == "code_error" and "exit code 3" in record.reason
