"""Helpers for the sandbox tests: tiny fake scripts and a fake protected tree. No network."""

import textwrap
from pathlib import Path

from kaggle_agent.sandbox import ProtectedPath, SandboxPolicy

EVAL_NAME = "src/kaggle_agent/evaluation"
PROTOCOL_NAME = "protocol.json"
RUNNER_NAME = "runner.py"
HOLDOUT_NAME = "data/holdout.csv"


def write_script(directory: Path, name: str, body: str) -> Path:
    """Write a Python script (dedented ``body``) into ``directory`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def ok_script(d: Path) -> Path:
    return write_script(d, "ok.py", """
        import sys
        print("ok")
        print("warn", file=sys.stderr)
    """)


def sleeper(d: Path) -> Path:
    """Spawns a sleeping grandchild (pid written to argv[1]) and sleeps itself."""
    return write_script(d, "sleeper.py", """
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open(sys.argv[1], "w").write(str(child.pid))
        time.sleep(30)
    """)


def background_then_exit(d: Path) -> Path:
    """Starts a long-sleeping grandchild (pid in argv[1]) and exits 0 at once."""
    return write_script(d, "background.py", """
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open(sys.argv[1], "w").write(str(child.pid))
    """)


def writer(d: Path) -> Path:
    """Opens argv[1] for writing; ``protected:NAME`` means NAME in SANDBOX_PROTECTED_DIR."""
    return write_script(d, "writer.py", """
        import os, sys
        target = sys.argv[1]
        if target.startswith("protected:"):
            target = os.path.join(os.environ["SANDBOX_PROTECTED_DIR"], target[len("protected:"):])
        with open(target, "a") as handle:
            handle.write("tampered")
    """)


def chmod_writer(d: Path) -> Path:
    """Like ``writer`` but chmods the target first."""
    return write_script(d, "chmod_writer.py", """
        import os, sys
        target = os.path.join(os.environ["SANDBOX_PROTECTED_DIR"], sys.argv[1])
        os.chmod(target, 0o644)
        with open(target, "a") as handle:
            handle.write("tampered")
    """)


def env_reader(d: Path) -> Path:
    """Prints ``NAME=value`` or ``NAME=ABSENT`` for each name in argv[1:]."""
    return write_script(d, "env_reader.py", """
        import os, sys
        for name in sys.argv[1:]:
            print(name + "=" + os.environ.get(name, "ABSENT"))
    """)


def env_dumper(d: Path) -> Path:
    return write_script(d, "env_dumper.py", """
        import json, os, sys
        print(json.dumps({"env": dict(os.environ), "argv": sys.argv}))
    """)


def home_reader(d: Path) -> Path:
    return write_script(d, "home_reader.py", """
        import os
        home = os.environ.get("HOME", "")
        print("HOME=" + home)
        print("KAGGLE_DIR_EXISTS=" + str(os.path.exists(os.path.expanduser("~/.kaggle"))))
        print("HOME_LISTING=" + ",".join(sorted(os.listdir(home))) if os.path.isdir(home) else "")
    """)


def failer(d: Path) -> Path:
    return write_script(d, "failer.py", """
        import sys
        print("boom", file=sys.stderr)
        sys.exit(3)
    """)


def flooder(d: Path) -> Path:
    return write_script(d, "flooder.py", """
        import sys
        sys.stdout.write("o" * 2_000_000)
        sys.stdout.flush()
        sys.stderr.write("e" * 2_000_000)
        sys.stderr.flush()
    """)


def cwd_script(d: Path) -> Path:
    return write_script(d, "cwd.py", """
        import os
        print(os.getcwd())
        print(len(os.listdir(".")))
        open("made.txt", "w").write("x")
    """)


def import_probe(d: Path) -> Path:
    """Tries ``import leaky_mod``; optional argv[1] is an explicit folder to add to sys.path."""
    return write_script(d, "import_probe.py", """
        import os, sys
        print("PYTHONPATH=" + os.environ.get("PYTHONPATH", "ABSENT"))
        print("SYSPATH=" + "|".join(sys.path))
        if len(sys.argv) > 1:
            sys.path.insert(0, sys.argv[1])
        try:
            import leaky_mod
            print("IMPORTED")
        except ImportError:
            print("IMPORT_FAILED")
    """)


def marker_writer(d: Path) -> Path:
    """Creates the file named in argv[1]: proves the script started."""
    return write_script(d, "marker.py", """
        import sys
        open(sys.argv[1], "w").write("started")
    """)


def make_policy(root: Path, timeout: float = 20, **kwargs) -> SandboxPolicy:
    """A fake repository under ``root`` with a protected package, protocol, runner and holdout."""
    package = root / EVAL_NAME
    (package / "sub").mkdir(parents=True, exist_ok=True)
    (package / "__pycache__").mkdir(exist_ok=True)
    (package / "metrics.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "sub" / "inner.py").write_text("INNER = 2\n", encoding="utf-8")
    (package / "__pycache__" / "metrics.pyc").write_bytes(b"cache")
    (root / PROTOCOL_NAME).write_text('{"metric": "auc"}\n', encoding="utf-8")
    (root / RUNNER_NAME).write_text("# runner\n", encoding="utf-8")
    (root / "data").mkdir(exist_ok=True)
    (root / HOLDOUT_NAME).write_text("secret,holdout\n1,2\n", encoding="utf-8")
    (root / "registry.sqlite").write_bytes(b"registry")
    home = root / "fakehome" / ".kaggle"
    home.mkdir(parents=True, exist_ok=True)
    (home / "kaggle.json").write_text('{"key": "SUPER-SECRET-KAGGLE-KEY"}', encoding="utf-8")
    protected = (
        ProtectedPath(EVAL_NAME, package),
        ProtectedPath(PROTOCOL_NAME, root / PROTOCOL_NAME),
        ProtectedPath(RUNNER_NAME, root / RUNNER_NAME),
        ProtectedPath(HOLDOUT_NAME, root / HOLDOUT_NAME, expose=False),
    )
    hidden = (root / HOLDOUT_NAME, root / "registry.sqlite", home / "kaggle.json")
    return SandboxPolicy(timeout_seconds=timeout, protected=protected, hidden_paths=hidden, **kwargs)
