"""The web app only reads: GET routes, no form, a read-only registry (no network)."""

import ast
import hashlib
import inspect
from pathlib import Path

import pytest
from web_helpers import (  # noqa: F401  (fixtures)
    all_urls,
    db_path,
    no_network,
    page,
    registry,
    web_client,
)

from kaggle_agent.registry import Cycle, Registry, RegistryError, SqliteRegistry
from kaggle_agent.web import create_app

WEB = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent" / "web"
WRITE_PREFIXES = ("add_", "set_", "close_", "link_", "update_", "open_")
READ_PREFIXES = ("list_", "get_", "metric_series", "cycles_of_suggestion", "count_")


# 21
def test_every_rule_accepts_only_get(web_client):
    rules = [r for r in web_client.application.url_map.iter_rules()]
    assert rules
    for rule in rules:
        assert (rule.methods or set()) - {"HEAD", "OPTIONS"} == {"GET"}, rule.rule


# 22
def test_other_methods_answer_405_and_leave_the_file_unchanged(web_client, db_path):
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    for url in all_urls().values():
        for method in ("post", "put", "patch", "delete"):
            response = getattr(web_client, method)(url, data={"text": "x"}, json=None)
            assert response.status_code == 405, (method, url)
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before


# 23
def test_no_form_or_input_in_pages_or_sources(web_client):
    for url in all_urls().values():
        html = page(web_client, url).lower()
        for token in ("<form", "<input", "<textarea", "<button", "method="):
            assert token not in html, (url, token)
    for path in WEB.rglob("*"):
        if path.suffix in {".py", ".html"}:
            text = path.read_text(encoding="utf-8")
            for token in ("request.form", "request.json", "request.data", "request.files"):
                assert token not in text, (path, token)


class SpyRegistry:
    """Delegates to a real registry and records the name of every method called."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = set()

    def __getattribute__(self, name):
        if name.startswith("_") or name in ("calls",):
            return object.__getattribute__(self, name)
        inner = object.__getattribute__(self, "_inner")
        calls = object.__getattribute__(self, "calls")
        target = getattr(inner, name)
        if callable(target):
            calls.add(name)
        return target


# 24
def test_the_app_calls_only_read_methods_of_the_registry(registry):
    spy = SpyRegistry(registry)
    client = create_app(spy).test_client()
    for url in all_urls().values():
        page(client, url)
    for url in ("/cycles/main/traces?role=modeling", "/cycles/main/metrics?partition=cv"):
        page(client, url)
    assert spy.calls
    for name in spy.calls:
        assert not name.startswith(WRITE_PREFIXES), name
        assert name.startswith(READ_PREFIXES), name


# 25
def test_the_file_is_untouched_and_the_read_only_instance_refuses_writes(db_path):
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    with SqliteRegistry(db_path, read_only=True) as reg:
        client = create_app(reg).test_client()
        for url in all_urls().values():
            page(client, url)
        from registry_helpers import make_cycle

        with pytest.raises(RegistryError):
            reg.add_cycle(make_cycle("new"))
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before
    assert sorted(p.name for p in db_path.parent.iterdir()) == [db_path.name]


# 26
def test_code_inspection_finds_no_write_call_and_no_sqlite_import():
    for path in sorted(WEB.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr != "add_argument":  # argparse
                assert not node.attr.startswith(WRITE_PREFIXES), f"{path.name}: {node.attr}"
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] != "sqlite3" for a in node.names), path
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module.split(".")[0] != "sqlite3", path
                imports_sqlite = module == "kaggle_agent.registry.sqlite" or (
                    node.level == 2 and module == "registry.sqlite"
                )
                assert not imports_sqlite or path.name == "__main__.py", path


# 29
def test_metric_direction_is_not_readable_through_the_registry():
    protocol_words = ("protocol", "direction")
    for name, member in inspect.getmembers(Registry, inspect.isfunction):
        if name.startswith(("get_", "list_")):
            assert not any(w in name for w in protocol_words), name
    assert not any(
        "direction" in name or "protocol" in name and name != "protocol_hash"
        for name in Cycle.__dataclass_fields__
    )
    assert "set_protocol_hash" in Registry.__abstractmethods__
    web_text = "\n".join(p.read_text(encoding="utf-8") for p in WEB.glob("*.py"))
    assert "kaggle_agent.evaluation" not in web_text
    assert "evaluation" not in {
        n.module.split(".")[-1] for n in ast.walk(ast.parse(web_text)) if isinstance(n, ast.ImportFrom) and n.module
    }
