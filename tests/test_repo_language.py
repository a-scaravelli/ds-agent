"""Convention: the entire repository is in English (heuristic checks, no network)."""

import ast
import io
import json
import re
import tokenize
from pathlib import Path

import pytest

from kaggle_agent.proposals import Attempt, ShortPlan, Suggestion, ValidationError
from test_design_doc import parse_sections

ROOT = Path(__file__).resolve().parent.parent
THIS_FILE = Path(__file__).resolve()
PY_FILES = sorted(p for d in ("src", "tests") for p in (ROOT / d).rglob("*.py"))
MEMORY = ROOT / ".asl" / "process-memory.md"
CONVENTION = "entire repository in english"

ITALIAN_IDENTIFIER_WORDS = {
    "piano", "tentativo", "tentativi", "ipotesi", "cosa", "cambia", "metrica",
    "motivazione", "evidenza", "validazione", "modelli", "voci", "suggerimento",
    "suggerimenti", "ciclo", "cicli", "fase", "misura", "rischi", "artefatti",
    "costo", "stimato", "effetto", "atteso", "esplorativa", "esplorativo",
    "errore", "nessuna", "nessun", "sezione", "campo", "campi", "esempio",
}
ITALIAN_FUNCTION_WORDS = [
    "della", "delle", "degli", "dello", "nel", "nella", "sono", "che", "gli",
    "anche", "questo", "quindi", "perché", "ogni",
]
ITALIAN_FIELD_NAMES = [
    "cosa_cambia", "ipotesi", "motivazione", "artefatti", "effetto_atteso", "costo_stimato",
]
ENGLISH_FIELDS = ["hypothesis", "rationale", "parent_id", "what_changed", "exploratory", "artifacts"]

PLAN = {
    "metric": "m", "validation": "v", "feature_engineering": ["f"], "models": ["x"], "baseline": "b",
}
ATTEMPT = {
    "id": "T-1", "phase": "p", "hypothesis": "h", "rationale": "r",
    "what_changed": "w", "measure": "m", "exploratory": True,
}


def _memory_is_translated() -> bool:
    return CONVENTION in MEMORY.read_text(encoding="utf-8").lower()


def _defined_names(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.arg):
            yield node.arg
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.id


def test_code_identifiers_are_english():
    for path in PY_FILES:
        for name in _defined_names(ast.parse(path.read_text(encoding="utf-8"))):
            words = set(name.lower().split("_"))
            assert not words & ITALIAN_IDENTIFIER_WORDS, f"{path.name}: {name}"
    for path in (ROOT / "src").rglob("*.py"):
        assert not set(path.stem.split("_")) & ITALIAN_IDENTIFIER_WORDS, path


def test_to_dict_keys_and_error_messages_are_english():
    assert list(ShortPlan.from_dict(PLAN).to_dict()) == list(PLAN)
    assert set(Attempt.from_dict(ATTEMPT).to_dict()) == {
        "id", "phase", "parent_id", "hypothesis", "rationale", "evidence", "exploratory",
        "what_changed", "expected_effect", "measure", "estimated_cost", "risks", "artifacts",
    }
    assert list(Suggestion.from_dict(
        {"id": "S", "text": "t", "origin_cycle": "c"}).to_dict()
    ) == ["id", "text", "origin_cycle", "spawned_cycles"]
    cases = [
        (ShortPlan, {k: v for k, v in PLAN.items() if k != "models"}, "models"),
        (ShortPlan, {**PLAN, "baseline": ""}, "baseline"),
        (Attempt, {k: v for k, v in ATTEMPT.items() if k != "what_changed"}, "what_changed"),
        (Attempt, {k: v for k, v in ATTEMPT.items() if k != "hypothesis"}, "hypothesis"),
        (Attempt, {**ATTEMPT, "exploratory": False}, "evidence"),
        (Suggestion, {"id": "S", "text": "", "origin_cycle": "c"}, "text"),
    ]
    for cls, data, field in cases:
        with pytest.raises(ValidationError) as exc:
            cls.from_dict(data)
        assert field in str(exc.value) and exc.value.field.startswith(field)
        words = set(re.findall(r"[a-zA-Z_]+", str(exc.value).lower()))
        assert not words & (ITALIAN_IDENTIFIER_WORDS | set(ITALIAN_FUNCTION_WORDS))


def _docstrings_and_comments(path: Path):
    source = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                yield doc
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            yield tok.string


def _function_word_hits(text: str) -> list[str]:
    pattern = r"\b(" + "|".join(ITALIAN_FUNCTION_WORDS) + r")\b"
    return re.findall(pattern, text, flags=re.I)


def test_repo_text_has_no_italian_function_words():
    texts = {
        name: (ROOT / name).read_text(encoding="utf-8")
        for name in ("docs/design.md", "README.md", ".asl/test-map.json", "pyproject.toml", ".gitignore")
    }
    if _memory_is_translated():
        texts[".asl/process-memory.md"] = MEMORY.read_text(encoding="utf-8")
    for path in PY_FILES:
        if path != THIS_FILE:
            texts[str(path.relative_to(ROOT))] = "\n".join(_docstrings_and_comments(path))
    for name, text in texts.items():
        assert not _function_word_hits(text), f"{name}: {_function_word_hits(text)}"


def test_italian_field_names_absent():
    texts = {
        name: (ROOT / name).read_text(encoding="utf-8")
        for name in ("docs/design.md", "README.md")
    }
    if _memory_is_translated():
        texts[".asl/process-memory.md"] = MEMORY.read_text(encoding="utf-8")
    for path in (ROOT / "src").rglob("*.py"):
        texts[str(path)] = path.read_text(encoding="utf-8")
    for name, text in texts.items():
        for field in ITALIAN_FIELD_NAMES:
            assert not re.search(rf"\b{field}\b", text, flags=re.I), f"{name}: {field}"


def test_english_field_names_present_in_design_and_readme():
    parsed = {n: body for n, _, body in parse_sections((ROOT / "docs/design.md").read_text(encoding="utf-8"))}
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name, text in (("section 4", parsed[4]), ("section 7", parsed[7]), ("README", readme)):
        for field in ENGLISH_FIELDS:
            assert field in text, f"{name}: {field}"


def test_process_memory_records_english_convention():
    if not _memory_is_translated():
        pytest.skip("process memory is updated by the later documentation phase")
    low = MEMORY.read_text(encoding="utf-8").lower()
    assert CONVENTION in low
    for item in ("code", "documentation", "comments", "tests", "commit", "pr"):
        assert item in low
    for entry in ("ticket #6", "ticket #7", "ticket 1"):
        assert entry in low, entry


def test_readme_cites_public_api():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("ShortPlan", "Attempt", "Suggestion", "what_changed", "MAX_PLAN_WORDS"):
        assert name in readme, name
