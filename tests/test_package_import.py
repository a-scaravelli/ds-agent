"""Il pacchetto e i sottopacchetti previsti si importano."""

import importlib

import pytest

SUBPACKAGES = [
    "loop", "proposals", "evaluation", "registry", "kaggle",
    "llm", "sandbox", "review", "budget",
]


def test_import_pacchetto_e_versione():
    pkg = importlib.import_module("kaggle_agent")
    assert isinstance(pkg.__version__, str) and pkg.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_import_sottopacchetto(name):
    mod = importlib.import_module(f"kaggle_agent.{name}")
    assert mod.__doc__
