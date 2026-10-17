"""The package and the expected subpackages can be imported."""

import importlib

import pytest

SUBPACKAGES = [
    "loop", "proposals", "evaluation", "registry", "kaggle",
    "llm", "sandbox", "review", "budget",
]


def test_package_import_and_version():
    pkg = importlib.import_module("kaggle_agent")
    assert isinstance(pkg.__version__, str) and pkg.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    mod = importlib.import_module(f"kaggle_agent.{name}")
    assert mod.__doc__
