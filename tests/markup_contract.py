"""Extracts the functional markup of the cockpit templates (ids, names, links, hx-*, data-*, aria-*, classes).

``python tests/markup_contract.py`` regenerates tests/fixtures/web_markup_contract.json; the design
tests check that a restyle never removes anything recorded there (it may only add).
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "src" / "kaggle_agent" / "web" / "templates"
FIXTURE = ROOT / "tests" / "fixtures" / "web_markup_contract.json"

ATTRIBUTE = re.compile(r'\b(id|name|type|href|action|method|data-[\w-]+|hx-[\w-]+|aria-[\w-]+)="([^"]*)"')
CLASS = re.compile(r'\bclass="([^"]*)"')
CONTROL = re.compile(r"<(input|select|textarea|button|form|dialog|table|details)\b")


def extract(text: str) -> dict:
    found = {f"{n}={v}" for n, v in ATTRIBUTE.findall(text)}
    classes = set()
    for value in CLASS.findall(text):
        classes.update(c for c in re.sub(r"\{[{%].*?[}%]\}", " ", value).split() if c)
    return {"attributes": sorted(found), "classes": sorted(classes), "controls": sorted(set(CONTROL.findall(text)))}


def current() -> dict:
    return {p.name: extract(p.read_text(encoding="utf-8")) for p in sorted(TEMPLATES.glob("*.html"))}


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(current(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
