"""The final report: sections, export, print view, labels in every surface, the bundle page (cases 7, 8, 9, 49)."""

import tomllib
from pathlib import Path

from cockpit_finish_helpers import LABELS, FinishWorld, labelled_world

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("Chosen bundle", "Holdout", "Rounds", "What worked", "What failed", "Suggestions",
            "Persisted bundle", "Submissions", "Decisions of the DS")


# 7
def test_the_report_page_has_every_section_and_escapes_what_it_shows(tmp_path):
    fw = labelled_world(tmp_path)
    fw.ok("/api/notes", {"target": {"type": "run", "id": "toy-1"}, "text": "<script>alert(1)</script> note"})
    page = fw.page("/cycles/toy-1/report")
    for section in SECTIONS:
        assert section in page, section
    assert "<script>alert(1)" not in page and "chosen by the DS" in page
    export = fw.client.get("/api/runs/toy-1/report.md").get_data(as_text=True)
    for section in SECTIONS:
        assert f"## {section}" in export, section
    # the registry holds the report once: the page is a view of it
    assert fw.get_json("/api/runs/toy-1/final-report")["available"] is True


# 8
def test_the_markdown_export_is_masked_while_sealed_and_the_print_view_has_no_external_assets(tmp_path):
    fw = FinishWorld(tmp_path)
    response = fw.client.get("/api/runs/toy-1/report.md")
    assert response.status_code == 200 and "attachment" in response.headers["Content-Disposition"]
    sealed = response.get_data(as_text=True)
    assert "sealed" in sealed and "0.4167" not in sealed
    print_view = fw.client.get("/cycles/toy-1/report/print")
    html = print_view.get_data(as_text=True)
    assert "css/print.css" in html and "http://" not in html and "https://" not in html
    assert "<style" not in html and "style=" not in html  # the CSP allows no inline style: print.css does it
    csp = print_view.headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp and "style-src 'self'" in csp
    fw.ok("/api/holdout/unseal", {"kind": "first", "confirm": True})
    unsealed = fw.client.get("/api/runs/toy-1/report.md").get_data(as_text=True)
    assert "0.4167" in unsealed and "0.4167" in fw.page("/cycles/toy-1/report/print")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert not any("weasy" in d.lower() or "pdf" in d.lower() or "reportlab" in d.lower()
                   for d in project["dependencies"])  # the PDF is the browser's print, no new dependency


# 9
def test_the_bundle_page_shows_the_hashes_and_the_library_versions(tmp_path):
    fw = FinishWorld(tmp_path)
    bundle = fw.get_json("/api/runs/toy-1/bundle")
    assert bundle["available"] is True and bundle["bundle_id"] == "toy-1-R0-c1-b1"
    for key in ("folder", "model", "protocol", "data"):
        assert len(bundle["hashes"][key]) == 64
    assert bundle["library_versions"]["numpy"] and bundle["library_versions"]["scikit-learn"]
    assert bundle["code"]["note"] == "the code is never shown or served"
    page = fw.page("/cycles/toy-1/bundle")
    assert bundle["hashes"]["folder"] in page and bundle["hashes"]["model"] in page and "numpy" in page
    assert "code/" not in " ".join(f["name"] for f in bundle["files"])


# 49
def test_every_surface_shows_the_same_labels_reasons_and_the_program_choice(tmp_path):
    fw = labelled_world(tmp_path)
    labels = fw.get_json("/api/runs/toy-1/labels")
    assert {x["label"] for x in labels["labels"]} >= set(LABELS)
    reasons = {x["label"]: x["reason"] for x in labels["labels"]}
    assert reasons["chosen by the DS"] == "simpler model" and reasons["bar overridden by the DS"].startswith("the gain")
    surfaces = {
        "page": fw.page("/cycles/toy-1/report"),
        "print": fw.page("/cycles/toy-1/report/print"),
        "markdown": fw.client.get("/api/runs/toy-1/report.md").get_data(as_text=True),
        "file": (fw.project / "out/reports/toy-1.md").read_text(encoding="utf-8"),
    }
    for name, body in surfaces.items():
        for label in LABELS:
            assert label in body, (name, label)
        assert "simpler model" in body and "toy-1-R0-c1" in body, name  # the program's own pick is shown too
    assert "<!-- ds-decisions:begin -->" in surfaces["file"] and "<!-- ds-decisions:end -->" in surfaces["file"]
    api = fw.get_json("/api/runs/toy-1/final-report")
    assert api["available"] and "chosen by the DS" in str(fw.get_json("/api/runs/toy-1/labels"))
