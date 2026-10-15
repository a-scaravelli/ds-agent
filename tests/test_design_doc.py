"""Verifica struttura e contenuto di docs/design.md (nessuna rete)."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DESIGN = ROOT / "docs" / "design.md"

SECTION_TITLES = {
    1: "Obiettivi e non-obiettivi",
    2: "Ciclo di lavoro",
    3: "Fasi del lavoro",
    4: "Formato di una proposta",
    5: "Scelta della metrica e del protocollo di valutazione",
    6: "Miglioramento reale, rumore e overfitting",
    7: "Registro degli esperimenti",
    8: "Ruoli: modello e codice deterministico",
    9: "Analisi degli errori",
    10: "Integrazione con Kaggle",
    11: "Criteri di stop e budget",
    12: "Riproducibilità",
    13: "Interazione con l'umano",
    14: "Architettura e struttura della repo",
    15: "Piano di implementazione",
    16: "Rischi aperti e domande per l'umano",
}
DECISION_SECTIONS = [2, 5, 6, 7, 8, 10, 11, 14]
MIN_WORDS = 80


def parse_sections(text: str) -> list[tuple[int, str, str]]:
    """Restituisce (numero, titolo, corpo) per ogni header `## N. Titolo`."""
    parts = re.split(r"^## (\d+)\. (.+)$", text, flags=re.M)
    return [
        (int(parts[i]), parts[i + 1].strip(), parts[i + 2])
        for i in range(1, len(parts), 3)
    ]


@pytest.fixture(scope="module")
def sections() -> dict[int, str]:
    assert DESIGN.is_file()
    return {n: body for n, _, body in parse_sections(DESIGN.read_text(encoding="utf-8"))}


def test_design_ha_tutte_le_16_sezioni_in_ordine():
    parsed = parse_sections(DESIGN.read_text(encoding="utf-8"))
    assert [n for n, _, _ in parsed] == list(range(1, 17))
    assert {n: t for n, t, _ in parsed} == SECTION_TITLES


def test_sezioni_con_contenuto_concreto():
    for n, _, body in parse_sections(DESIGN.read_text(encoding="utf-8")):
        assert len(body.split()) >= MIN_WORDS, f"sezione {n} troppo corta"
        assert not re.search(r"\b(TODO|TBD|FIXME|lorem ipsum)\b", body, re.I), n


def test_sezioni_decisionali_hanno_alternative_scartate(sections):
    for n in DECISION_SECTIONS:
        low = sections[n].lower()
        assert "alternativ" in low and "scartat" in low, f"sezione {n}"


PRESCRIPTIVE = [
    r"la metrica (è|sarà) (sempre )?[A-Z]{2,}",
    r"si usa sempre",
    r"\b\d+-?fold\b",
    r"\bsempre\s+\d+\b",
    r"si usa(no)? (sempre )?(lightgbm|xgboost|catboost)",
]


def test_design_non_prescrive_metriche_o_tecniche_fisse(sections):
    text = sections[5] + sections[6]
    for pattern in PRESCRIPTIVE:
        assert not re.search(pattern, text, re.I if "lightgbm" in pattern else 0), pattern
    low = text.lower()
    assert "giustific" in low and "derivazione" in low
    assert "non fissa" in sections[5].lower()


def test_metrica_e_protocollo_congelati(sections):
    low = sections[5].lower()
    assert "congel" in low and "hash" in low
    assert "proposta esplicita" in low and "approvata" in low


def test_stack_punto_di_partenza_non_vincolante(sections):
    for n in (1, 15):
        low = sections[n].lower()
        assert "punto di partenza" in low, n
        assert "non vincolante" in low or "non un vincolo" in low, n
    low1 = sections[1].lower()
    for criterio in ("senza gpu", "a pagamento", "senza rete", "manutenzione"):
        assert criterio in low1, criterio
    assert "alternative" in sections[15].lower()


def test_pyproject_senza_dipendenze_di_runtime_ne_modellazione():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project.get("dependencies", []) == []
    declared = " ".join(
        d.lower() for deps in project.get("optional-dependencies", {}).values() for d in deps
    )
    for lib in ("pandas", "scikit-learn", "lightgbm", "mlflow", "numpy"):
        assert lib not in declared


def test_sottopacchetti_citati_nella_sezione_14(sections):
    pkg = ROOT / "src" / "kaggle_agent"
    subs = [p.name for p in pkg.iterdir() if (p / "__init__.py").is_file()]
    assert subs
    for name in subs:
        assert f"`{name}`" in sections[14], name


def test_readme_con_scopo_e_rimando_al_design():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/design.md" in text
    assert "Kaggle" in text and "data scientist" in text


def test_piano_di_implementazione_ticket_con_criteri_e_fake(sections):
    body = sections[15]
    tickets = re.findall(r"^\d+\. \*\*.+$", body, flags=re.M)
    assert len(tickets) >= 10
    for t in tickets:
        assert "Obiettivo:" in t and "Criteri di accettazione:" in t, t[:50]
    low = body.lower()
    assert "senza rete" in low
    assert "fake" in low and "kaggle" in low and "claude" in low
    assert "simulazion" in low


def test_credenziali_e_protezione_valutazione(sections):
    s10 = sections[10].lower()
    assert "credenziali" in s10 and "kaggle.json" in s10 and "600" in s10
    assert ".gitignore" in s10 and "contesto del modello" in s10
    s8 = sections[8].lower()
    assert "sola lettura" in s8 and "hash" in s8 and "holdout" in s8 and "sandbox" in s8


def test_visione_predict_optimize_come_punto_di_estensione(sections):
    low = sections[14].lower()
    assert "predict + optimize" in low
    assert "uplift" in low and "allocazione" in low
    assert "punto di estensione" in low
    assert "non è progettata" in low
