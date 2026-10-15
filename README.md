# kaggle-agent

Agente che fa il lavoro iterativo di un data scientist su competizioni Kaggle tabellari, dal download dei dati alla submission. L'agente propone un passo, un umano lo approva, l'agente lo esegue e analizza i risultati, poi propone il successivo. Il modello che ragiona è Claude Code headless con la subscription (nessuna API a pagamento); gira su Mac senza GPU.

## Stato attuale

Solo design e scheletro: il documento di design è in [docs/design.md](docs/design.md) e il pacchetto `kaggle_agent` contiene sottopacchetti vuoti. Non c'è ancora logica di modellazione.

## Stack

Python 3.12, pandas, scikit-learn, LightGBM e MLflow locale sono un punto di partenza, non un vincolo: le librerie si scelgono ticket per ticket, con i criteri della sezione 1 del design. Per questo `pyproject.toml` non dichiara dipendenze di runtime.

## Installazione e test

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
```

I test non usano la rete.
