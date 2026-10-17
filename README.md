# kaggle-agent

Agente che fa il lavoro iterativo di un data scientist su competizioni Kaggle tabellari, dal download dei dati alla submission. L'utente approva solo un piano iniziale breve; poi l'agente lavora in autonomia (split validation/test holdout, baseline, 10 iterazioni di miglioramento per default, con stop anticipato) e l'utente interviene a fine ciclo: accetta il report oppure dà suggerimenti, che avviano un nuovo ciclo dallo stato esistente. Il modello che ragiona è Claude Code headless con la subscription (nessuna API a pagamento); gira su Mac senza GPU.

## Stato attuale

Solo design e scheletro: il documento di design è in [docs/design.md](docs/design.md) e il pacchetto `kaggle_agent` contiene sottopacchetti vuoti. Non c'è ancora logica di modellazione.

## Stack

Python 3.12, pandas, scikit-learn, LightGBM e MLflow locale sono un punto di partenza, non un vincolo: le librerie si scelgono ticket per ticket, con i criteri della sezione 1 del design. Per questo `pyproject.toml` non dichiara dipendenze di runtime.

## Deliverable e tracciabilità

L'output di ogni ciclo è una pipeline Python end-to-end eseguibile con un solo comando e riproducibile (o più pipeline, per confrontare varianti), collegata al tentativo da cui nasce. Ogni tentativo registra ipotesi e motivazione, tentativo di origine e cosa cambia, configurazione, metriche, durata, esito e artefatti; un'interfaccia web (lista/timeline, grafico della metrica, dettaglio, suggerimenti e cicli) li rende consultabili senza leggere codice. Il design grafico curato è rinviato a un ticket successivo.

## Installazione e test

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
```

I test non usano la rete.

## Struttura

- `docs/design.md`: design in 16 sezioni (ciclo, valutazione congelata, registro, Kaggle, piano dei ticket).
- `src/kaggle_agent/`: pacchetto con i sottopacchetti vuoti `loop`, `proposals`, `evaluation`, `registry`, `kaggle`, `llm`, `sandbox`, `review`, `budget`.
- `tests/`: controllano che il design contenga le 16 sezioni, che il pacchetto si importi e che non ci sia logica o dipendenze di terze parti nel codice.

## Limiti noti

- Non c'è ancora nessun comando eseguibile dell'agente: il pacchetto è solo uno scheletro.
- I test sul design verificano struttura e parole chiave, non la qualità del contenuto. Il test sull'assenza di dipendenze di terze parti andrà rilassato quando arriveranno i ticket di sviluppo.
