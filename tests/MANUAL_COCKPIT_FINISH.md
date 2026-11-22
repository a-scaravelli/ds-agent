# Manual checklist: the end of the journey in the cockpit (ticket 30)

Run on a real project folder (a finished or a stopped run), on Linux and on macOS.

    KAGGLE_AGENT_RUN_COCKPIT_DIR=<project folder> .venv/bin/pytest -q tests/test_cockpit_finish_manual.py

Open the page with `python -m kaggle_agent.web --db <folder>/registry.db --project <folder>` and go through the steps.

1. **Sealed holdout.** The report page says the holdout is sealed. No holdout number is in the page, the Markdown export or the print view.
2. **Unseal at review.** At the review, "Unseal and read" shows the number, one audit line and one `cockpit/holdout-reads.jsonl` record. A second read asks for a reason and the run is labelled "holdout re-read".
3. **Report export and print.** Download the Markdown export; open the print view and use the browser's "Save as PDF". No external asset is requested.
4. **Reproduce.** The Reproduce button runs as a background job and shows OK, or the exact mismatch.
5. **Predict upload.** Upload a small CSV with the training columns: one prediction per row. A CSV with a missing column is refused and the column is named.
6. **Dry-run submission (real).** Preview shows file, sha256, rows, columns, message and the real daily limit left. Nothing is sent. A confirmed send is out of scope by default: do it only on purpose, on a competition where it is wanted.
7. **Fork with carry-over.** "Start from run n / attempt m" with the same protocol, then with a new threshold_k and a reason: the new protocol is labelled. The first round of the new cycle shows the carry-over in its trace.
8. **Steering.** Pin a feature, set a focus, limit the families: the next round's traces show the typed steering; the run is labelled "steered by the DS".
9. **Try this.** Write a hypothesis while the cycle waits: candidate 1 of the next round carries it.
10. **Bar override.** Accept a candidate the strict bar rejected, with a reason: the numbers are shown first and the report says "bar overridden by the DS".
11. **Choose the model.** Pick another attempt with a reason: the report says "chosen by the DS" and shows the program's own pick.
12. **Look.** Light and dark theme; narrow width (about 360 px) and 1280 px; keyboard focus on the dialogs; macOS and Linux.
13. **Token.** No page, job file or console line ever shows the Kaggle token.
