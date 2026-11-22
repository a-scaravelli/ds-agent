# Manual checklist: the performance analytics of the cockpit

Not part of the automatic suite beyond what is marked. Run it once per release that changes the analytics, on macOS
and on Linux, in a normal Terminal, over a project folder that has finished a run with the current version (a real
Kaggle token and a logged-in Claude are needed to make one; the checks below only read it).

## Preparation

1. `python -m kaggle_agent.web --project <folder>` and open the printed address in a browser.
2. Optional automatic part: `KAGGLE_AGENT_RUN_COCKPIT_DIR=<folder> .venv/bin/pytest -q tests/test_cockpit_analytics_manual.py`
   (every analytics page and API over the real folder must answer 200).
3. Real child processes: the tests marked `real_process` (`pytest -m real_process`) start the real sandbox and the
   importance job. Run them on macOS as well as on Linux:
   `test_the_importance_job_runs_through_the_controller_and_the_sandbox_with_progress_events`,
   `test_a_running_job_never_blocks_the_pages_and_shows_its_progress`.

## Steps and expected screens

| # | Step | Expected screen | Record |
|---|------|-----------------|--------|
| 1 | Open an attempt, then **Performance analytics** | sections Folds and stability, Metric panels, Errors by segment, Hardest rows, Leakage and sanity flags, Features; each has a plain-words explanation; the page shows at once, heavy sections say "Loading" and then fill in | screenshot |
| 2 | Look at **Folds and stability** | one dot per fold, the mean line, the worst fold named; with 2 or more repetitions a score per repetition; a warning appears when the gain over the previous best is inside the noise | the warning text, if any |
| 3 | Look at **Metric panels** on a probabilities run and on a labels-only run | ROC, precision-recall, calibration, log loss, threshold sweep and the confusion matrix; on a labels-only run only the confusion matrix and the reason "labels only" for the rest | both screens |
| 4 | Move the **minimum segment size** slider | the table refreshes after a short pause; small segments disappear, the order of the rest does not change | slider values tried |
| 5 | **Hardest rows**: change Sort by and Order | rows come back sorted; only development features appear | any odd column |
| 6 | **Features**: press **Compute permutation importance** | a progress bar and a percentage; the page stays usable; when the job ends the importance appears by family; the run, if one is live, is only slightly slower | job duration |
| 7 | Open **Compare and history** from the run navigation | score history chart (best so far, baseline, improvement bar), the all-attempts table (sort by the column links, filter), the ablation table (display only) | screenshot |
| 8 | Fill **Side by side** with two or three attempt ids | hypothesis, features, hyperparameters and per-fold scores; "not provided" for an attempt without describe.json; no code | screenshot |
| 9 | Open a paired comparison (`/runs/<id>/paired?a=..&b=..`) | differences fold by fold, interval, wins, the verdict in words, the program's rule beside it and why they may disagree; the correlation fills in | the verdict |
| 10 | Open an attempt of an OLD project (before this version) | "Not recorded for this attempt" with the reason; the per-fold scores still show | screenshot |
| 11 | Widths: repeat steps 1, 7 and 9 at 1280 px and at a narrow width (about 400 px) | tables scroll inside their frame, grids collapse to one column, nothing is cut off; the navigation moves to the top | both widths |
| 12 | Theme: switch to the dark theme and back with the toggle of the status bar | charts are redrawn with the dark tokens; text stays readable | both themes |
| 13 | Holdout: open the browser network panel while using every page | no request mentions the holdout; the analytics only use validation data | the list of requests |
