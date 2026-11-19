# Manual checklist: a real run through the cockpit page

Not part of the automatic suite (it needs a real Kaggle token, a logged-in Claude and the network).
Run it once per release that changes the cockpit, on macOS and on Linux, in a normal Terminal.

## Preparation

1. Pick an empty folder for the project and a small competition you have accepted the rules of.
2. `python -m kaggle_agent.web --project <folder>` and open the printed address in a browser.
3. Optional automatic part: `KAGGLE_AGENT_RUN_COCKPIT_DIR=<folder> .venv/bin/pytest -q tests/test_cockpit_manual.py`
   (it uses the default controller commands, so it starts the real orchestrator).

## Steps and expected screens

| # | Step | Expected screen | Record |
|---|------|-----------------|--------|
| 1 | Open **Environment** | Cards load after "checking"; Kaggle shows only present/missing (never a token); Claude shows logged in, or the fix `claude auth login`; library versions, OpenMP, Python, disk | screenshot, any card that says unknown |
| 2 | Open **Setup**, fill slug, target, metric, limits (rounds 2, candidates 2) and press Validate, then Save | inline errors name the key path; saved file appears as `config.toml`, the old one in `cockpit/config-history/` | the saved config |
| 3 | Press **Preview** and **Profile** | protocol preview (folds, holdout share, hash) and development-only profile appear | hash, row count |
| 4 | **Run control**: Start | state goes to running (stage), then waiting for you (approval of the plan); the ticker in the top bar changes | time to first event |
| 5 | **Live**: watch the timeline | stages with durations, a progress bar per attempt (fold k of N with partial score), call log with tokens, console panel | any gap or missing event |
| 6 | Press **Enable notifications** (click only) | the browser asks for permission once; nothing is asked at page load; tick the sound box for a short beep | browser and answer |
| 7 | **Decisions**: Approve the plan | resume offered (or automatic when the setting is on); exactly one process runs | audit entry |
| 8 | While a call runs press **Pause** | the run finishes the current call, exits with code 14, state "paused"; Resume continues the same cycle | exit text |
| 9 | Press **Stop** (confirm, with reason) | SIGTERM, exit code 15 "interrupted by stop"; no leftover bundle process (`ps`); Resume works | `ps` output |
| 10 | After round 1 give **Guidance** text, then Continue after round 2 | round results beside the box; the guidance appears in the cycle page | notification fired |
| 11 | **Limits**: add one round with a reason and confirmation | effective limits and history shown; the cycle runs the extra round after resume | limits table |
| 12 | **Review**: read the report, Accept (confirm) | state finished; notification | the final page |
| 13 | Open **Audit** | every action above with who, when, what, why, in order; add a note and a tag on a run and search for them | audit export |
| 14 | Restart the web server while a run is going | the page re-attaches by pid and heartbeat, the run is not killed | state after restart |
| 15 | Kill the browser tab and reopen during a run | the SSE stream resumes from the last event id, no gap | |

## What to record

Operating system and browser, the competition, the exit codes seen, the time of each step, any "unknown" card,
any event that is missing from the live view, any process left behind after Stop, and any secret-looking text
seen in the console panel (there must be none).
