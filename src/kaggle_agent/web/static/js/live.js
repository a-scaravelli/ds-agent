/* Live progress: the ticker of the status bar, the timeline, the progress bars, the call log and the console.
   One EventSource per page (reconnects with Last-Event-ID); when it cannot be kept open the page polls. */
(function () {
  "use strict";

  var ticker = document.getElementById("live-ticker");
  if (!ticker) { return; }
  var chip = document.getElementById("run-chip");
  var streamUrl = ticker.getAttribute("data-stream-url");
  var eventsUrl = ticker.getAttribute("data-events-url");
  var runUrl = ticker.getAttribute("data-run-url");
  var consoleUrl = ticker.getAttribute("data-console-url");
  var timeline = document.getElementById("timeline");
  var now = document.getElementById("live-now");
  var progress = document.getElementById("progress-list");
  var calls = document.querySelector("#call-log tbody");
  var consoleBox = document.getElementById("console");

  var lastId = 0, source = null, pollTimer = null, seen = {}, open = {}, bars = {};

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    (children || []).forEach(function (c) { node.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return node;
  }

  function clock(iso) {
    var d = new Date(iso);
    return isNaN(d) ? "" : d.toLocaleTimeString();
  }

  function seconds(v) { return typeof v === "number" ? v.toFixed(1) + " s" : ""; }

  function describe(e) {
    var d = e.data || {};
    switch (e.type) {
      case "run_start": return "Run started";
      case "cycle_state": return "State: " + d.state;
      case "stage_start": return "Started " + (d.role || d.stage) + (d.detail ? " (" + d.detail + ")" : "");
      case "stage_end": return "Finished " + (d.role || d.stage) + " (" + (d.outcome || "ok") + ") " + seconds(d.seconds);
      case "round_start": return "Round opened: " + d.round;
      case "candidate_start": return "Candidate " + d.attempt + " registered";
      case "candidate_end": return "Candidate " + d.attempt + " ended: " + (d.outcome || "unknown");
      case "fold_start": return "Fold " + d.fold + " of " + d.folds + " started";
      case "fold_end": return "Fold " + d.fold + " of " + d.folds + " done" + (d.score !== undefined ? ", score " + d.score : "");
      case "llm_call": return "Claude call (" + d.role + "): " + d.outcome + " " + seconds(d.seconds);
      case "guidance_recorded": return "Guidance recorded for " + d.round;
      case "waiting": return "Waiting for you: " + (d.question || d.state || "a decision");
      case "suspended": return "Suspended (" + d.kind + ")";
      case "error": return "Error: " + (d.kind || d.type);
      case "finished": return "Finished: " + (d.reason || "");
      case "paused": return "Paused at a safe point";
      case "stopped": return "Stopped";
      case "decision_applied": return "Your decision was applied (" + d.kind + ")";
      case "decision_stale": return "A decision was ignored (" + (d.reason || "stale") + ")";
      case "limits_applied": return "Limits now: rounds " + d.rounds + ", candidates " + d.candidates + ", stagnation " + d.stagnation;
      default: return e.type;
    }
  }

  function handle(e) {
    if (seen[e.id]) { return; }
    seen[e.id] = true;
    lastId = Math.max(lastId, e.id);
    var text = describe(e);
    ticker.textContent = text;
    if (now) { now.textContent = text; }
    if (e.type === "stage_start") { open[(e.data.role || e.data.stage)] = e; }
    if (e.type === "stage_end") { delete open[(e.data.role || e.data.stage)]; }
    if (timeline) {
      var item = el("li", { "class": "ev-" + e.type }, [el("time", {}, [clock(e.at)]), " " + text]);
      timeline.insertBefore(item, timeline.firstChild);
      var limit = parseInt(timeline.getAttribute("data-limit") || "80", 10);
      while (timeline.children.length > limit) { timeline.removeChild(timeline.lastChild); }
    }
    if (calls && e.type === "llm_call") {
      var d = e.data;
      var row = el("tr", {}, [el("td", {}, [clock(e.at)]), el("td", {}, [d.role || ""]), el("td", {}, [d.outcome || ""]),
        el("td", { "class": "num" }, [d.input_tokens === undefined ? "-" : String(d.input_tokens)]),
        el("td", { "class": "num" }, [d.output_tokens === undefined ? "-" : String(d.output_tokens)]),
        el("td", { "class": "num" }, [seconds(d.seconds)])]);
      calls.insertBefore(row, calls.firstChild);
    }
    if (progress && (e.type === "fold_start" || e.type === "fold_end")) { updateBar(e); }
    if (["waiting", "suspended", "error", "finished", "paused", "stopped", "run_start"].indexOf(e.type) >= 0) { refreshRun(); }
    document.dispatchEvent(new CustomEvent("cockpit:event", { detail: e }));
  }

  function updateBar(e) {
    var d = e.data, key = d.bundle || "attempt";
    var bar = bars[key];
    if (!bar) {
      var meter = el("progress", { max: String(d.folds || 1), value: "0", "aria-label": "folds of " + key });
      var label = el("span", {}, []);
      var scores = el("span", { "class": "muted" }, []);
      bar = bars[key] = { meter: meter, label: label, scores: scores, done: 0, list: [] };
      progress.appendChild(el("div", { "class": "bar-row" }, [el("code", {}, [key]), meter, label, scores]));
    }
    bar.meter.max = d.folds || 1;
    if (e.type === "fold_end") {
      bar.done = d.fold;
      if (d.score !== undefined) { bar.list.push(d.score); }
    }
    bar.meter.value = bar.done;
    bar.label.textContent = " fold " + bar.done + " of " + (d.folds || "?");
    bar.scores.textContent = bar.list.length ? " partial scores: " + bar.list.join(", ") : "";
  }

  function refreshRun() {
    fetch(runUrl, { credentials: "same-origin" }).then(function (r) { return r.json(); }).then(function (j) {
      var run = j.data.run;
      if (chip) { chip.textContent = run.label || run.state; chip.className = "badge status-" + run.state; }
      document.dispatchEvent(new CustomEvent("cockpit:run", { detail: run }));
    }).catch(function () {});
  }

  function poll() {
    fetch(eventsUrl + "?after=" + lastId + "&limit=200", { credentials: "same-origin" })
      .then(function (r) { return r.json(); }).then(function (j) { j.data.events.forEach(handle); }).catch(function () {});
  }

  function startPolling() {
    if (pollTimer) { return; }
    pollTimer = setInterval(poll, 3000);
    poll();
    setTimeout(function () { stopPolling(); connect(); }, 30000);  // try the stream again later
  }
  function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

  function connect() {
    if (!window.EventSource) { startPolling(); return; }
    source = new EventSource(streamUrl + (lastId ? "?after=" + lastId : ""));
    source.onopen = stopPolling;
    source.onmessage = function (m) { handle(JSON.parse(m.data)); };
    var types = ["run_start", "cycle_state", "stage_start", "stage_end", "round_start", "candidate_start", "candidate_end",
      "fold_start", "fold_end", "llm_call", "guidance_recorded", "waiting", "suspended", "error", "finished", "paused",
      "stopped", "decision_applied", "decision_stale", "limits_applied"];
    types.forEach(function (t) { source.addEventListener(t, function (m) { handle(JSON.parse(m.data)); }); });
    source.onerror = function () {
      if (source.readyState === 2) { source.close(); startPolling(); }  // refused (429) or gone: poll instead
    };
  }

  function pollConsole() {
    fetch(consoleUrl + "?tail=8192", { credentials: "same-origin" }).then(function (r) { return r.json(); }).then(function (j) {
      consoleBox.textContent = j.data.text || "(empty)";
      consoleBox.scrollTop = consoleBox.scrollHeight;
    }).catch(function () {});
  }

  refreshRun();
  setInterval(refreshRun, 10000);
  connect();
  if (consoleBox) { pollConsole(); setInterval(pollConsole, 3000); }
})();
