/* The forms and buttons of the final steps, the memory across runs and the overrides (ticket 30).
   Every write is a fetch POST with the CSRF token of the page; every text from the server goes in with textContent. */
(function () {
  "use strict";

  var token = (document.querySelector('meta[name="csrf-token"]') || {}).content || "";

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    (children || []).forEach(function (c) {
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return node;
  }

  function newId() {
    if (window.crypto && crypto.randomUUID) { return crypto.randomUUID().replace(/-/g, ""); }
    return String(Date.now()) + String(Math.random()).slice(2, 12);
  }

  function post(url, body) {
    return fetch(url, {
      method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": token },
      body: JSON.stringify(body || {})
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) { return { status: r.status, json: j }; });
    });
  }

  function errorText(result) {
    var e = (result.json || {}).error;
    return e ? e.message : "The request failed (" + result.status + ").";
  }

  function say(id, text, bad) {
    var node = document.getElementById(id);
    if (!node) { return; }
    node.textContent = text;
    node.className = "note" + (bad ? " error" : "");
  }

  function list(text) {
    return text.split(/[\n,]/).map(function (s) { return s.trim(); }).filter(Boolean);
  }

  function checked(form, name) {
    return Array.prototype.slice.call(form.querySelectorAll('input[name="' + name + '"]:checked'))
      .map(function (i) { return i.value; });
  }

  // -- a form that posts one decision --------------------------------------------------------------
  function decide(form, messageId, build, after) {
    if (!form) { return; }
    var requestId = newId();
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var body = build(form);
      body.request_id = requestId;
      post(form.getAttribute("data-url"), body).then(function (r) {
        var ok = r.status >= 200 && r.status < 300;
        if (!ok) { requestId = newId(); say(messageId, errorText(r), true); return; }
        var warnings = (r.json.data || {}).warnings || [];
        say(messageId, "Saved. " + warnings.join(" "), false);
        if (after) { after(r.json.data); }
      });
    });
  }

  var steering = document.getElementById("steering-form");
  decide(steering, "steering-message", function (f) {
    return { steering: {
      focus: f.elements.focus.value || null,
      pin_features: list(f.elements.pin_features.value), avoid_features: list(f.elements.avoid_features.value),
      try_families: checked(f, "try_families"), avoid_families: checked(f, "avoid_families"),
      allowed_model_families: checked(f, "allowed_model_families"), notes: f.elements.notes.value } };
  });
  var clear = document.getElementById("steering-clear");
  if (clear) {
    clear.addEventListener("click", function () {
      post(steering.getAttribute("data-url"), { clear: true, request_id: newId() }).then(function (r) {
        say("steering-message", r.status === 200 ? "The steering was cleared." : errorText(r), r.status !== 200);
      });
    });
  }

  decide(document.getElementById("try-form"), "try-message", function (f) {
    return { request: { hypothesis: f.elements.hypothesis.value, constraints: list(f.elements.constraints.value) } };
  });

  var bar = document.getElementById("bar-panel");
  decide(document.getElementById("bar-form"), "bar-message", function (f) {
    return { scope: f.elements.scope.value, round: parseInt(bar.getAttribute("data-round"), 10),
             attempt: bar.getAttribute("data-candidate"), reason: f.elements.reason.value,
             confirm: f.elements.confirm.checked };
  });

  var choose = document.getElementById("choose-form");
  decide(choose, "choose-message", function (f) {
    return { attempt: choose.getAttribute("data-attempt"), reason: f.elements.reason.value,
             confirm: f.elements.confirm.checked };
  });

  // -- background jobs -------------------------------------------------------------------------------
  function pollJob(urlTemplate, jobId, done) {
    var url = urlTemplate.replace("JOB", jobId);
    (function poll() {
      fetch(url, { credentials: "same-origin" }).then(function (r) { return r.json(); }).then(function (j) {
        if (j.data.status === "running") { setTimeout(poll, 600); return; }
        done(j.data);
      }).catch(function () { done({ status: "failed", error: "the job state could not be read", result: null }); });
    })();
  }

  function facts(rows) {
    var dl = el("dl", { "class": "facts" }, []);
    rows.forEach(function (r) {
      dl.appendChild(el("dt", {}, [r[0]]));
      dl.appendChild(el("dd", {}, [String(r[1])]));
    });
    return dl;
  }

  function renderJob(kind, target, job) {
    target.textContent = "";
    var r = job.result;
    if (job.status !== "done" || !r) {
      target.appendChild(el("p", { "class": "error" }, ["The job did not finish: " + (job.error || "no result")]));
      return;
    }
    if (kind === "reproduce") {
      target.appendChild(el("p", { "class": r.ok ? "note" : "error" }, [r.ok ? "OK: the stored code reproduces the reported metric." : "Mismatch"]));
      target.appendChild(el("pre", {}, [(r.lines || [r.error || ""]).join("\n")]));
      return;
    }
    if (kind === "submit-preview" || kind === "submit-confirm") {
      var d = r.data;
      if (d && d.sha256) {
        var rows = [["File", d.file], ["sha256", d.sha256], ["Rows", d.rows], ["Columns", d.columns.join(", ")],
          ["Message", d.message]];
        if (d.daily_limit) { rows.push(["Daily limit", d.daily_limit.left + " of " + d.daily_limit.limit + " left (resets " + d.daily_limit.resets_at + ")"]); }
        var box = facts(rows);
        box.id = "preview-facts";
        box.setAttribute("data-sha", d.sha256);
        target.appendChild(box);
      }
      target.appendChild(el("pre", {}, [(r.lines || []).join("\n")]));
      target.appendChild(el("p", { "class": "note" }, ["Exit code " + r.code]));
      return;
    }
    target.appendChild(el("p", { "class": r.ok === false ? "error" : "note" }, [r.ok === false ? (r.error || "failed") : "Done."]));
    if (r.attempt) { target.appendChild(el("p", {}, ["New attempt: " + r.attempt + (r.score !== undefined && r.score !== null ? ", score " + r.score : "")])); }
  }

  document.addEventListener("click", function (event) {
    var button = event.target.closest("button[data-job-post]");
    if (!button) { return; }
    event.preventDefault();
    var target = document.getElementById(button.getAttribute("data-target"));
    target.textContent = "Working in the background...";
    post(button.getAttribute("data-job-post"), { request_id: newId() }).then(function (r) {
      if (r.status !== 202) { target.textContent = ""; target.appendChild(el("p", { "class": "error" }, [errorText(r)])); return; }
      pollJob(button.getAttribute("data-job-url"), r.json.data.job.id, function (job) {
        renderJob(button.getAttribute("data-render"), target, job);
      });
    });
  });

  // -- re-run ------------------------------------------------------------------------------------------
  var rerun = document.getElementById("rerun-form");
  if (rerun) {
    rerun.addEventListener("submit", function (event) {
      event.preventDefault();
      var target = document.getElementById("rerun-result");
      post(rerun.getAttribute("data-url"), { attempt: rerun.getAttribute("data-attempt"),
        seed_offset: parseInt(rerun.elements.seed_offset.value, 10), request_id: newId() }).then(function (r) {
        if (r.status !== 202) { say("rerun-message", errorText(r), true); return; }
        say("rerun-message", "Running in the background...", false);
        pollJob(rerun.getAttribute("data-job-url"), r.json.data.job.id, function (job) { renderJob("rerun", target, job); });
      });
    });
  }

  // -- predict upload ------------------------------------------------------------------------------------
  var predict = document.getElementById("predict-form");
  if (predict) {
    predict.addEventListener("submit", function (event) {
      event.preventDefault();
      var file = predict.elements.file.files[0];
      if (!file) { say("predict-message", "Choose a CSV file first.", true); return; }
      var form = new FormData();
      form.append("file", file);
      fetch(predict.getAttribute("data-url"), { method: "POST", credentials: "same-origin",
        headers: { "X-CSRF-Token": token }, body: form }).then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (j) { return { status: r.status, json: j }; });
      }).then(function (r) {
        if (r.status !== 202) { say("predict-message", errorText(r), true); return; }
        var warn = (r.json.data.warnings || []).join(" ");
        say("predict-message", "Predicting " + r.json.data.rows + " rows... " + warn, false);
        var jobId = r.json.data.job.id;
        pollJob(predict.getAttribute("data-job-url"), jobId, function (job) {
          var target = document.getElementById("predict-result");
          target.textContent = "";
          if (job.status !== "done" || !job.result || job.result.ok === false) {
            target.appendChild(el("p", { "class": "error" }, [job.error || (job.result && job.result.error) || "failed"]));
            return;
          }
          target.appendChild(el("p", {}, [job.result.rows + " predictions."]));
          target.appendChild(el("a", { "class": "button", href: predict.getAttribute("data-csv-url").replace("JOB", jobId), download: "" }, ["Download the predictions (CSV)"]));
        });
      });
    });
  }

  // -- the submission: confirm dialog and the recovery forms -----------------------------------------------
  var go = document.getElementById("send-go");
  if (go) {
    go.addEventListener("click", function (event) {
      event.preventDefault();
      var facts_ = document.getElementById("preview-facts");
      if (!facts_) { say("submission-message", "Preview the submission first: the file to send is the previewed one.", true); return; }
      post(go.getAttribute("data-url"), { sha256: facts_.getAttribute("data-sha"),
        typed: document.getElementById("send-typed").value, confirm: document.getElementById("send-confirm").checked,
        request_id: newId() }).then(function (r) {
        if (r.status !== 202) { say("submission-message", errorText(r), true); return; }
        var target = document.getElementById("send-result");
        target.textContent = "Sending...";
        pollJob(go.getAttribute("data-job-url"), r.json.data.job.id, function (job) { renderJob("submit-confirm", target, job); });
      });
    });
  }

  function recover(formId, build) {
    var form = document.getElementById(formId);
    if (!form) { return; }
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var body = build(form);
      body.request_id = newId();
      post(form.getAttribute("data-url"), body).then(function (r) {
        if (r.status !== 202) { say("submission-message", errorText(r), true); return; }
        var target = document.getElementById("send-result");
        target.textContent = "Working...";
        pollJob(form.getAttribute("data-job-url"), r.json.data.job.id, function (job) { renderJob("submit-confirm", target, job); });
      });
    });
  }
  recover("recover-not-sent", function (f) {
    return { action: "not_sent", reason: f.elements.reason.value, confirm: f.elements.confirm.checked };
  });
  recover("recover-adopt", function (f) {
    return { action: "adopt", submission_id: f.elements.submission_id.value, confirm: f.elements.confirm.checked };
  });

  // -- fork --------------------------------------------------------------------------------------------------
  var fork = document.getElementById("fork-form");
  if (fork) {
    var attempts = JSON.parse(fork.getAttribute("data-attempts") || "{}");
    var fill = function () {
      var select = fork.elements.from_attempt;
      select.textContent = "";
      select.appendChild(el("option", { value: "" }, ["(the run as a whole)"]));
      (attempts[fork.elements.from_run.value] || []).forEach(function (a) {
        select.appendChild(el("option", { value: a.id }, [a.id + " (" + (a.mean === null ? "no score" : a.mean) + ")"]));
      });
    };
    fork.elements.from_run.addEventListener("change", fill);
    fill();
    decide(fork, "fork-message", function (f) {
      var body = { from_run: f.elements.from_run.value, confirm: f.elements.confirm.checked };
      if (f.elements.from_attempt.value) { body.from_attempt = f.elements.from_attempt.value; }
      if (checked(f, "protocol")[0] === "new" || f.elements.protocol.value === "new") {
        body.threshold_k = parseFloat(f.elements.threshold_k.value);
        body.reason = f.elements.reason.value;
      }
      return body;
    });
  }

  // -- the note of the competition (learnings page) ----------------------------------------------------------
  var note = document.getElementById("competition-note");
  if (note) {
    var target = { type: "competition", id: note.getAttribute("data-id") };
    note.addEventListener("submit", function (event) {
      event.preventDefault();
      post(note.getAttribute("data-notes-url"), { target: target, op: "add", text: note.elements.text.value }).then(function (r) {
        say("decision-message", r.status === 200 ? "Saved." : errorText(r), r.status !== 200);
        if (r.status === 200) { window.location.reload(); }
      });
    });
    note.querySelector('button[name="save-tags"]').addEventListener("click", function () {
      post(note.getAttribute("data-tags-url"), { target: target, tags: list(note.elements.tags.value) }).then(function (r) {
        say("decision-message", r.status === 200 ? "Saved." : errorText(r), r.status !== 200);
      });
    });
  }
})();
