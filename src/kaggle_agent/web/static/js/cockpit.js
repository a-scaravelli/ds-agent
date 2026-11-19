/* Forms and buttons of the cockpit: every write is a fetch POST with the CSRF token of the page. */
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
  window.cockpitPost = post;

  function show(node, text, bad) {
    if (!node) { return; }
    node.textContent = text;
    node.className = "note" + (bad ? " error" : "");
  }

  function messageNode() {
    return document.getElementById("decision-message") || document.getElementById("run-message") ||
      document.getElementById("config-message");
  }

  function errorText(result) {
    var e = (result.json || {}).error;
    return e ? e.message : "The request failed (" + result.status + ").";
  }

  // -- buttons with data-post ------------------------------------------------------------------
  document.addEventListener("click", function (event) {
    var button = event.target.closest("button[data-post]");
    if (!button || button.type === "submit") { return; }
    event.preventDefault();
    var body = {};
    try { body = JSON.parse(button.getAttribute("data-body") || "{}"); } catch (e) { body = {}; }
    (button.getAttribute("data-fields") || "").split(",").filter(Boolean).forEach(function (pair) {
      var parts = pair.split(":");
      var input = document.getElementById(parts[0]);
      if (input) { body[parts[1]] = input.value; }
    });
    var reason = button.getAttribute("data-reason");
    if (reason) { body.reason = document.getElementById(reason).value; }
    var confirmId = button.getAttribute("data-confirm");
    if (confirmId) { body.confirm = document.getElementById(confirmId).checked; }
    if (!button.dataset.requestId) { button.dataset.requestId = newId(); }
    body.request_id = button.dataset.requestId;  // a double click sends the same id: saved once
    var jobTarget = button.getAttribute("data-target");
    post(button.getAttribute("data-post"), body).then(function (r) {
      var ok = r.status >= 200 && r.status < 300;
      show(messageNode(), ok ? "Done." : errorText(r), !ok);
      if (!ok) { button.dataset.requestId = ""; return; }
      if (jobTarget) { followJob(button, r.json.data.id, jobTarget); return; }
      if (button.getAttribute("data-after") === "reload") { window.location.reload(); }
    });
  });

  document.addEventListener("click", function (event) {
    var opener = event.target.closest("[data-dialog]");
    if (!opener) { return; }
    var dialog = document.getElementById(opener.getAttribute("data-dialog"));
    if (dialog && dialog.showModal) { dialog.showModal(); }
  });

  var auto = document.getElementById("auto-resume");
  if (auto) {
    auto.addEventListener("change", function () {
      post(auto.getAttribute("data-post"), { auto_resume: auto.checked }).then(function (r) {
        show(document.getElementById("run-message"), r.status === 200 ? "Saved." : errorText(r), r.status !== 200);
      });
    });
  }

  // -- forms ------------------------------------------------------------------------------------
  function lines(text) {
    return text.split("\n").map(function (s) { return s.trim(); }).filter(Boolean);
  }

  var own = document.getElementById("own-plan");
  if (own) {
    own.addEventListener("submit", function (event) {
      event.preventDefault();
      var f = own.elements;
      var body = {
        action: "own", address: parseInt(own.getAttribute("data-address"), 10), reason: f.reason.value,
        confirm: f.confirm.checked, request_id: newId(),
        plan: { metric: f.metric.value, validation: f.validation.value, baseline: f.baseline.value,
                feature_engineering: lines(f.feature_engineering.value), models: lines(f.models.value),
                tasks: lines(f.tasks.value) }
      };
      post(own.getAttribute("data-post"), body).then(function (r) {
        var ok = r.status === 200;
        show(document.getElementById("decision-message"), ok ? "Done." : errorText(r), !ok);
        if (ok) { window.location.reload(); }
      });
    });
  }

  var limits = document.getElementById("limits-form");
  if (limits) {
    limits.addEventListener("submit", function (event) {
      event.preventDefault();
      var f = limits.elements, values = {};
      ["rounds", "candidates", "stagnation"].forEach(function (k) {
        if (f[k].value !== "") { values[k] = parseInt(f[k].value, 10); }
      });
      post(limits.getAttribute("data-post"), { values: values, reason: f.reason.value,
        confirm: f.confirm.checked, request_id: newId() }).then(function (r) {
        var ok = r.status === 200;
        var warn = ok ? (r.json.data.warnings || []).join(" ") : "";
        show(document.getElementById("decision-message"), ok ? "Saved. " + warn : errorText(r), !ok);
        if (ok && !warn) { window.location.reload(); }
      });
    });
  }

  // -- notes and tags ---------------------------------------------------------------------------
  document.querySelectorAll(".note-form").forEach(function (f) {
    var target = { type: f.getAttribute("data-kind"), id: f.getAttribute("data-id") };
    function done(r) { if (r.status === 200) { window.location.reload(); } else { window.alert(errorText(r)); } }
    f.addEventListener("submit", function (event) {
      event.preventDefault();
      if (!f.elements.text.value.trim()) { return; }
      post(f.getAttribute("data-notes-url"), { target: target, op: "add", text: f.elements.text.value }).then(done);
    });
    f.querySelector('[name="save-tags"]').addEventListener("click", function () {
      var tags = f.elements.tags.value.split(",").map(function (t) { return t.trim(); }).filter(Boolean);
      post(f.getAttribute("data-tags-url"), { target: target, tags: tags }).then(done);
    });
  });

  // -- the setup form ---------------------------------------------------------------------------
  var form = document.getElementById("config-form");

  function setPath(root, path, value) {
    var keys = path.split("."), node = root;
    for (var i = 0; i < keys.length - 1; i++) {
      node[keys[i]] = node[keys[i]] || {};
      node = node[keys[i]];
    }
    node[keys[keys.length - 1]] = value;
  }

  function collect() {
    var config = {};
    form.querySelectorAll("input[name], select[name]").forEach(function (input) {
      if (input.disabled || input.readOnly) { return; }
      var value = input.value;
      if (input.type === "number") { value = value === "" ? "" : Number(value); }
      setPath(config, input.name, value);
    });
    return config;
  }

  function showErrors(errors) {
    form.querySelectorAll("[data-error-for]").forEach(function (n) { n.textContent = ""; });
    var general = [];
    errors.forEach(function (e) {
      var node = e.key && form.querySelector('[data-error-for="' + e.key + '"]');
      if (node) { node.textContent = e.message; } else { general.push((e.key ? e.key + ": " : "") + e.message); }
    });
    return general;
  }

  function submitConfig(url) {
    post(url, { config: collect() }).then(function (r) {
      var message = document.getElementById("config-message");
      if (r.status !== 200) { show(message, errorText(r), true); return; }
      var d = r.json.data, general = showErrors(d.errors || []);
      if (d.ok) {
        show(message, d.saved ? "Saved. config.toml was written; the previous copy is in config-history." : "The settings are valid.", false);
      } else {
        show(message, general.join(" ") || "Some settings are not valid: see the messages next to the fields.", true);
      }
    });
  }

  if (form) {
    form.addEventListener("submit", function (event) { event.preventDefault(); submitConfig(form.getAttribute("data-save")); });
    document.getElementById("config-validate").addEventListener("click", function () { submitConfig(form.getAttribute("data-validate")); });
    var k = document.getElementById("threshold-k");
    if (k) {
      k.addEventListener("input", function () { document.getElementById("k-value").textContent = k.value; });
    }
  }

  // -- background jobs: preview and profile -----------------------------------------------------
  function followJob(button, jobId, targetId) {
    var target = document.getElementById(targetId);
    target.textContent = "Working in the background...";
    var url = button.getAttribute("data-job-url").replace("JOB", jobId);
    (function poll() {
      fetch(url, { credentials: "same-origin" }).then(function (r) { return r.json(); }).then(function (j) {
        var job = j.data;
        if (job.status === "running") { setTimeout(poll, 700); return; }
        renderJob(target, job);
      }).catch(function () { target.textContent = "unknown: the job state could not be read."; });
    })();
  }

  function table(rows, head) {
    var t = el("table", {}, [el("thead", {}, [el("tr", {}, head.map(function (h) { return el("th", {}, [h]); }))]),
      el("tbody", {}, rows.map(function (r) { return el("tr", {}, r.map(function (c) { return el("td", {}, [String(c)]); })); }))]);
    return el("div", { "class": "table-wrap" }, [t]);
  }

  function renderJob(target, job) {
    target.textContent = "";
    var r = job.result;
    if (job.status !== "done" || !r || r.ok === false) {
      target.appendChild(el("p", { "class": "error" }, ["unknown: " + (job.error || (r && r.error) || "the job did not produce a result")]));
      return;
    }
    if (r.protocol_hash) {
      target.appendChild(el("dl", { "class": "facts" }, [
        el("dt", {}, ["Protocol hash"]), el("dd", {}, [el("code", {}, [r.protocol_hash])]),
        el("dt", {}, ["Rows"]), el("dd", {}, [r.rows + " (development " + r.development_rows + ", holdout " + r.holdout_rows + ")"]),
        el("dt", {}, ["Holdout share"]), el("dd", {}, [String(r.holdout_share)]),
        el("dt", {}, ["Partitions"]), el("dd", {}, [String(r.partitions)]),
        el("dt", {}, ["The bar"]), el("dd", {}, [r.bar_text])
      ]));
      target.appendChild(table(r.folds.map(function (f) { return [f.id, f.train, f.valid]; }), ["Partition", "Train rows", "Validation rows"]));
      return;
    }
    target.appendChild(el("p", {}, [r.rows + " development rows; duplicates: " + r.duplicate_rows +
      "; constant columns: " + (r.constant_columns.join(", ") || "none") + "; ID-like columns: " + (r.id_like_columns.join(", ") || "none")]));
    target.appendChild(table(r.columns.map(function (c) { return [c.name, c.dtype, c.missing, c.distinct]; }), ["Column", "Type", "Missing", "Distinct"]));
    target.appendChild(el("p", {}, ["Target balance: " + JSON.stringify(r.target_balance)]));
  }

  // -- the environment page ---------------------------------------------------------------------
  var envBox = document.getElementById("env-cards");
  function renderEnv(cards) {
    function put(key, state, lines) {
      var card = document.getElementById("card-" + key);
      if (!card) { return; }
      card.querySelector(".card-state").textContent = state;
      card.querySelector(".card-state").setAttribute("data-state", state);
      var body = card.querySelector(".card-body");
      body.textContent = "";
      lines.forEach(function (l) { body.appendChild(el("p", {}, [l])); });
    }
    var k = cards.kaggle || {}, c = cards.claude || {}, l = cards.libraries || {}, o = cards.openmp || {},
        p = cards.python || {}, d = cards.disk || {};
    put("kaggle", k.state || "unknown", [k.source ? "Found in " + k.source + "." : ""].concat(k.warnings || []).concat(k.steps || []).filter(Boolean));
    put("claude", c.state === "logged_in" ? "logged in" : (c.state === "logged_out" ? "not logged in" : (c.state || "unknown")),
      [c.detail || "", c.fix || ""].filter(Boolean));
    put("libraries", l.state || "unknown", Object.keys(l.versions || {}).map(function (n) { return n + " " + l.versions[n]; }));
    put("openmp", o.state || "unknown", [o.library ? "Library: " + o.library : "No OpenMP library found (LightGBM and XGBoost may be slower or fail on a Mac; `brew install libomp`)."]);
    put("python", p.state || "unknown", [p.version ? "Python " + p.version : "", p.platform || ""].filter(Boolean));
    put("disk", d.state || "unknown", [d.free_gb !== null && d.free_gb !== undefined ? d.free_gb + " GB free of " + d.total_gb + " GB" : "unknown"]);
    var stamp = document.getElementById("env-checked");
    if (stamp && cards.checked_at) { stamp.textContent = "Checked " + new Date(cards.checked_at * 1000).toLocaleTimeString(); }
  }
  function loadEnv(url, tries) {
    fetch(url, { credentials: "same-origin" }).then(function (r) { return r.json(); }).then(function (j) {
      renderEnv(j.data);
      if ((j.data.claude || {}).state === "checking" && tries < 30) { setTimeout(function () { loadEnv(envBox.getAttribute("data-url"), tries + 1); }, 700); }
    }).catch(function () { renderEnv({}); });
  }
  if (envBox) {
    loadEnv(envBox.getAttribute("data-url"), 0);
    document.getElementById("env-refresh").addEventListener("click", function () {
      loadEnv(document.getElementById("env-refresh").getAttribute("data-url"), 0);
    });
  }
})();
