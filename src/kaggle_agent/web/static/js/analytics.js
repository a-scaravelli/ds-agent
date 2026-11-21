/* Performance analytics pages (ticket 29): fills the heavy sections from the GET API after the page is shown, so a
   slow section never blocks rendering. Everything is built with textContent (never innerHTML); the colours of the
   charts come from the design tokens through charts.js. Development data only: no endpoint used here touches the
   holdout. */
(function () {
  "use strict";

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    (children || []).forEach(function (c) {
      if (c === null || c === undefined) { return; }
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return node;
  }

  function fmt(v, digits) {
    if (v === null || v === undefined) { return "-"; }
    if (typeof v === "number") { return Number(v.toPrecision(digits || 4)).toString(); }
    return String(v);
  }

  function get(url) {
    return fetch(url, { headers: { Accept: "application/json" }, credentials: "same-origin" })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, json: j }; }); });
  }

  function notRecorded(reason) {
    return el("div", { "class": "empty not-recorded", role: "status" },
      [el("strong", {}, ["Not recorded for this attempt"]), reason]);
  }

  function explain(text) { return el("p", { "class": "explain muted" }, [text]); }

  function table(head, rows, numeric) {
    var thead = el("tr", {}, head.map(function (h, i) {
      return el("th", { "class": numeric && numeric.indexOf(i) >= 0 ? "num" : "" }, [h]);
    }));
    var body = rows.map(function (row) {
      return el("tr", {}, row.map(function (cell, i) {
        return el("td", { "class": numeric && numeric.indexOf(i) >= 0 ? "num" : "" }, [cell === null || cell === undefined ? "-" : cell]);
      }));
    });
    return el("div", { "class": "table-wrap" }, [el("table", {}, [el("thead", {}, [thead]), el("tbody", {}, body)])]);
  }

  function chartBox(spec) {
    var box = el("div", { "class": "chart", role: "img", "aria-label": spec.ylabel + " against " + spec.xlabel });
    setTimeout(function () { window.analyticsCharts.plot(box, spec); }, 0);
    return box;
  }

  function bars(labels, values, max) {
    return el("div", { "class": "bars" }, labels.map(function (label, i) {
      return el("div", { "class": "bar-row" }, [
        el("span", { "class": "bar-label" }, [label]),
        el("progress", { max: String(max || 1), value: String(Math.max(0, values[i])) }, []),
        el("span", { "class": "muted" }, [fmt(values[i])])
      ]);
    }));
  }

  // -- renderers --------------------------------------------------------------------------------

  var render = {};

  render.folds = function (d) {
    if (d.not_recorded) { return [notRecorded(d.reason)]; }
    var ids = d.points.map(function (p, i) { return i + 1; });
    var out = [chartBox({
      x: ids, xlabel: "partition (repetition / fold)", ylabel: "score", height: 260,
      series: [{ label: "partition score", y: d.points.map(function (p) { return p.score; }), kind: "points", size: 10 },
               { label: "mean", y: d.points.map(function () { return d.mean; }), token: "--chart-baseline", dash: [6, 4] }]
    })];
    if (d.stability && d.stability.warning) {
      out.push(el("div", { "class": "warnings", role: "status" }, [d.stability.warning]));
    }
    if (d.warning) { out.push(el("div", { "class": "warnings", role: "status" }, [d.warning])); }
    return out;
  };

  function classBlock(title, panel, builder) {
    var box = el("section", { "class": "subpanel" }, [el("h3", {}, [title])]);
    if (!panel || panel.available === false) {
      box.appendChild(notRecorded(panel ? panel.reason : "no data"));
      return box;
    }
    box.appendChild(explain(panel.explanation || ""));
    builder(box, panel);
    return box;
  }

  function curveSpec(panel, xlabel, ylabel, diagonal) {
    var grid = []; var i;
    for (i = 0; i <= 100; i += 1) { grid.push(i / 100); }
    var curves = panel.one_vs_rest || [{ "class": "", x: panel.x, y: panel.y }];
    var series = curves.map(function (c) {
      var xs = c.x.slice(); var ys = c.y.slice();
      if (xs.length > 1 && xs[0] > xs[xs.length - 1]) { xs.reverse(); ys.reverse(); }
      return { label: c["class"] ? "class " + c["class"] : ylabel, y: window.analyticsCharts.resample(xs, ys, grid) };
    });
    if (diagonal) { series.push({ label: "chance", y: grid.slice(), token: "--chart-baseline", dash: [6, 4] }); }
    return { x: grid, xlabel: xlabel, ylabel: ylabel, series: series, xrange: [0, 1], yrange: [0, 1] };
  }

  render.panels = function (d) {
    if (d.not_recorded) { return [notRecorded(d.reason)]; }
    if (d.available === false) { return [notRecorded(d.reason)]; }
    var out = [];
    if (d.task === "regression") {
      out.push(el("p", {}, ["Bias " + fmt(d.bias) + ", RMSE " + fmt(d.rmse) + ", MAE " + fmt(d.mae) + " (repetition " + d.repetition + ", " + d.rows + " rows)."]));
      out.push(explain(d.explanations.bias));
      var h = d.histogram;
      var mid = h.counts.map(function (c, i) { return fmt((h.edges[i] + h.edges[i + 1]) / 2, 3); });
      out.push(el("h3", {}, ["Residuals"]), explain(d.explanations.residuals),
        bars(mid, h.counts, Math.max.apply(null, h.counts) || 1));
      out.push(el("h3", {}, ["Residual against prediction"]), explain(d.explanations.scatter),
        chartBox({ x: d.scatter.x, xlabel: "prediction", ylabel: "residual", series: [{ label: "residual", y: d.scatter.y, kind: "points", size: 5 }] }));
      out.push(el("h3", {}, ["Error by fifth of the truth"]), explain(d.explanations.quantiles),
        table(["Fifth", "Rows", "Truth from", "Truth to", "Mean abs error", "Bias"],
          d.by_quantile.map(function (q) { return [String(q.quantile + 1), String(q.count), fmt(q.truth_min), fmt(q.truth_max), fmt(q.mean_abs_error), fmt(q.bias)]; }), [1, 2, 3, 4, 5]));
      return out;
    }
    out.push(el("p", { "class": "muted" }, ["Repetition " + d.repetition + ", " + d.rows + " rows; classes " + d.classes.join(", ") + "."]));
    out.push(classBlock("Confusion matrix", d.confusion, function (box, p) {
      var rows = p.matrix.map(function (r, i) { return [p.labels[i]].concat(r.map(String)); });
      box.appendChild(table(["true \\ predicted"].concat(p.labels), rows, p.labels.map(function (l, i) { return i + 1; })));
    }));
    out.push(classBlock("ROC", d.roc, function (box, p) {
      if (p.auc !== undefined) { box.appendChild(el("p", {}, ["AUC " + fmt(p.auc)])); }
      box.appendChild(chartBox(curveSpec(p, "false positive rate", "true positive rate", true)));
    }));
    out.push(classBlock("Precision and recall", d.pr, function (box, p) {
      if (p.average_precision !== undefined) { box.appendChild(el("p", {}, ["Average precision " + fmt(p.average_precision)])); }
      box.appendChild(chartBox(curveSpec(p, "recall", "precision", false)));
    }));
    out.push(classBlock("Calibration", d.calibration, function (box, p) {
      box.appendChild(el("p", {}, ["Expected calibration error " + fmt(p.ece)]));
      var xs = p.bins.map(function (b) { return b.mean_predicted; });
      box.appendChild(chartBox({ x: xs, xlabel: "mean predicted probability", ylabel: "fraction positive", xrange: [0, 1], yrange: [0, 1],
        series: [{ label: "observed", y: p.bins.map(function (b) { return b.fraction_positive; }), kind: "points", size: 9 },
                 { label: "perfect", y: xs.slice(), token: "--chart-baseline", dash: [6, 4] }] }));
    }));
    out.push(classBlock("Log loss", d.log_loss, function (box, p) { box.appendChild(el("p", { "class": "big" }, [fmt(p.value)])); }));
    out.push(classBlock("Threshold sweep", d.sweep, function (box, p) {
      box.appendChild(table(["Threshold", "Precision", "Recall", "F1", "Accuracy", "Share flagged"],
        p.rows.map(function (r) { return [fmt(r.threshold), fmt(r.precision), fmt(r.recall), fmt(r.f1), fmt(r.accuracy), fmt(r.flagged)]; }), [0, 1, 2, 3, 4, 5]));
    }));
    return out;
  };

  function segmentRows(list) {
    return table(["Feature", "Segment", "Rows", "Error", "Lift vs overall"],
      list.map(function (s) { return [s.feature, s.segment, String(s.count), fmt(s.error), fmt(s.lift, 3)]; }), [2, 3, 4]);
  }

  render.segments = function (d) {
    if (d.not_recorded) { return [notRecorded(d.reason)]; }
    return [
      el("p", {}, ["Overall error " + fmt(d.overall_error) + " over " + d.rows + " rows; showing segments of at least " + d.min_size + " rows."]),
      el("h3", {}, ["Single features, worst first"]),
      d.segments.length ? segmentRows(d.segments) : el("p", { "class": "muted" }, ["No segment is that large."]),
      el("h3", {}, ["Two features together"]),
      d.interactions.length ? segmentRows(d.interactions) : el("p", { "class": "muted" }, ["No combination is that large."])
    ];
  };

  render.hardest = function (d) {
    if (d.not_recorded) { return [notRecorded(d.reason)]; }
    var cols = d.columns;
    return [table(cols, d.rows.map(function (r) { return cols.map(function (c) { return r[c] === null || r[c] === undefined ? "-" : String(r[c]); }); }))];
  };

  render.flags = function (d) {
    if (d.not_recorded) { return [notRecorded(d.reason)]; }
    var out = [];
    if (d.columns && d.columns.not_recorded) { out.push(notRecorded(d.columns.reason)); }
    if (!d.flags.length) {
      out.push(el("p", {}, ["Nothing suspicious found by: " + d.checked.join(", ") + ". This is a heuristic, not proof."]));
    } else {
      out.push(el("ul", {}, d.flags.map(function (f) { return el("li", {}, [f.text]); })));
    }
    return out;
  };

  function groupsTable(groups) {
    return table(["Family", "Importance", "Share", "Features"],
      groups.map(function (g) {
        return [g.family, fmt(g.importance), g.share === null ? "-" : fmt(100 * g.share, 3) + "%",
          g.features.map(function (f) { return f.feature; }).join(", ")];
      }), [1, 2]);
  }

  render.features = function (d) {
    if (d.not_recorded) { return [notRecorded(d.reason)]; }
    var out = [];
    var ds = d.describe;
    if (ds && ds.not_recorded) { out.push(notRecorded(ds.reason)); }
    else if (ds && ds.status !== "valid") { out.push(el("div", { "class": "empty", role: "status" }, [el("strong", {}, [ds.status === "missing" ? "Not provided" : "Invalid"]), ds.reason])); }
    else if (ds) {
      var c = ds.content;
      var facts = [];
      if (c.model_family) { facts.push(el("dt", {}, ["Model family"]), el("dd", {}, [c.model_family])); }
      if (c.features_used) { facts.push(el("dt", {}, ["Features used"]), el("dd", {}, [c.features_used.join(", ") || "-"])); }
      if (c.derived_features) {
        facts.push(el("dt", {}, ["Derived features"]), el("dd", {}, [c.derived_features.map(function (f) { return f.name + " (from " + (f.sources.join(", ") || "?") + ")"; }).join("; ") || "-"]));
      }
      if (c.hyperparameters) {
        facts.push(el("dt", {}, ["Hyperparameters"]), el("dd", {}, [Object.keys(c.hyperparameters).map(function (k) { return k + " = " + JSON.stringify(c.hyperparameters[k]); }).join(", ") || "-"]));
      }
      out.push(el("dl", { "class": "facts" }, facts));
    }
    out.push(el("h3", {}, ["Importance the model reports"]));
    out.push(d.native ? groupsTable(d.native.groups) : notRecorded("the bundle did not provide native importances"));
    out.push(el("h3", {}, ["Permutation importance"]));
    var p = d.permutation;
    if (p && p.not_recorded) { out.push(notRecorded(p.reason)); }
    else if (p) {
      out.push(explain(p.meaning));
      out.push(groupsTable(p.groups));
    }
    return out;
  };

  render.history = function (d) {
    if (!d.points.length) { return [notRecorded("no scored attempt yet")]; }
    var xs = d.points.map(function (p) { return p.iteration; });
    var series = [{ label: "score", y: d.points.map(function (p) { return p.score; }), kind: "points", size: 9 },
                  { label: "best so far", y: d.points.map(function (p) { return p.best; }), token: "--chart-chosen" }];
    if (d.baseline) { series.push({ label: "baseline", y: xs.map(function () { return d.baseline.mean; }), token: "--chart-baseline", dash: [6, 4] }); }
    if (d.bar !== null && d.bar !== undefined) { series.push({ label: "improvement bar", y: xs.map(function () { return d.bar; }), token: "--chart-bar", dash: [2, 3] }); }
    return [chartBox({ x: xs, xlabel: "iteration", ylabel: d.metric || "score", series: series })];
  };

  render.correlation = function (d) {
    var c = d.correlation;
    if (!c || c.not_recorded) { return [notRecorded(c ? c.reason : "no data")]; }
    return [el("dl", { "class": "facts" }, [
      el("dt", {}, ["Rows compared"]), el("dd", {}, [String(c.rows)]),
      el("dt", {}, ["Pearson"]), el("dd", {}, [fmt(c.pearson)]),
      el("dt", {}, ["Spearman"]), el("dd", {}, [fmt(c.spearman)]),
      el("dt", {}, ["Label agreement"]), el("dd", {}, [fmt(c.label_agreement)]),
      el("dt", {}, ["Both wrong"]), el("dd", {}, [fmt(c.joint_error_rate)])
    ]), explain(c.text || "")];
  };

  // -- loading ---------------------------------------------------------------------------------

  function load(node) {
    var name = node.getAttribute("data-analytics-section");
    var url = node.getAttribute("data-url");
    var extra = node.getAttribute("data-query") || "";
    node.setAttribute("aria-busy", "true");
    get(url + extra).then(function (r) {
      node.textContent = "";
      if (!r.ok) {
        node.appendChild(el("p", { "class": "error", role: "alert" }, [(r.json.error || {}).message || "The data could not be loaded."]));
      } else {
        (render[name](r.json.data) || []).forEach(function (child) { node.appendChild(child); });
      }
    }).catch(function () {
      node.textContent = "";
      node.appendChild(el("p", { "class": "error", role: "alert" }, ["The data could not be loaded."]));
    }).then(function () { node.removeAttribute("aria-busy"); });
  }

  function wireSegments(node) {
    var slider = document.getElementById("segment-min-size");
    var label = document.getElementById("segment-min-size-value");
    if (!slider) { return; }
    var timer = null;
    slider.addEventListener("input", function () {
      label.textContent = slider.value;
      clearTimeout(timer);
      timer = setTimeout(function () { node.setAttribute("data-query", "?min_size=" + encodeURIComponent(slider.value)); load(node); }, 250);
    });
  }

  function wireHardest(node) {
    var sort = document.getElementById("hardest-sort");
    var order = document.getElementById("hardest-order");
    if (!sort || !order) { return; }
    function again() {
      node.setAttribute("data-query", "?sort=" + encodeURIComponent(sort.value) + "&order=" + encodeURIComponent(order.value));
      load(node);
    }
    sort.addEventListener("change", again);
    order.addEventListener("change", again);
  }

  // -- the permutation importance job -------------------------------------------------------------

  function wireImportance(button) {
    var out = document.getElementById("importance-progress");
    var features = document.querySelector('[data-analytics-section="features"]');
    button.addEventListener("click", function () {
      button.disabled = true;
      window.cockpitPost(button.getAttribute("data-url"), { attempt: button.getAttribute("data-attempt") }).then(function (res) {
        if (res.status !== 202) {
          out.textContent = ((res.json || {}).error || {}).message || "The job could not be started.";
          button.disabled = false;
          return;
        }
        poll(res.json.data.id);
      });
    });
    function poll(id) {
      get(button.getAttribute("data-job-url") + id).then(function (r) {
        var job = r.json.data;
        var progress = job.progress || {};
        out.textContent = "";
        out.appendChild(el("progress", { max: "100", value: String(progress.percent || 0) }, []));
        out.appendChild(document.createTextNode(" " + (progress.percent === null || progress.percent === undefined ? "working" : progress.percent + "%") + " (" + job.status + ")"));
        if (job.status === "running") { setTimeout(function () { poll(id); }, 1500); return; }
        button.disabled = false;
        if (job.status === "done" && features) { load(features); }
        else if (job.error) { out.appendChild(el("p", { "class": "error" }, [job.error])); }
      });
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-analytics-section]").forEach(function (node) {
      var name = node.getAttribute("data-analytics-section");
      if (name === "segments") { wireSegments(node); }
      if (name === "hardest") { wireHardest(node); }
      load(node);
    });
    var button = document.getElementById("importance-start");
    if (button) { wireImportance(button); }
  });
})();
