/* Score chart of the run overview: uPlot with error bars (mean and dispersion), the baseline,
   the improvement bar and the chosen attempt. Data comes from the chart API (data-url). */
(function () {
  "use strict";

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function build(el, payload) {
    var points = payload.points;
    if (!points.length || typeof uPlot === "undefined") {
      el.textContent = points.length ? "The chart library is not available." : "No scored attempt yet.";
      return null;
    }
    var colors = {
      line: css("--chart-line"), error: css("--chart-error"), baseline: css("--chart-baseline"),
      bar: css("--chart-bar"), chosen: css("--chart-chosen"), grid: css("--chart-grid"),
      text: css("--chart-axis"), background: css("--chart-bg"), failed: css("--chart-failed")
    };
    var font = "12px " + css("--font-sans");
    var xs = points.map(function (p) { return p.iteration; });
    var means = points.map(function (p) { return p.mean; });
    var baseline = payload.baseline ? xs.map(function () { return payload.baseline.mean; }) : null;
    var bar = payload.bar !== null && payload.bar !== undefined
      ? xs.map(function () { return payload.bar; }) : null;

    var lo = Infinity;
    var hi = -Infinity;
    points.forEach(function (p) {
      lo = Math.min(lo, p.mean - p.dispersion);
      hi = Math.max(hi, p.mean + p.dispersion);
    });
    [payload.bar, payload.baseline ? payload.baseline.mean : null].forEach(function (v) {
      if (v !== null && v !== undefined) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    });
    var pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.05 || 1;

    var series = [{}, {
      label: "score (mean)", stroke: colors.line, width: 2.5,
      points: { show: true, size: 8, stroke: colors.background, fill: colors.line, width: 2 }
    }];
    var data = [xs, means];
    if (baseline) {
      series.push({ label: "baseline", stroke: colors.baseline, width: 1, dash: [6, 4], points: { show: false } });
      data.push(baseline);
    }
    if (bar) {
      series.push({ label: "improvement bar", stroke: colors.bar, width: 1, dash: [2, 3], points: { show: false } });
      data.push(bar);
    }

    function drawOverlay(u) {
      var ctx = u.ctx;
      var scale = window.devicePixelRatio || 1;
      ctx.save();
      ctx.strokeStyle = colors.error;
      ctx.lineWidth = 1.5 * scale;
      var cap = 4 * scale;
      points.forEach(function (p) {
        var x = u.valToPos(p.iteration, "x", true);
        var top = u.valToPos(p.mean + p.dispersion, "y", true);
        var bottom = u.valToPos(p.mean - p.dispersion, "y", true);
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, bottom);
        ctx.moveTo(x - cap, top);
        ctx.lineTo(x + cap, top);
        ctx.moveTo(x - cap, bottom);
        ctx.lineTo(x + cap, bottom);
        ctx.stroke();
      });
      ctx.strokeStyle = colors.chosen;
      ctx.lineWidth = 3 * scale;
      points.forEach(function (p) {
        if (!p.chosen) { return; }
        var x = u.valToPos(p.iteration, "x", true);
        var y = u.valToPos(p.mean, "y", true);
        ctx.beginPath();
        ctx.arc(x, y, 9 * scale, 0, 2 * Math.PI);
        ctx.stroke();
      });
      ctx.restore();
    }

    var options = {
      width: Math.max(el.clientWidth, 320),
      height: 320,
      scales: {
        x: { time: false },
        y: { range: function () { return [lo - pad, hi + pad]; } }
      },
      axes: [
        { label: "iteration", stroke: colors.text, font: font, labelFont: font, grid: { stroke: colors.grid, width: 1 },
          ticks: { stroke: colors.grid }, values: function (u, ticks) {
            return ticks.map(function (t) { return Number.isInteger(t) ? String(t) : ""; });
          } },
        { label: payload.metric || "score", stroke: colors.text, font: font, labelFont: font, grid: { stroke: colors.grid, width: 1 },
          ticks: { stroke: colors.grid } }
      ],
      series: series,
      hooks: { draw: [drawOverlay] }
    };
    el.textContent = "";
    var plot = new uPlot(options, data, el);
    var observer = null;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(function () {
        plot.setSize({ width: Math.max(el.clientWidth, 320), height: 320 });
      });
      observer.observe(el);
    }
    return { plot: plot, observer: observer };
  }

  var charts = [];

  function render(entry) {
    if (entry.handle) {
      if (entry.handle.observer) { entry.handle.observer.disconnect(); }
      entry.handle.plot.destroy();
    }
    entry.handle = build(entry.el, entry.payload);
  }

  function init() {
    document.querySelectorAll("[data-chart-url]").forEach(function (el) {
      fetch(el.getAttribute("data-chart-url"), { headers: { Accept: "application/json" } })
        .then(function (response) { return response.json(); })
        .then(function (body) {
          var entry = { el: el, payload: body.data, handle: null };
          charts.push(entry);
          render(entry);
        })
        .catch(function () { el.textContent = "The chart data could not be loaded; see the table below."; });
    });
  }

  // the colours come from the CSS variables: when the theme changes the chart is rebuilt
  document.addEventListener("themechange", function () {
    charts.forEach(function (entry) { if (entry.handle) { render(entry); } });
  });

  document.addEventListener("DOMContentLoaded", init);
})();

/* Generic charts of the performance analytics (ticket 29): dot plot, history, ROC, PR, calibration, residuals.
   Every colour is a design token read from the CSS variables; the plots are rebuilt on theme change. */
(function () {
  "use strict";

  var SERIES_TOKENS = ["--chart-line", "--chart-chosen", "--chart-bar", "--chart-baseline", "--chart-failed"];
  var registry = [];

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function resample(x, y, grid) {
    return grid.map(function (g) {
      var i = 0;
      while (i < x.length - 1 && x[i + 1] < g) { i += 1; }
      if (i >= x.length - 1) { return y[y.length - 1]; }
      var span = x[i + 1] - x[i];
      return span === 0 ? y[i + 1] : y[i] + (y[i + 1] - y[i]) * (g - x[i]) / span;
    });
  }

  function draw(entry) {
    var spec = entry.spec;
    var el = entry.el;
    if (entry.plot) { entry.plot.destroy(); entry.plot = null; }
    if (typeof uPlot === "undefined") { el.textContent = "The chart library is not available; see the table."; return; }
    var font = "12px " + css("--font-sans");
    var axis = css("--chart-axis");
    var grid = css("--chart-grid");
    var series = [{}];
    spec.series.forEach(function (s, i) {
      var colour = css(s.token || SERIES_TOKENS[i % SERIES_TOKENS.length]);
      var item = { label: s.label, stroke: colour, width: s.width || 2 };
      if (s.kind === "points") {
        item.paths = function () { return null; };
        item.points = { show: true, size: s.size || 8, stroke: css("--chart-bg"), fill: colour, width: 1 };
      } else {
        item.points = { show: false };
        if (s.dash) { item.dash = s.dash; }
      }
      series.push(item);
    });
    var options = {
      width: Math.max(el.clientWidth, 320), height: spec.height || 300,
      scales: { x: { time: false, range: spec.xrange || undefined }, y: { range: spec.yrange || undefined } },
      axes: [
        { label: spec.xlabel, stroke: axis, font: font, labelFont: font, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid } },
        { label: spec.ylabel, stroke: axis, font: font, labelFont: font, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid } }
      ],
      series: series
    };
    el.textContent = "";
    var data = [spec.x].concat(spec.series.map(function (s) { return s.y; }));
    entry.plot = new uPlot(options, data, el);
    if (typeof ResizeObserver !== "undefined" && !entry.observer) {
      entry.observer = new ResizeObserver(function () {
        if (entry.plot) { entry.plot.setSize({ width: Math.max(el.clientWidth, 320), height: spec.height || 300 }); }
      });
      entry.observer.observe(el);
    }
  }

  window.analyticsCharts = {
    resample: resample,
    plot: function (el, spec) {
      var entry = { el: el, spec: spec, plot: null, observer: null };
      registry.push(entry);
      draw(entry);
    }
  };

  document.addEventListener("themechange", function () { registry.forEach(draw); });
})();
