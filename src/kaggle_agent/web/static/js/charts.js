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
      text: css("--text-muted")
    };
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
      label: "score (mean)", stroke: colors.line, width: 1.5,
      points: { show: true, size: 7, stroke: colors.line, fill: colors.line }
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
      ctx.lineWidth = 2.5 * scale;
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
        { label: "iteration", stroke: colors.text, grid: { stroke: colors.grid, width: 1 },
          ticks: { stroke: colors.grid }, values: function (u, ticks) {
            return ticks.map(function (t) { return Number.isInteger(t) ? String(t) : ""; });
          } },
        { label: payload.metric || "score", stroke: colors.text, grid: { stroke: colors.grid, width: 1 },
          ticks: { stroke: colors.grid } }
      ],
      series: series,
      hooks: { draw: [drawOverlay] }
    };
    el.textContent = "";
    var plot = new uPlot(options, data, el);
    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(function () {
        plot.setSize({ width: Math.max(el.clientWidth, 320), height: 320 });
      }).observe(el);
    }
    return plot;
  }

  function init() {
    document.querySelectorAll("[data-chart-url]").forEach(function (el) {
      fetch(el.getAttribute("data-chart-url"), { headers: { Accept: "application/json" } })
        .then(function (response) { return response.json(); })
        .then(function (body) { build(el, body.data); })
        .catch(function () { el.textContent = "The chart data could not be loaded; see the table below."; });
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
