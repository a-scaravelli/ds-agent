/* Behaviour of the cockpit: keyboard shortcuts, client-side filters and sorting of tables. */
(function () {
  "use strict";

  // -- keyboard: "/" focuses the search box; "g" then a letter follows a navigation link ----
  var pendingGo = false;
  var goTimer = null;

  function typing(target) {
    var tag = target && target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (target && target.isContentEditable);
  }

  document.addEventListener("keydown", function (event) {
    if (event.ctrlKey || event.metaKey || event.altKey || typing(event.target)) { return; }
    if (event.key === "/") {
      var box = document.querySelector("[data-filter-input]");
      if (box) { event.preventDefault(); box.focus(); }
      return;
    }
    if (pendingGo) {
      pendingGo = false;
      clearTimeout(goTimer);
      var link = document.querySelector('a[data-key="' + event.key.toLowerCase() + '"]');
      if (link) { event.preventDefault(); window.location.href = link.href; }
      return;
    }
    if (event.key === "g") {
      pendingGo = true;
      goTimer = setTimeout(function () { pendingGo = false; }, 1500);
    }
  });

  // -- filters: inputs with data-filter-for name a table; rows carry data-* values ----------
  function rowMatches(row, controls) {
    for (var i = 0; i < controls.length; i++) {
      var c = controls[i];
      var kind = c.getAttribute("data-filter");
      var value = c.value.trim();
      if (value === "") { continue; }
      if (kind === "text") {
        if ((row.getAttribute("data-text") || "").indexOf(value.toLowerCase()) === -1) { return false; }
      } else if (kind === "min" || kind === "max") {
        var score = parseFloat(row.getAttribute("data-score"));
        var limit = parseFloat(value);
        if (isNaN(limit)) { continue; }
        if (isNaN(score)) { return false; }
        if (kind === "min" && score < limit) { return false; }
        if (kind === "max" && score > limit) { return false; }
      } else if (row.getAttribute("data-" + kind) !== value) {
        return false;
      }
    }
    return true;
  }

  function applyFilters(tableId) {
    var table = document.getElementById(tableId);
    if (!table) { return; }
    var controls = document.querySelectorAll('[data-filter-for="' + tableId + '"]');
    var shown = 0;
    var rows = table.querySelectorAll("tbody tr");
    rows.forEach(function (row) {
      var ok = rowMatches(row, controls);
      row.hidden = !ok;
      if (ok) { shown++; }
    });
    var count = document.querySelector('[data-filter-count="' + tableId + '"]');
    if (count) { count.textContent = shown + " of " + rows.length; }
  }

  document.querySelectorAll("[data-filter-for]").forEach(function (control) {
    var id = control.getAttribute("data-filter-for");
    control.addEventListener("input", function () { applyFilters(id); });
    control.addEventListener("change", function () { applyFilters(id); });
  });
  document.querySelectorAll("table[data-filterable]").forEach(function (table) { applyFilters(table.id); });

  // -- sorting: buttons inside th[data-sort] (number or text) --------------------------------
  document.querySelectorAll("th[data-sort] button.sort").forEach(function (button) {
    button.addEventListener("click", function () {
      var th = button.closest("th");
      var table = th.closest("table");
      var index = Array.prototype.indexOf.call(th.parentNode.children, th);
      var numeric = th.getAttribute("data-sort") === "number";
      var descending = th.getAttribute("aria-sort") === "ascending";
      table.querySelectorAll("th").forEach(function (other) { other.removeAttribute("aria-sort"); });
      th.setAttribute("aria-sort", descending ? "descending" : "ascending");
      var body = table.tBodies[0];
      var rows = Array.prototype.slice.call(body.rows);
      function key(row) {
        var cell = row.cells[index];
        var raw = cell.getAttribute("data-value");
        if (raw === null) { raw = cell.textContent.trim(); }
        if (numeric) { var n = parseFloat(raw); return isNaN(n) ? -Infinity : n; }
        return raw.toLowerCase();
      }
      rows.sort(function (a, b) {
        var ka = key(a);
        var kb = key(b);
        var order = ka < kb ? -1 : ka > kb ? 1 : 0;
        return descending ? -order : order;
      });
      rows.forEach(function (row) { body.appendChild(row); });
    });
  });
})();
