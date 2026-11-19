/* Theme opt-in. The page is light for everybody; dark only when the user chose it with the toggle
   (the choice is kept in localStorage under "cockpit-theme"). The operating system or browser
   colour preference is never read. Loaded in <head> without defer so the theme is set before the
   first paint. */
(function () {
  "use strict";

  var KEY = "cockpit-theme";
  var root = document.documentElement;

  function stored() {
    try { return window.localStorage.getItem(KEY); } catch (error) { return null; }
  }

  function remember(dark) {
    try {
      if (dark) { window.localStorage.setItem(KEY, "dark"); } else { window.localStorage.removeItem(KEY); }
    } catch (error) { /* storage blocked: the choice is not kept, the page keeps working */ }
  }

  function apply(dark) {
    if (dark) { root.dataset.theme = "dark"; } else { delete root.dataset.theme; }
  }

  apply(stored() === "dark");

  document.addEventListener("DOMContentLoaded", function () {
    var button = document.getElementById("theme-toggle");
    if (!button) { return; }
    function sync() { button.setAttribute("aria-pressed", root.dataset.theme === "dark" ? "true" : "false"); }
    sync();
    button.addEventListener("click", function () {
      var dark = root.dataset.theme !== "dark";
      apply(dark);
      remember(dark);
      sync();
      document.dispatchEvent(new CustomEvent("themechange", { detail: { dark: dark } }));
    });
  });
})();
