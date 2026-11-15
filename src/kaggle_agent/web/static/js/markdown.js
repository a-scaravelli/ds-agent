/* Renders the text of elements marked data-markdown with marked (vendored, version 5 or newer).
   Model text is untrusted: raw HTML is shown as escaped text and a link or image keeps its URL
   only when it is http, https or relative (so no javascript: link survives). */
(function () {
  "use strict";
  if (typeof marked === "undefined") { return; }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function safeUrl(href) {
    var value = String(href || "").trim();
    if (value === "") { return true; }
    var match = /^([a-zA-Z][a-zA-Z0-9+.-]*):/.exec(value);
    if (!match) { return true; }
    var scheme = match[1].toLowerCase();
    return scheme === "http" || scheme === "https";
  }

  marked.use({
    renderer: {
      html: function (token) {
        return escapeHtml(token.text);
      }
    },
    walkTokens: function (token) {
      if (token.type === "link" || token.type === "image") {
        if (!safeUrl(token.href)) { token.href = ""; }
      }
    }
  });

  function render(root) {
    var nodes = root.querySelectorAll("[data-markdown]:not(.rendered)");
    nodes.forEach(function (node) {
      node.innerHTML = marked.parse(node.textContent, { async: false });
      node.classList.add("rendered");
    });
  }

  document.addEventListener("DOMContentLoaded", function () { render(document); });
  document.addEventListener("htmx:afterSwap", function (event) {
    render(event.target || document);
  });
  window.cockpitRenderMarkdown = render;
})();
