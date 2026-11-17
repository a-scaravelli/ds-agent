/* Browser notifications and an optional short sound when the run needs you.
   The permission is asked ONLY inside a click handler, never at load. The choices stay in localStorage. */
(function () {
  "use strict";

  var button = document.getElementById("enable-notifications");
  var sound = document.getElementById("notify-sound");
  var state = document.getElementById("notify-state");
  var audio = null;

  function enabled() {
    return window.Notification && Notification.permission === "granted" && localStorage.getItem("notify") === "on";
  }

  function describeState() {
    if (!state) { return; }
    if (!window.Notification) { state.textContent = "This browser has no notifications."; return; }
    if (Notification.permission === "denied") { state.textContent = "Notifications are blocked in the browser settings."; return; }
    state.textContent = enabled() ? "Notifications are on." : "Notifications are off. They are asked for only when you press the button.";
  }

  if (button) {
    button.addEventListener("click", function () {
      if (!window.Notification) { describeState(); return; }
      Notification.requestPermission().then(function (permission) {
        localStorage.setItem("notify", permission === "granted" ? "on" : "off");
        describeState();
      });
    });
  }

  if (sound) {
    sound.checked = localStorage.getItem("notify-sound") === "on";
    sound.addEventListener("click", function () {
      localStorage.setItem("notify-sound", sound.checked ? "on" : "off");
      if (sound.checked && window.AudioContext && !audio) { audio = new AudioContext(); }
    });
  }

  function beep() {
    if (localStorage.getItem("notify-sound") !== "on") { return; }
    try {
      audio = audio || new AudioContext();
      var osc = audio.createOscillator(), gain = audio.createGain();
      osc.frequency.value = 880; gain.gain.value = 0.08;
      osc.connect(gain); gain.connect(audio.destination);
      osc.start(); osc.stop(audio.currentTime + 0.25);
    } catch (e) { /* no sound is not an error */ }
  }

  var TITLES = {
    waiting: "The run is waiting for you", suspended: "The run was suspended",
    error: "The run failed", finished: "The run finished"
  };

  document.addEventListener("cockpit:event", function (event) {
    var e = event.detail;
    if (!TITLES[e.type] || !e.at || Date.now() - new Date(e.at).getTime() > 60000) { return; }  // only fresh events
    var body = (e.data && (e.data.question || e.data.kind || e.data.reason)) || "";
    if (e.type === "suspended" && e.data.kind === "login") { body = "Run `claude auth login` in a Terminal, then resume."; }
    if (enabled()) { try { new Notification(TITLES[e.type], { body: String(body) }); } catch (err) { /* ignore */ } }
    beep();
  });

  describeState();
})();
