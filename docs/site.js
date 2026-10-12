// OS tabs (pre-selects the visitor's OS) and copy buttons. No dependencies.
(function () {
  function detectOS() {
    var p = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || navigator.userAgent;
    if (/win/i.test(p)) return "windows";
    if (/mac|iphone|ipad/i.test(p)) return "macos";
    if (/linux|x11/i.test(p)) return "linux";
    return null;
  }
  document.querySelectorAll(".tabs").forEach(function (box) {
    var tabs = Array.prototype.slice.call(box.querySelectorAll('[role="tab"]'));
    function select(tab, focus) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
      });
      if (focus) tab.focus();
    }
    tabs.forEach(function (t, i) {
      t.addEventListener("click", function () { select(t, false); });
      t.addEventListener("keydown", function (e) {
        var k = e.key, n = tabs.length;
        if (k === "ArrowRight") select(tabs[(i + 1) % n], true);
        else if (k === "ArrowLeft") select(tabs[(i - 1 + n) % n], true);
        else return;
        e.preventDefault();
      });
    });
    var os = detectOS();
    var pre = os && tabs.filter(function (t) { return t.dataset.os === os; })[0];
    select(pre || tabs[0], false);
    if (pre) {
      var note = box.querySelector(".os-note");
      if (note) note.textContent = "Showing instructions for " + pre.textContent.trim() + ", detected from your browser.";
    }
  });
  document.querySelectorAll(".cmd").forEach(function (block) {
    var btn = document.createElement("button");
    btn.className = "copy"; btn.type = "button"; btn.textContent = "Copy";
    btn.setAttribute("aria-label", "Copy command to clipboard");
    btn.addEventListener("click", function () {
      var text = block.querySelector("code").innerText.trim();
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = "Copied"; setTimeout(function () { btn.textContent = "Copy"; }, 1800);
      }, function () { btn.textContent = "Select and copy manually"; });
    });
    block.appendChild(btn);
  });
})();
