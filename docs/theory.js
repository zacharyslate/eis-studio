// Calculators on the theory page. Same formulas as eisstudio/conductivity.py and the methods page.
// Recalculate on every input; nothing is stored or sent anywhere.
(function () {
  "use strict";
  var EPS0 = 8.8541878128e-12;   // F/m, CODATA 2018
  var KB = 8.617333262e-5;       // eV/K, CODATA 2018
  var EV_KJMOL = 96.48533212;    // kJ/mol per eV (= Faraday constant / 1000)

  function num(id) {
    var v = String(document.getElementById(id).value).trim().replace(/−/g, "-").replace(/×\s*10\^?/g, "e");
    if (v === "") return NaN;
    return Number(v);
  }
  // 3 significant figures, with ×10ⁿ for very small or large numbers
  var SUP = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
  function fmt(x, sig) {
    sig = sig || 3;
    if (!isFinite(x)) return "–";
    if (x === 0) return "0";
    var e = Math.floor(Math.log10(Math.abs(x)));
    if (e >= -3 && e < 5) return Number(x.toPrecision(sig)).toString();
    var m = x / Math.pow(10, e);
    var ms = Number(m.toPrecision(sig)).toString();
    if (ms === "10") { ms = "1"; e += 1; }
    return ms + " × 10" + String(e).split("").map(function (c) { return SUP[c]; }).join("");
  }
  // value and uncertainty to the same decimal place, uncertainty to 2 significant figures
  function pm(v, u) {
    if (!(u > 0)) return fmt(v);
    var dec = Math.max(0, 1 - Math.floor(Math.log10(u)));
    return v.toFixed(dec) + " ± " + u.toFixed(dec);
  }
  function show(form, html) { form.querySelector("output").innerHTML = html; }
  function err(form, msg) { show(form, '<p class="error">' + msg + "</p>"); }
  function wire(form, fn) {
    if (!form) return;
    form.addEventListener("input", fn);
    form.addEventListener("change", fn);
    fn();
  }

  // ---------------------------------------------------------------- conductivity
  var cond = document.getElementById("calc-cond");
  wire(cond, function () {
    var byArea = document.getElementById("c-mode").value === "a";
    document.getElementById("c-d-lab").textContent = byArea ? "Electrode area A (cm²)" : "Diameter d (mm)";
    document.getElementById("c-dd-lab").textContent = byArea ? "± area (cm²)" : "± diameter (mm)";
    var L = num("c-l") / 10, dL = (num("c-dl") || 0) / 10;          // cm
    var size = num("c-d"), dsize = num("c-dd") || 0;
    var R = num("c-r"), dR = num("c-dr") || 0;
    if (!(L > 0) || !(size > 0) || !(R > 0)) return err(cond, "Enter a positive thickness, size and resistance.");
    var A, relA;
    if (byArea) { A = size; relA = dsize / size; }
    else { var d = size / 10; A = Math.PI * d * d / 4; relA = 2 * (dsize / size); }
    var K = L / A, sigma = K / R;
    var rel = { R: dR / R, L: dL / L, A: relA };
    var tot = Math.sqrt(rel.R * rel.R + rel.L * rel.L + rel.A * rel.A);
    var ds = sigma * tot;
    show(cond,
      '<p class="result">σ = ' + pm(sigma * 1e3, ds * 1e3) + " mS cm⁻¹</p>" +
      '<p class="detail">= ' + fmt(sigma) + " S cm⁻¹ = " + fmt(sigma * 100) + " S m⁻¹ · A = " + fmt(A, 4) + " cm² · cell constant L/A = " +
      fmt(K, 4) + " cm⁻¹ · ASR = R·A = " + fmt(R * A) + " Ω cm²</p>" +
      (ds > 0 ? '<p class="detail">Relative uncertainty ' + (tot * 100).toFixed(2) + " %: resistance " + (rel.R * 100).toFixed(2) +
        " %, thickness " + (rel.L * 100).toFixed(2) + " %, area " + (rel.A * 100).toFixed(2) + " %.</p>" : ""));
  });

  // ---------------------------------------------------------------- CPE
  var cpe = document.getElementById("calc-cpe");
  wire(cpe, function () {
    var R = num("cpe-r"), Q = num("cpe-q"), a = num("cpe-a");
    if (!(R > 0) || !(Q > 0)) return err(cpe, "Enter positive values for R and Q.");
    if (!(a > 0 && a <= 1)) return err(cpe, "α must be between 0 and 1.");
    var tau = Math.pow(R * Q, 1 / a);
    var ceff = Math.pow(Q, 1 / a) * Math.pow(R, (1 - a) / a);
    var fc = 1 / (2 * Math.PI * tau);
    show(cpe,
      '<p class="result">C<sub>eff</sub> = ' + fmt(ceff) + " F</p>" +
      '<p class="detail">τ = (RQ)<sup>1/α</sup> = ' + fmt(tau) + " s · characteristic frequency f<sub>c</sub> = 1/(2πτ) = " + fmt(fc) + " Hz" +
      (a < 0.8 ? " · α below 0.8: treat C<sub>eff</sub> as an order of magnitude" : "") + "</p>");
  });

  // ---------------------------------------------------------------- bulk arc frequency
  var bulk = document.getElementById("calc-bulk");
  wire(bulk, function () {
    var s = num("bulk-s"), er = num("bulk-e"), fmax = num("bulk-f") * 1e6;
    if (!(s > 0) || !(er >= 1)) return err(bulk, "Enter a positive conductivity and a permittivity of at least 1.");
    var fb = (s * 100) / (2 * Math.PI * EPS0 * er);
    var verdict;
    if (!(fmax > 0)) verdict = "";
    else if (fb > fmax) verdict = "Above your analyser's limit: the bulk arc is not measured. Use the high-frequency intercept as the bulk (or bulk + grain-boundary) resistance.";
    else if (fb > fmax / 10) verdict = "Close to your analyser's limit: only part of the bulk arc is measured, so a fitted bulk R and C will be uncertain.";
    else verdict = "Within your analyser's range: the bulk arc should be visible.";
    show(bulk,
      '<p class="result">f<sub>b</sub> = ' + fmt(fb) + " Hz</p>" +
      '<p class="detail">τ<sub>b</sub> = ε₀ε<sub>r</sub>/σ = ' + fmt(EPS0 * er / (s * 100)) + " s. " + verdict + "</p>");
  });

  // ---------------------------------------------------------------- Arrhenius
  var arr = document.getElementById("calc-arr");
  wire(arr, function () {
    var lines = document.getElementById("arr-data").value.split(/\r?\n/);
    var withT = document.getElementById("arr-form").value === "sT";
    var x = [], y = [], bad = [];
    lines.forEach(function (ln, i) {
      ln = ln.trim();
      if (!ln || ln.charAt(0) === "#") return;
      var p = ln.split(/[\s,;\t]+/).filter(Boolean).map(function (t) { return Number(t.replace(/−/g, "-")); });
      if (p.length < 2 || !isFinite(p[0]) || !(p[1] > 0)) { bad.push(i + 1); return; }
      var T = p[0] + 273.15;
      if (!(T > 0)) { bad.push(i + 1); return; }
      x.push(1 / T);
      y.push(Math.log(withT ? p[1] * T : p[1]));
    });
    if (bad.length) return err(arr, "Could not read line" + (bad.length > 1 ? "s " : " ") + bad.join(", ") + ". Use: temperature in °C, conductivity in S cm⁻¹.");
    var n = x.length;
    if (n < 2) return err(arr, "Enter at least two temperatures.");
    var mx = x.reduce(function (a, b) { return a + b; }, 0) / n;
    var my = y.reduce(function (a, b) { return a + b; }, 0) / n;
    var sxx = 0, sxy = 0, syy = 0;
    for (var i = 0; i < n; i++) { sxx += (x[i] - mx) * (x[i] - mx); sxy += (x[i] - mx) * (y[i] - my); syy += (y[i] - my) * (y[i] - my); }
    if (sxx === 0) return err(arr, "All temperatures are the same.");
    var m = sxy / sxx, c = my - m * mx;
    var Ea = -m * KB;
    var html = '<p class="result">E<sub>a</sub> = ' + Ea.toFixed(3);
    if (n > 2) {
      var ssr = 0;
      for (var k = 0; k < n; k++) { var r = y[k] - (m * x[k] + c); ssr += r * r; }
      var se = Math.sqrt(ssr / (n - 2) / sxx) * KB;
      var r2 = syy > 0 ? 1 - ssr / syy : 1;
      html += " ± " + se.toFixed(3) + " eV</p>";
      html += '<p class="detail">= ' + (Ea * EV_KJMOL).toFixed(1) + " ± " + (se * EV_KJMOL).toFixed(1) + " kJ mol⁻¹ · " + n +
        " points · R² = " + r2.toFixed(5) + " · prefactor " + (withT ? "A = " + fmt(Math.exp(c)) + " S K cm⁻¹" : "σ₀ = " + fmt(Math.exp(c)) + " S cm⁻¹") +
        " · ± is one standard error of the slope</p>";
    } else {
      html += " eV</p><p class=\"detail\">= " + (Ea * EV_KJMOL).toFixed(1) + " kJ mol⁻¹ from two points: no uncertainty can be estimated.</p>";
    }
    show(arr, html);
  });
})();
