"""Headless end-to-end test of the Streamlit app (streamlit.testing.AppTest)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run(sample, circuit):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.selectbox[0].set_value(sample).run()
    assert not at.exception, at.exception
    # ECM: type the circuit (the visual editor writes the same field), then fit
    at.text_input(key="circuit_input").set_value(circuit).run()
    assert not at.exception, at.exception
    assert at.session_state["circuit_text"] == circuit
    [b for b in at.button if b.label == "Fit circuit"][0].click().run()
    assert not at.exception, at.exception
    errs = [e.value for e in at.error]
    assert not errs, errs
    # DRT
    [b for b in at.button if b.label == "Compute DRT"][0].click().run()
    assert not at.exception, at.exception
    assert not [e.value for e in at.error], [e.value for e in at.error]
    return at


def test_two_arc_sample():
    at = _run("synthetic_2arc.csv", "R0-p(R1,CPE1)-p(R2,CPE2)")
    md = " ".join(m.value for m in at.markdown)
    assert "Fit quality" in md and "R∞" in md
    metrics = [m.value for m in at.metric]
    assert len(metrics) == 2 and all("S cm" in v for v in metrics), metrics


def test_blocking_pellet_sample():
    at = _run("synthetic_blocking_pellet.txt", "R0-p(R1,CPE1)-CPE2")
    warns = " ".join(w.value for w in at.warning)
    assert "outside the measured" in warns or "not determined" in warns
    assert "capacitive tail" in warns or "blocking" in warns


def test_bad_circuit_string_is_rejected_not_applied():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.selectbox[0].set_value("synthetic_2arc.csv").run()
    before = at.session_state["circuit_text"]
    at.text_input(key="circuit_input").set_value("R0-p(R1,CPE1").run()
    assert not at.exception
    assert at.session_state["circuit_text"] == before
    assert any("not applied" in e.value for e in at.error)


def test_biologic_multisweep():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.selectbox[0].set_value("synthetic_2arc_biologic.mpt").run()
    assert not at.exception
    assert any(s.label == "Sweep" for s in at.sidebar.selectbox)


if __name__ == "__main__":
    for t in (test_two_arc_sample, test_blocking_pellet_sample, test_bad_circuit_string_is_rejected_not_applied,
              test_biologic_multisweep):
        t()
        print("PASS", t.__name__)


def test_series_mode():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.radio(key="mode").set_value("Series (several measurements)").run()
    at.sidebar.selectbox(key="s_sample").set_value("Temperature series + DC steps (one EC-Lab file)").run()
    assert not at.exception, at.exception
    [t for t in at.text_input if (t.key or "").startswith("s_tlist")][0].set_value("25, 40, 55, 70, 85").run()
    [b for b in at.button if b.label == "Fill"][0].click().run()
    form_method = [s for s in at.selectbox if s.label == "Ionic resistance from"][0]
    form_method.set_value("Z′ at the minimum of −Z″ (arc/spike junction)")
    [b for b in at.button if b.label == "Analyse spectra"][0].click().run()
    assert not at.exception, at.exception
    m = {x.label: x.value for x in at.metric}
    assert m["Activation energy"].startswith("0.45"), m
    assert "e-09" in m["σ_e"], m
    # density set with the dense-equivalent normalisation renders
    at.sidebar.selectbox(key="s_sample").set_value("Five pellets pressed to different densities").run()
    at.selectbox(key="s_norm").set_value("Geometry and density: Z·(A/L)·ρ_rel^m (Ω cm, dense-equivalent)").run()
    assert not at.exception, at.exception
    assert any("Normalisation not possible" in w.value for w in at.warning)   # no masses entered yet


def test_series_mode_circuit_fit_drt_and_individual_settings():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    at.sidebar.radio(key="mode").set_value("Series (several measurements)").run()
    at.sidebar.selectbox(key="s_sample").set_value("Temperature series as separate files + DC table").run()
    assert at.session_state["s_circuit_text"] == "p(R1,CPE1)-CPE2"      # same editor as single mode
    at.text_input(key="s_circuit_input").set_value("R0-p(R1,CPE1)-CPE2").run()
    assert at.session_state["s_circuit_text"] == "R0-p(R1,CPE1)-CPE2"
    # R0 + R1: at 85 °C the arc leaves the window and the R0/R1 split is not determined, so R1 alone
    # would bend the Arrhenius line (the app warns about this)
    at.multiselect(key="s_rsel_R0-p(R1,CPE1)-CPE2").set_value(["R0", "R1"]).run()
    at.radio(key="s_start").set_value("The previous spectrum's result (coldest first)").run()
    [b for b in at.button if b.label == "Analyse spectra"][0].click().run()
    assert not at.exception, at.exception
    m = {x.label: x.value for x in at.metric}
    assert m["Activation energy"].startswith(("0.44", "0.45")), m
    heads = [x.value for x in at.markdown if x.value.startswith("####")]
    assert {"#### Fitted parameters", "#### DRT peaks", "#### Inspect one spectrum"} <= set(heads), heads
    [b for b in at.button if b.label == "Refit this spectrum"][0].click().run()
    assert not at.exception, at.exception
    assert any(":" in k for k in at.session_state["s_overrides"])
    assert not any("changed since the last analysis" in i.value for i in at.info)
