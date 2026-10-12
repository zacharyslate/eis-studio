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
