"""Headless end-to-end test of the Streamlit app (streamlit.testing.AppTest)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run(sample, preset_index=2):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.selectbox[0].set_value(sample).run()
    assert not at.exception, at.exception
    # ECM: choose preset, fit
    at.selectbox(key=None) if False else None
    ecm_preset = [s for s in at.selectbox if s.label == "Preset"][0]
    ecm_preset.set_value(list(ecm_preset.options)[preset_index]).run()
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
    at = _run("synthetic_2arc.csv", preset_index=5)  # two arcs
    md = " ".join(m.value for m in at.markdown)
    assert "Fit quality" in md and "R∞" in md
    metrics = [m.value for m in at.metric]
    assert len(metrics) == 2 and all("S cm" in v for v in metrics), metrics


def test_blocking_pellet_sample():
    at = _run("synthetic_blocking_pellet.txt", preset_index=1)
    warns = " ".join(w.value for w in at.warning)
    assert "outside the measured" in warns or "not determined" in warns
    assert "capacitive tail" in warns or "blocking" in warns


def test_biologic_multisweep():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.selectbox[0].set_value("synthetic_2arc_biologic.mpt").run()
    assert not at.exception
    assert any(s.label == "Sweep" for s in at.sidebar.selectbox)


if __name__ == "__main__":
    for t in (test_two_arc_sample, test_blocking_pellet_sample, test_biologic_multisweep):
        t()
        print("PASS", t.__name__)
