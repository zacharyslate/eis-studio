"""Streamlit wrapper for the visual circuit editor (components/circuit_editor.js)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from . import circuit_tree as ct

_DIR = Path(__file__).with_name("components")

PRESETS = [
    ("Pellet: bulk arc + blocking electrodes", "R0-p(R1,CPE1)-CPE2"),
    ("Pellet: bulk, grain boundary, electrodes", "R0-p(R1,CPE1)-p(R2,CPE2)-CPE3"),
    ("Pellet: arcs above the window (fast conductors)", "R0-CPE1"),
    ("One arc", "R0-p(R1,CPE1)"),
    ("Two arcs", "R0-p(R1,CPE1)-p(R2,CPE2)"),
    ("Two arcs + lead inductance", "L0-R0-p(R1,CPE1)-p(R2,CPE2)"),
    ("Randles, finite diffusion", "R0-p(R1-Wo1,CPE1)"),
    ("Randles, semi-infinite diffusion", "R0-p(R1-W1,CPE1)"),
]


_JS = (_DIR / "circuit_editor.js").read_text(encoding="utf-8")
_CSS = (_DIR / "circuit_editor.css").read_text(encoding="utf-8")


def _component():
    # Register on every run: the registry lives on the Streamlit runtime (a cached object
    # would point at a stale runtime), and re-registering an identical definition is silent.
    return st.components.v2.component("eisstudio_circuit_editor", js=_JS, css=_CSS)


def circuit_editor(circuit_text: str, key: str = "circuit_editor"):
    """Render the editor for ``circuit_text``. Returns the edited tree when the user changed
    something since the last call (as a dict with 'tree' and 'nonce'), else None."""
    try:
        tree = ct.parse(circuit_text)
    except ct.CircuitSyntaxError:
        tree = None
    data = {
        "tree": tree,
        "current": ct.to_string(tree) if tree else "",
        "kinds": ct.KINDS,
        "presets": [{"label": label, "text": text, "tree": ct.parse(text)} for label, text in PRESETS],
    }
    result = _component()(key=key, data=data, on_tree_change=lambda: None, height="content")
    value = getattr(result, "tree", None)
    if not value:
        return None
    seen = st.session_state.get(f"{key}__nonce")
    if value.get("nonce") == seen:
        return None
    st.session_state[f"{key}__nonce"] = value.get("nonce")
    return value
