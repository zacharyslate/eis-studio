"""Two-way conversion between impedance.py circuit strings and a JSON tree used by the
visual circuit editor.

Tree nodes (JSON):
    {"t": "e", "k": "CPE", "n": "CPE1"}      element: kind, name (name may be null for new ones)
    {"t": "s", "c": [node, node, ...]}        series group
    {"t": "p", "c": [node, node, ...]}        parallel group

impedance.py syntax: '-' joins elements in series, p(a,b,...) puts them in parallel, and a
series chain may appear inside a parallel branch: R0-p(R1-Wo1,CPE1).
"""

from __future__ import annotations

import re

from impedance.models.circuits.elements import circuit_elements

KINDS = [k for k in circuit_elements if k not in ("p", "s")]
_NAME = re.compile(r"([A-Za-z]+)(\d+)")


class CircuitSyntaxError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# string -> tree
# --------------------------------------------------------------------------- #
def parse(text: str):
    """Parse an impedance.py circuit string into a tree. Empty string -> None."""
    s = (text or "").replace(" ", "")
    if not s:
        return None
    pos = 0

    def peek():
        return s[pos] if pos < len(s) else ""

    def series():
        nonlocal pos
        items = [term()]
        while peek() == "-":
            pos += 1
            items.append(term())
        return items[0] if len(items) == 1 else {"t": "s", "c": items}

    def term():
        nonlocal pos
        if s.startswith("p(", pos):
            pos += 2
            branches = [series()]
            while peek() == ",":
                pos += 1
                branches.append(series())
            if peek() != ")":
                raise CircuitSyntaxError(f"Expected ')' at position {pos + 1} in '{s}'.")
            pos += 1
            return {"t": "p", "c": branches}
        m = _NAME.match(s, pos)
        if not m:
            raise CircuitSyntaxError(f"Expected an element (e.g. R1, CPE2) or p(...) at position {pos + 1} in '{s}'.")
        kind = m.group(1)
        if kind not in KINDS:
            raise CircuitSyntaxError(f"Unknown element '{m.group(0)}'. Available: {', '.join(KINDS)}.")
        pos = m.end()
        return {"t": "e", "k": kind, "n": m.group(0)}

    tree = series()
    if pos != len(s):
        raise CircuitSyntaxError(f"Unexpected '{s[pos]}' at position {pos + 1} in '{s}'.")
    return normalize(tree)


# --------------------------------------------------------------------------- #
# tree utilities
# --------------------------------------------------------------------------- #
def normalize(node):
    """Flatten nested groups of the same type, drop empty groups, unwrap single-child groups."""
    if node is None:
        return None
    if node.get("t") == "e":
        kind = node.get("k")
        if kind not in KINDS:
            raise CircuitSyntaxError(f"Unknown element kind '{kind}'.")
        return {"t": "e", "k": kind, "n": node.get("n")}
    kids = []
    for child in node.get("c", []):
        child = normalize(child)
        if child is None:
            continue
        if child["t"] == node["t"]:
            kids.extend(child["c"])
        else:
            kids.append(child)
    if not kids:
        return None
    if len(kids) == 1:
        return kids[0]
    return {"t": node["t"], "c": kids}


def _leaves(node):
    if node is None:
        return
    if node["t"] == "e":
        yield node
    else:
        for c in node["c"]:
            yield from _leaves(c)


def assign_names(tree):
    """Keep valid unique existing names; number new or duplicate elements with the next free index."""
    tree = normalize(tree)
    used = set()
    pending = []
    for leaf in _leaves(tree):
        n = leaf.get("n")
        m = _NAME.fullmatch(n) if isinstance(n, str) else None
        if m and m.group(1) == leaf["k"] and n not in used:
            used.add(n)
        else:
            pending.append(leaf)
    for leaf in pending:
        i = 0
        while f"{leaf['k']}{i}" in used:
            i += 1
        leaf["n"] = f"{leaf['k']}{i}"
        used.add(leaf["n"])
    return tree


# --------------------------------------------------------------------------- #
# tree -> string
# --------------------------------------------------------------------------- #
def to_string(tree) -> str:
    tree = assign_names(tree)
    if tree is None:
        return ""

    def emit(node):
        if node["t"] == "e":
            return node["n"]
        if node["t"] == "s":
            return "-".join(emit(c) for c in node["c"])
        return "p(" + ",".join(emit(c) for c in node["c"]) + ")"

    return emit(tree)
