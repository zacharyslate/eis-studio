"""Reading EIS data from the formats potentiostats actually export.

Convention used everywhere in this package
------------------------------------------
* ``freq`` is in Hz, sorted from high to low frequency.
* ``Z`` is complex with the *true signed* imaginary part, Z = Z' + jZ''.
  Capacitive behaviour therefore gives Z'' < 0. Plots show -Z''.

Strategy
--------
1. Vendor formats with a dedicated reader in impedance.py (Gamry .DTA, ZView .z,
   VersaStudio, Parstat, Autolab, CH Instruments, PowerSuite) are tried first
   when the user picks that instrument or the extension is unambiguous.
2. Everything else (BioLogic .mpt, generic CSV/TXT/TSV, Excel) goes through a
   header-aware table detector that finds the numeric data block, maps columns
   by name (frequency, Re, Im, or |Z| + phase), handles decimal commas
   (European locale exports), and detects whether the imaginary column is
   Z'' or -Z''.

Every decision the parser makes is recorded in ``ParseReport.notes`` so the user
can see (and override) it. Silent guessing is the failure mode we avoid.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

INSTRUMENTS = {
    "Auto-detect": None,
    "Generic table (CSV/TXT/TSV/Excel)": "generic",
    "BioLogic EC-Lab (.mpt)": "generic",
    "Gamry (.DTA)": "gamry",
    "ZView / ZPlot (.z)": "zplot",
    "VersaStudio (text export)": "versastudio",
    "Parstat": "parstat",
    "Autolab": "autolab",
    "CH Instruments": "chinstruments",
    "PowerSuite": "powersuite",
}

EXTENSION_READERS = {".dta": "gamry", ".z": "zplot"}

IMAG_SIGN_OPTIONS = ("Auto", "Column is Z'' (signed)", "Column is -Z''")


@dataclass
class EISData:
    freq: np.ndarray
    Z: np.ndarray
    name: str = "spectrum"

    @property
    def n(self) -> int:
        return int(self.freq.size)

    def cropped(self, f_min=None, f_max=None, drop_inductive=False) -> "EISData":
        m = np.ones(self.n, dtype=bool)
        if f_min is not None:
            m &= self.freq >= f_min
        if f_max is not None:
            m &= self.freq <= f_max
        if drop_inductive:
            m &= self.Z.imag <= 0
        return EISData(self.freq[m], self.Z[m], self.name)


@dataclass
class ParseReport:
    reader: str = ""
    columns: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    n_sweeps: int = 1
    raw_header: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Column recognition
# --------------------------------------------------------------------------- #
_FREQ = re.compile(r"(^|[^a-z])(freq|frequency|f)\b|^f\s*[/(\[]|hz", re.I)
_RE = re.compile(r"re\s*\(\s*z\s*\)|z\s*'(?!')|z\s*′(?!′)|\bzreal\b|\bz_?re(al)?\b|\breal\b|z\s*re\b", re.I)
_IM = re.compile(r"im\s*\(\s*z\s*\)|z\s*''|z\s*\"|z\s*″|\bzimag\b|\bz_?im(ag)?\b|\bimag(inary)?\b|z\s*im\b", re.I)
_MOD = re.compile(r"\|\s*z\s*\||\bzmod\b|\bmod(ulus)?\b|\bmagnitude\b|abs\s*\(\s*z", re.I)
_PHASE = re.compile(r"phase|\bzphz\b|\bphz\b|theta|\bangle\b", re.I)
_ADMIT = re.compile(r"\(\s*y\s*\)|\by'|admit", re.I)


def _is_negated(label: str) -> bool:
    s = label.strip().lower().replace(" ", "")
    return s.startswith("-") or s.startswith("−") or "-im" in s or "-z''" in s or "-z\"" in s or "-zim" in s


def _classify(headers):
    """Return {role: index} for freq / re / im / mod / phase from header strings."""
    roles = {}
    for i, h in enumerate(headers):
        hs = str(h).strip()
        if not hs or _ADMIT.search(hs):
            continue
        low = hs.lower()
        # order matters: imag before real because "z''" contains "z'"
        if "im" not in roles and _IM.search(low):
            roles["im"] = i
        elif "re" not in roles and _RE.search(low) and not _IM.search(low):
            roles["re"] = i
        elif "mod" not in roles and _MOD.search(low):
            roles["mod"] = i
        elif "phase" not in roles and _PHASE.search(low):
            roles["phase"] = i
        elif "freq" not in roles and _FREQ.search(low):
            roles["freq"] = i
    return roles


# --------------------------------------------------------------------------- #
# Generic table detection
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            txt = raw.decode(enc)
            if enc == "utf-16" and "\x00" in txt:
                continue
            return txt
        except UnicodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _split(line: str, delim):
    if delim is None:
        return line.split()
    return [t.strip() for t in line.split(delim)]


def _to_float(tok: str, decimal_comma: bool):
    t = tok.strip().strip('"')
    if decimal_comma:
        if "," in t and "." in t:  # e.g. 1.234,5 (thousands separator)
            t = t.replace(".", "")
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def _numeric_count(tokens, decimal_comma):
    return sum(1 for t in tokens if t and _to_float(t, decimal_comma) is not None)


def _find_table(text: str):
    """Find the largest block of numeric rows; return (rows, header, delim, decimal_comma)."""
    lines = [ln for ln in text.splitlines()]
    best = None
    for delim in ("\t", ";", ",", None):
        for decimal_comma in (False, True):
            if decimal_comma and delim == ",":
                continue
            run_start, run = None, []
            blocks = []
            for idx, ln in enumerate(lines):
                if not ln.strip():
                    if run:
                        blocks.append((run_start, run))
                        run_start, run = None, []
                    continue
                toks = _split(ln, delim)
                vals = [_to_float(t, decimal_comma) for t in toks]
                nnum = sum(v is not None for v in vals)
                if nnum >= 3 and nnum >= 0.6 * len([t for t in toks if t != ""]):
                    if run_start is None:
                        run_start = idx
                    run.append(toks)
                else:
                    if run:
                        blocks.append((run_start, run))
                    run_start, run = None, []
            if run:
                blocks.append((run_start, run))
            for start, rows in blocks:
                ncol = int(np.median([len(r) for r in rows]))
                score = len(rows) * min(ncol, 12)
                # a decimal-comma reading that turns "1,5" into a float only
                # wins if the plain reading was not already numeric
                if best is None or score > best[0]:
                    best = (score, start, rows, delim, decimal_comma, ncol)
    if best is None:
        return None
    _, start, rows, delim, decimal_comma, ncol = best
    header = None
    # look upward for a header row with (roughly) the same number of tokens
    for back in range(1, 6):
        j = start - back
        if j < 0:
            break
        toks = _split(lines[j], delim)
        if len(toks) >= 3 and _numeric_count(toks, decimal_comma) == 0:
            cls = _classify(toks)
            if cls:
                header = toks
                break
    return rows, header, delim, decimal_comma, ncol


def _rows_to_array(rows, ncol, decimal_comma):
    out = np.full((len(rows), ncol), np.nan)
    for i, r in enumerate(rows):
        for j in range(min(ncol, len(r))):
            v = _to_float(r[j], decimal_comma)
            if v is not None:
                out[i, j] = v
    return out


def _excel_to_text(raw: bytes) -> str:
    df = pd.read_excel(io.BytesIO(raw), header=None, sheet_name=0)
    return df.to_csv(sep="\t", index=False, header=False)


def _generic(raw: bytes, filename: str, report: ParseReport, colmap=None):
    ext = os.path.splitext(filename)[1].lower()
    text = _excel_to_text(raw) if ext in (".xlsx", ".xls") else _decode(raw)
    found = _find_table(text)
    if found is None:
        raise ValueError("No numeric table with at least 3 columns was found in the file.")
    rows, header, delim, decimal_comma, ncol = found
    arr = _rows_to_array(rows, ncol, decimal_comma)
    report.reader = "generic table"
    report.raw_header = list(header) if header else [f"col {i}" for i in range(ncol)]
    if decimal_comma:
        report.notes.append("Decimal commas detected (European locale export); converted to points.")

    roles = _classify(header) if header else {}
    if colmap:  # explicit user mapping wins
        roles = {k: v for k, v in colmap.items() if v is not None}
        report.notes.append("Using the column mapping you selected.")
    elif not header:
        report.notes.append("No recognisable header: assuming columns are f, Z', Z''.")
        roles = {"freq": 0, "re": 1, "im": 2}

    if "freq" not in roles:
        raise ValueError(f"Could not identify a frequency column. Header seen: {report.raw_header}")
    f = arr[:, roles["freq"]]
    explicit_sign = None
    if "re" in roles and "im" in roles:
        re_ = arr[:, roles["re"]]
        im_ = arr[:, roles["im"]]
        if header and not colmap:
            explicit_sign = -1 if _is_negated(str(header[roles["im"]])) else None
            if explicit_sign == -1:
                im_ = -im_
                report.notes.append(f"Header '{header[roles['im']].strip()}' indicates -Z''; sign converted.")
        Z = re_ + 1j * im_
    elif "mod" in roles and "phase" in roles:
        mod = arr[:, roles["mod"]]
        ph = arr[:, roles["phase"]]
        if np.nanmax(np.abs(ph)) > 2 * np.pi + 0.1:
            ph = np.deg2rad(ph)
            report.notes.append("Phase column read in degrees.")
        else:
            report.notes.append("Phase column read in radians (all |values| <= 2π). Check this.")
        if header and _is_negated(str(header[roles["phase"]])):
            ph = -ph
        Z = mod * np.exp(1j * ph)
        report.notes.append("Impedance built from |Z| and phase.")
    else:
        raise ValueError(f"Could not identify Re/Im or |Z|/phase columns. Header seen: {report.raw_header}")
    report.columns = {k: report.raw_header[v] for k, v in roles.items() if v < len(report.raw_header)}
    return f, Z, explicit_sign


def _vendor(raw: bytes, filename: str, instrument: str):
    from impedance import preprocessing

    suffix = os.path.splitext(filename)[1] or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        path = tmp.name
    try:
        f, Z = preprocessing.readFile(path, instrument=instrument)
    finally:
        os.unlink(path)
    return np.asarray(f, float), np.asarray(Z, complex)


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def split_sweeps(freq: np.ndarray):
    """Split concatenated sweeps (e.g. multi-cycle BioLogic files) into index arrays.

    A new sweep starts wherever log10(f) jumps by more than one decade *against*
    the dominant sweep direction.
    """
    if freq.size < 4:
        return [np.arange(freq.size)]
    d = np.diff(np.log10(freq))
    direction = np.sign(np.median(d)) or -1.0
    breaks = np.where(d * direction < -1.0)[0] + 1
    edges = np.concatenate([[0], breaks, [freq.size]])
    return [np.arange(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a >= 3]


def load_spectrum(raw: bytes, filename: str, instrument_label: str = "Auto-detect",
                  imag_sign: str = "Auto", colmap: dict | None = None):
    """Parse an uploaded file. Returns (list_of_EISData_sweeps, ParseReport)."""
    report = ParseReport()
    instrument = INSTRUMENTS.get(instrument_label)
    ext = os.path.splitext(filename)[1].lower()
    if instrument is None:
        instrument = EXTENSION_READERS.get(ext, "generic")

    explicit_sign = None
    if instrument != "generic" and colmap is None:
        try:
            f, Z = _vendor(raw, filename, instrument)
            report.reader = f"impedance.py readFile(instrument='{instrument}')"
            explicit_sign = 1  # vendor readers return signed Z''
        except Exception as exc:  # fall back to the table detector
            report.notes.append(f"Vendor reader '{instrument}' failed ({type(exc).__name__}: {exc}); "
                                "fell back to the generic table detector.")
            f, Z, explicit_sign = _generic(raw, filename, report, colmap)
    else:
        f, Z, explicit_sign = _generic(raw, filename, report, colmap)

    ok = np.isfinite(f) & np.isfinite(Z.real) & np.isfinite(Z.imag) & (f > 0)
    if (~ok).sum():
        report.notes.append(f"Dropped {(~ok).sum()} rows with non-finite values or f <= 0.")
    f, Z = f[ok], Z[ok]
    if f.size < 5:
        raise ValueError("Fewer than 5 valid points after parsing.")

    # imaginary sign convention
    if imag_sign == "Column is -Z''":
        Z = Z.real - 1j * Z.imag if explicit_sign != -1 else Z
        report.notes.append("Imaginary column treated as -Z'' (user override).")
    elif imag_sign == "Column is Z'' (signed)":
        report.notes.append("Imaginary column treated as signed Z'' (user override).")
    elif explicit_sign is None:
        if np.median(Z.imag) > 0:
            Z = Z.real - 1j * Z.imag
            report.notes.append("Most imaginary values were positive, so the column was taken as -Z'' and "
                                "converted. Override with 'Imaginary sign' if this is wrong.")

    sweeps = split_sweeps(f)
    report.n_sweeps = len(sweeps)
    if len(sweeps) > 1:
        report.notes.append(f"File contains {len(sweeps)} frequency sweeps; choose one to analyse.")

    out = []
    base = os.path.splitext(os.path.basename(filename))[0]
    for k, idx in enumerate(sweeps):
        fs, Zs = f[idx], Z[idx]
        order = np.argsort(fs)[::-1]
        fs, Zs = fs[order], Zs[order]
        _, uniq = np.unique(fs, return_index=True)
        if uniq.size < fs.size:
            keep = np.sort(uniq)
            report.notes.append(f"Sweep {k+1}: removed {fs.size-uniq.size} duplicate frequencies.")
            fs, Zs = fs[keep], Zs[keep]
            order = np.argsort(fs)[::-1]
            fs, Zs = fs[order], Zs[order]
        name = base if len(sweeps) == 1 else f"{base}_sweep{k+1}"
        out.append(EISData(fs, Zs, name))
    return out, report


def table_preview(raw: bytes, filename: str):
    """Header candidates for the manual column-mapping UI."""
    ext = os.path.splitext(filename)[1].lower()
    text = _excel_to_text(raw) if ext in (".xlsx", ".xls") else _decode(raw)
    found = _find_table(text)
    if not found:
        return []
    rows, header, *_ , ncol = found
    return list(header) if header else [f"col {i}" for i in range(ncol)]
