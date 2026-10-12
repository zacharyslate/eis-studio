# EIS Studio

A web app for analysing electrochemical impedance spectra:

* **Upload** Gamry `.DTA`, ZView `.z`, BioLogic `.mpt` (including decimal-comma exports and multi-cycle files), VersaStudio/Parstat/Autolab/CH Instruments text exports, or any CSV/TXT/Excel table with frequency + Re/Im (or |Z| + phase). Column detection, sign convention and sweep splitting are automatic and reported; all of them can be overridden.
* **Validate** with the Lin-KK Kramers–Kronig test.
* **Equivalent-circuit fitting** (impedance.py): user-defined circuit, editable initial values, fixed parameters and bounds; modulus, proportional or unit weighting; multi-start fitting with an identifiability check; effective capacitance and characteristic frequency for R–CPE pairs.
* **DRT** (pyDRTtools): all RBF types, Re/Im/combined data, inductance handling, 1st/2nd-order regularisation, six λ-selection criteria or manual λ, optional Bayesian (HMC) credible band, automatic peak detection and integration.
* **Conductivity** σ = L/(R·A) from disc or slab geometry, with uncertainty propagation.
* **Outputs:** Nyquist (data + fit), Bode |Z| and phase (data + fit), residuals, and γ(ln τ) vs τ (or g(τ) = γ/τ) with a dual τ/f axis. Each comes as an individual figure and a lettered composite in PNG/TIFF (300–1200 dpi), PDF or SVG with editable text, plus CSV tables and a settings JSON (software versions included). A single ZIP holds everything.

The sign convention is Z = Z′ + jZ″ with Z″ < 0 for capacitive behaviour; plots show −Z″.

## Run locally

```bash
python -m venv ZACH-venv
source ZACH-venv/bin/activate          # Windows: ZACH-venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                   # opens http://localhost:8501
```

## Deploy as a website from GitHub (Streamlit Community Cloud)

1. Push this folder to a GitHub repository (the `ZACH-venv/` folder is git-ignored).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app** and pick the repository, branch and `app.py`.
3. In **Advanced settings**, choose Python 3.12 or 3.13. `requirements.txt` installs the Python packages; `packages.txt` installs Liberation Sans, a metric-compatible substitute for Arial, so the figures use the intended font.

**Data privacy:** on Community Cloud, uploaded spectra are processed on Streamlit's servers. For unpublished or proprietary data, run it locally or on a lab server (`streamlit run app.py --server.address 0.0.0.0`) behind your institute's network. Nothing is stored after the session ends, but the upload still leaves your machine.

## Tests

```bash
python tests/make_samples.py          # regenerates the synthetic files in sample_data/
python -m pytest -q tests/            # 13 tests: parsers, ECM/DRT recovery of known parameters, app end-to-end
```

Every synthetic file has known parameters, documented in `tests/make_samples.py`. The tests check that:

* All five file formats parse to the same spectrum (< 1e-4 relative difference).
* The ECM recovers all seven parameters within 5 % and within ~4 standard errors.
* The DRT recovers R∞ (±3 Ω), the total polarisation resistance (±3 %), the individual arc resistances and the τ of the fast arc.
* An arc above the measured window, and a blocking-electrode tail, are both flagged.
* The conductivity arithmetic and error propagation are exact.
* The app runs end-to-end without errors.

## Upstream issues worked around (see `METHODS.md`)

| Library | Issue | Handling |
|---|---|---|
| pyDRTtools 0.2 | `import pyDRTtools` imports the PyQt5 GUI | GUI modules stubbed, so it runs headless |
| pyDRTtools 0.2 | `induct_used=2` behaves like 0 | inductive points (Z″ > 0) removed by the wrapper |
| pyDRTtools 0.2 | "Re Data" branch raises AttributeError (`lambda_lambda` typo) | attribute pre-set |
| pyDRTtools 0.2 | scikit-learn needed but undeclared | added to requirements |
| impedance.py 1.7.1 | `linKK` fails with NumPy ≥ 2 (`np.float64(...)` repr inside `eval`) | patched `eval_linKK` |
| matplotlib | not thread-safe; Streamlit renders sessions and downloads on separate threads | process-wide lock around all figure rendering |

## Licences

EIS Studio is released under the MIT licence (see `LICENSE`). impedance.py and pyDRTtools are also MIT-licensed.
