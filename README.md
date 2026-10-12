# EIS Studio

A web app for analysing electrochemical impedance spectra:

* **Upload** Gamry `.DTA`, ZView `.z`, BioLogic `.mpt` (including decimal-comma exports and multi-cycle files), VersaStudio/Parstat/Autolab/CH Instruments text exports, or any CSV/TXT/Excel table with frequency + Re/Im (or |Z| + phase). Column detection, sign convention and sweep splitting are automatic and reported; all of them can be overridden.
* **Validate** with the Lin-KK Kramers–Kronig test.
* **Equivalent-circuit fitting** (impedance.py): user-defined circuit, editable initial values, fixed parameters and bounds; modulus, proportional or unit weighting; multi-start fitting with an identifiability check; effective capacitance and characteristic frequency for R–CPE pairs.
* **DRT** (pyDRTtools): all RBF types, Re/Im/combined data, inductance handling, 1st/2nd-order regularisation, six λ-selection criteria or manual λ, optional Bayesian (HMC) credible band, automatic peak detection and integration.
* **Conductivity** σ = L/(R·A) from disc or slab geometry, with uncertainty propagation.
* **Outputs:** Nyquist (data + fit), Bode |Z| and phase (data + fit), residuals, and γ(ln τ) vs τ (or g(τ) = γ/τ) with a dual τ/f axis. Each comes as an individual figure and a lettered composite in PNG/TIFF (300–1200 dpi), PDF or SVG with editable text, plus CSV tables and a settings JSON (software versions included). A single ZIP holds everything.

The sign convention is Z = Z′ + jZ″ with Z″ < 0 for capacitive behaviour; plots show −Z″.

## Install (users)

Website with instructions, tutorial and demo video: **https://zacharyslate.github.io/eis-studio/**

| OS | Command |
|---|---|
| Windows (PowerShell) | `powershell -ExecutionPolicy ByPass -c "irm https://zacharyslate.github.io/eis-studio/install.ps1 \| iex"` |
| macOS / Linux | `curl -LsSf https://zacharyslate.github.io/eis-studio/install.sh \| sh` |

The installer uses [uv](https://docs.astral.sh/uv/) to set up a private Python 3.12 and EIS Studio in an isolated environment, with no administrator rights needed. It then adds an `eis-studio` command and a desktop/Start-menu shortcut. The app opens in the browser and listens only on `127.0.0.1`, so spectra never leave the computer. To update, rerun the same command; to uninstall, run `uv tool uninstall eis-studio`.

With an existing Python ≥ 3.10: `pip install "eis-studio @ https://github.com/zacharyslate/eis-studio/archive/refs/heads/main.zip"`, then `eis-studio`.

## Develop

```bash
python -m venv ZACH-venv && source ZACH-venv/bin/activate     # Windows: ZACH-venv\Scripts\activate
pip install -e ".[test]"
eis-studio                      # or: streamlit run app.py
```

| Path | What it is |
|---|---|
| `eisstudio/` | the package: parsers, ECM, DRT, conductivity, plotting, exports, the Streamlit UI (`app.py`), launcher (`cli.py`), sample data, methods text |
| `app.py` | entry point for hosted deployments (e.g. Streamlit Community Cloud) |
| `docs/` | the website, served by GitHub Pages, including `install.sh` / `install.ps1` |
| `site_src/`, `tools/` | website sources and build scripts: `python tools/make_hero.py && python tools/build_site.py` |
| `tests/` | validation against synthetic spectra with known parameters, plus app end-to-end tests |
| `.github/workflows/install-test.yml` | runs the real installers on Windows, macOS and Linux, launches the app and runs the tests on every push |

### Publishing the website

In the repository's **Settings → Pages**, set **Source: Deploy from a branch**, branch `main`, folder `/docs`. The site and the install scripts are then served from `https://zacharyslate.github.io/eis-studio/`.

### Optional: hosted version

The same repository also deploys to [Streamlit Community Cloud](https://share.streamlit.io) (main file `app.py`; `requirements.txt` and `packages.txt` are picked up automatically) for people who only want to try it. Uploaded data is then processed on Streamlit's servers, so the local install remains the recommended route for unpublished data.

## Tests

```bash
python tests/make_samples.py          # regenerates the synthetic files in eisstudio/sample_data/
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
