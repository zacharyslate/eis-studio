### Conventions

* Impedance: **Z = Z′ + jZ″**, with Z″ < 0 for capacitive behaviour. Plots show −Z″. Every export carries both `Zim_ohm` (signed) and `minus_Zim_ohm`.
* Frequencies are sorted high → low after import. Files containing several sweeps (e.g. EC-Lab cycles) are split; pick one in the sidebar.
* Residuals are relative to the modulus: Δ_re = (Z′ − Z′_fit)/|Z|, Δ_im = (Z″ − Z″_fit)/|Z|, in %.

### Recommended workflow

1. **Validate the data** with the Lin-KK test (Data tab). Non-KK-consistent data cannot be fitted meaningfully by any linear model.
2. **Crop the frequency window**. Remove inductive/cable artefacts at the top end and, for DRT, any blocking-electrode tail at the bottom end.
3. **DRT first, ECM second.** Use the DRT to count processes and locate their time constants without assuming a circuit, then build the circuit with that many elements.
4. **Report the fit quality** (residual plot, pseudo-χ²) together with the parameters.

### ECM (impedance.py)

Complex non-linear least squares via `impedance.models.circuits.fitting.circuit_fit` (SciPy `curve_fit`). Weighting options: modulus (σ = |Z|), proportional (σ = |Z′|, |Z″|), unit. Multi-start refits from log-perturbed initial values (×10^±1.5; CPE exponents in [0.5, 1]) and keeps the lowest objective. If several starts reach equally good minima with different parameter values, those parameters are reported as **not identifiable**.

Pseudo-χ² = Σ[(Δ_re)² + (Δ_im)²], modulus-weighted, in the sense used by Boukamp (1995).

For parallel R–CPE pairs: C_eff = Q^(1/α) R^((1−α)/α), f_c = 1/[2π(RQ)^(1/α)] (Hsu & Mansfeld 2001). For R–C pairs, f_c = 1/(2πRC).

### DRT (pyDRTtools)

Z(f) = R∞ + j2πfL + ∫ γ(ln τ)/(1 + j2πfτ) d ln τ. Solved by ridge (Tikhonov) regression with radial-basis-function discretisation and a non-negativity constraint (Wan et al. 2015; Saccoccio et al. 2014). λ is chosen by GCV, mGCV, rGCV, L-curve, Re–Im discrepancy or k-fold cross-validation, or entered manually (Maradesa et al. 2023). The optional credible band samples the truncated-Gaussian posterior by Hamiltonian Monte Carlo (pyDRTtools `Bayesian_run`; Ciucci & Chen 2015; Effat & Ciucci 2017).

* **Peak resistance** = area under the peak over ln τ, split at the minima of γ. **Peak frequency** f = 1/(2πτ).
* **R_pol** = ∫γ d ln τ over the whole computed τ range (which extends half a decade beyond the data on each side; the shaded bands in the γ plot).
* **What the DRT cannot do.** A capacitive tail from blocking electrodes has no finite DC resistance, so it violates the DRT model and produces a large artefact at long τ: crop it. Small satellite peaks next to a large peak are typical regularisation artefacts of depressed (CPE-like) arcs. Never interpret a peak from a single spectrum without checking how it responds to temperature, atmosphere or bias.

**Wrapper fixes for pyDRTtools 0.2:** the package imports its Qt GUI on import (stubbed here); `induct_used = 2` ("discard inductive data") behaves like 0 (we remove Z″ > 0 points ourselves); the "Re Data" branch references a misspelled attribute (worked around); in "Im Data" mode R∞ is estimated after the fit.

**Fix for impedance.py 1.7.1:** `linKK` evaluates a string built from NumPy reprs, which fails under NumPy ≥ 2 (`np.float64(...)` is undefined in its eval namespace); a corrected `eval_linKK` is patched in.

### Conductivity

σ = L/(R·A), with L the thickness and A the electrode area (π d²/4 for a disc). Relative uncertainties add in quadrature: (δσ/σ)² = (δL/L)² + (δR/R)² + (δA/A)², with δA/A = 2 δd/d. For a solid-electrolyte pellet at room temperature the bulk arc often lies above the highest measured frequency. Then "R0" (ECM) or R∞ (DRT) *is* the bulk (or bulk + grain-boundary) resistance, and the high-frequency intercept should be reported as such, not as a contact resistance.

### References

* B. A. Boukamp, A Linear Kronig-Kramers Transform Test for Immittance Data Validation, *J. Electrochem. Soc.* **142** (1995) 1885. [doi:10.1149/1.2044210](https://doi.org/10.1149/1.2044210)
* M. Schönleber, D. Klotz, E. Ivers-Tiffée, A Method for Improving the Robustness of linear Kramers-Kronig Validity Tests, *Electrochim. Acta* **131** (2014) 20–27. [doi:10.1016/j.electacta.2014.01.034](https://doi.org/10.1016/j.electacta.2014.01.034)
* C. H. Hsu, F. Mansfeld, Concerning the Conversion of the Constant Phase Element Parameter Y0 into a Capacitance, *Corrosion* **57** (2001) 747.
* J. T. S. Irvine, D. C. Sinclair, A. R. West, Electroceramics: Characterization by Impedance Spectroscopy, *Adv. Mater.* **2** (1990) 132–138. [doi:10.1002/adma.19900020304](https://doi.org/10.1002/adma.19900020304)
* T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, Influence of the discretization methods on the distribution of relaxation times deconvolution: Implementing radial basis functions with DRTtools, *Electrochim. Acta* **184** (2015) 483–499.
* M. Saccoccio, T. H. Wan, C. Chen, F. Ciucci, Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: Ridge and lasso regression methods, *Electrochim. Acta* **147** (2014) 470–482.
* A. Maradesa, B. Py, T. H. Wan, M. B. Effat, F. Ciucci, Selecting the Regularization Parameter in the Distribution of Relaxation Times, *J. Electrochem. Soc.* **170** (2023) 030502.
* F. Ciucci, C. Chen, Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach, *Electrochim. Acta* **167** (2015) 439–454.
* M. B. Effat, F. Ciucci, Bayesian and Hierarchical Bayesian Based Regularization for Deconvolving the Distribution of Relaxation Times from Electrochemical Impedance Spectroscopy Data, *Electrochim. Acta* **247** (2017) 1117–1129. [doi:10.1016/j.electacta.2017.07.050](https://doi.org/10.1016/j.electacta.2017.07.050)
* M. D. Murbach, B. Gerwe, N. Dawson-Elli, L. Tsui, impedance.py: A Python package for electrochemical impedance analysis, *J. Open Source Softw.* **5** (2020) 2349. [doi:10.21105/joss.02349](https://doi.org/10.21105/joss.02349)
* pyDRTtools: [github.com/ciuccislab/pyDRTtools](https://github.com/ciuccislab/pyDRTtools) (MIT licence). impedance.py: [impedancepy.readthedocs.io](https://impedancepy.readthedocs.io) (MIT licence).
