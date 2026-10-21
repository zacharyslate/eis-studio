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

Complex non-linear least squares with impedance.py's circuit model and the same SciPy `curve_fit` call as its `circuit_fit`, with one change: the trust-region steps are scaled by the Jacobian (`x_scale="jac"`), so parameters of very different magnitude (a CPE Q of 10⁻¹⁰ next to an R of 10³) are all optimised; with the default unit scaling small parameters barely move from their starting values. Starting values put each R–C/CPE pair at an observed maximum of −Z″ (or just above the window when an arc is cut off at high frequency). Weighting options: modulus (σ = |Z|), proportional (σ = |Z′|, |Z″|), unit. Multi-start refits from log-perturbed initial values (×10^±1.5; CPE exponents in [0.5, 1]) and keeps the lowest objective. If several starts reach equally good minima with different parameter values, those parameters are reported as **not identifiable**.

Pseudo-χ² = Σ[(Δ_re)² + (Δ_im)²], modulus-weighted, in the sense used by Boukamp (1995).

For parallel R–CPE pairs: C_eff = Q^(1/α) R^((1−α)/α), f_c = 1/[2π(RQ)^(1/α)] (Hsu & Mansfeld 2001). For R–C pairs, f_c = 1/(2πRC).

### DRT (pyDRTtools)

Z(f) = R∞ + j2πfL + ∫ γ(ln τ)/(1 + j2πfτ) d ln τ. Solved by ridge (Tikhonov) regression with radial-basis-function discretisation and a non-negativity constraint (Wan et al. 2015; Saccoccio et al. 2014). λ is chosen by GCV, mGCV, rGCV, L-curve or Re–Im discrepancy, or entered manually (Maradesa et al. 2023). The criterion (pyDRTtools' own function, on the run's matrices) is evaluated on a grid over λ = 10⁻⁷…1 and refined around the best point; pyDRTtools itself minimises it with SLSQP from one starting value, which can stop at that value or in a local minimum. k-fold cross-validation is not offered: it fails inside pyDRTtools 0.2 with a matrix-size error. The optional credible band samples the truncated-Gaussian posterior by Hamiltonian Monte Carlo (pyDRTtools `Bayesian_run`; Ciucci & Chen 2015; Effat & Ciucci 2017).

* **Peak resistance** = area under the peak over ln τ, split at the minima of γ. **Peak frequency** f = 1/(2πτ).
* **R_pol** = ∫γ d ln τ over the whole computed τ range (which extends half a decade beyond the data on each side; the shaded bands in the γ plot).
* **What the DRT cannot do.** A capacitive tail from blocking electrodes has no finite DC resistance, so it violates the DRT model and produces a large artefact at long τ: crop it. Small satellite peaks next to a large peak are typical regularisation artefacts of depressed (CPE-like) arcs. Never interpret a peak from a single spectrum without checking how it responds to temperature, atmosphere or bias.

**Wrapper fixes for pyDRTtools 0.2:** its basis functions are centred at τ = 1/f while its kernel uses 2πfτ, so it can only place peaks at characteristic frequencies between f_min/2π and f_max/2π; a process in the top ~0.8 decade of the measured window is pinned to the edge and R∞ absorbs part of it. EIS Studio centres the basis functions at τ = 1/(2πf). The package imports its Qt GUI on import (stubbed here); `induct_used = 2` ("discard inductive data") behaves like 0 (we remove Z″ > 0 points ourselves); the "Re Data" branch references a misspelled attribute (worked around); in "Im Data" mode R∞ is estimated after the fit.

**Fix for impedance.py 1.7.1:** `linKK` evaluates a string built from NumPy reprs, which fails under NumPy ≥ 2 (`np.float64(...)` is undefined in its eval namespace); a corrected `eval_linKK` is patched in.

### Conductivity

σ = L/(R·A), with L the thickness and A the electrode area (π d²/4 for a disc). Relative uncertainties add in quadrature: (δσ/σ)² = (δL/L)² + (δR/R)² + (δA/A)², with δA/A = 2 δd/d. For a solid-electrolyte pellet at room temperature the bulk arc often lies above the highest measured frequency. Then "R0" (ECM) or R∞ (DRT) *is* the bulk (or bulk + grain-boundary) resistance, and the high-frequency intercept should be reported as such, not as a contact resistance.

### Series analysis

*Reading.* Every numeric table in a file is read with its own header. Rows with a frequency > 0 and an impedance are spectra; rows with time and current are time series (BioLogic writes f = 0 on rows of other techniques in linked experiments). Measurements are separated where a step or technique index changes (`Ns`, `cycle number`, `Segment`, `ActionId`), where the frequency jumps back up by more than a decade, or where the time resets. Without an index column the potential record is split into plateaus: a point is flat when a 15-point running median varies by less than max(2 mV, 6 × the point-to-point noise) over 15 points; a short non-flat stretch between plateaus is one potential jump, and longer ones (relaxing open-circuit periods) are separate segments, so isolated noisy points never split a step. Units of current (A, mA, µA, nA, pA), potential (V, mV) and time are taken from the headers. A time series whose potential varies (5–95 % spread above max(5 mV, 2 %), or a linear drift above 2 mV) is an open-circuit or galvanostatic period, not a potentiostatic step; some instruments report a meaningless constant current during open circuit, which is therefore never used.

*Ionic resistance per spectrum.* (i) Equivalent-circuit fit: the sum of the chosen resistances. (ii) Z′ at the minimum of −Z″ between the arcs and the blocking spike (parabolic refinement over three points). (iii) The blocking spike extrapolated to −Z″ = 0 by a straight-line fit to its low-frequency two thirds. Method (ii) slightly overestimates R when the arc and the spike overlap; method (iii) depends on the spike being straight, and a spike far from vertical (CPE exponent well below 1) carries a model error. σ = L/(R·A) uses each sample's own thickness and diameter when given in the table.

*Activation energy.* Linear least squares of log₁₀(σT) (default) or log₁₀σ against 1000/T: Eₐ = −slope × 1000 k_B ln 10, k_B = 8.617333 × 10⁻⁵ eV K⁻¹; ±Eₐ is the standard error of the slope. σ at 25 °C is read from the fitted line and labelled extrapolated when 25 °C lies outside the measured temperatures. The σT form follows from the Nernst–Einstein relation; Eₐ from log σ is smaller by about k_BT.

*Electronic conductivity (DC polarisation).* With ion-blocking electrodes the ionic current decays after each potential step and the steady-state current is electronic. I_ss is the mean current over the last part of each step (20 % by default). Its uncertainty is the larger of the white-noise standard error and the standard error of six block means, so slowly wandering noise is not mistaken for precision. A step is *resolved* when |I_ss| exceeds three times its uncertainty, and *not at steady state* when the linear change over the window exceeds its own standard error threefold and both 5 % of I_ss and three times its uncertainty. A weighted straight line I_ss = E/R_e + I₀ (weights 1/uncertainty², optionally through the origin; uncertainties scaled by the reduced χ² when the scatter is larger) gives R_e and σ_e = L/(R_e·A). When the slope is less than twice its standard error, R_e is not determined and only an upper bound σ_e < (slope + 2 s.e.)·L/A is given. A quadratic term more than 2.5 standard errors from zero flags a curved I–V line; the electronic transference number t_e = σ_e/(σ_e + σ_ion) uses a chosen spectrum of the same pellet. The I–V line is only linear while the electrolyte's chemical potential stays in its stability window; Hebb–Wagner analysis (σ_e as a function of potential) is outside the scope of this linear treatment.

*Normalised plots.* Z·A/L (Ω cm) removes differences in thickness and area. For pellets of different relative density ρ_rel = m/(A·L)/ρ_theoretical, Z·(A/L)·ρ_rel^m gives a dense-equivalent resistivity under the assumption σ = σ_dense·ρ_rel^m (m = 1: linear in the solid fraction; m = 1.5: Bruggeman, spherical pores). Such corrections depend on pore shape and connectivity and are approximations; report the measured values with ρ_rel alongside. The same factor scales |Z| and γ of the DRT; the phase is unchanged.

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
