# Paper Draft Notes — Auto EA-SORAN (Current-State Review Draft)

**Title:** Auto EA-SORAN: A Safety-Gated AI-Native Closed-Loop Energy Optimization
Framework for 5G RANs in a Simu5G Digital Twin

**Status:** Current-state review draft. All quantitative claims correspond to the
**matched seed-42 pilot only**. Multi-seed validation has *not* been done and is
not claimed. The draft deliberately presents the result as a single-seed
feasibility pilot with an explicit energy/QoS trade-off.

**Deliverables in this folder:**
- `paper_main.tex` — IEEE conference LaTeX paper (uses the repo IEEE template class).
- `paper_main.pdf` — compiled output (7 pages; compiles cleanly with `pdflatex`,
  no undefined references or citations).
- `figures/` — copies of the five included figures (originals left untouched in
  the repository).
- `PAPER_DRAFT_NOTES.md` — this file.

To rebuild: `pdflatex paper_main.tex` (run twice for cross-references).

---

## 1. Repository files inspected and used

| File | Use in paper |
|---|---|
| `RESULTS_README.md` | Primary source of all headline KPIs, ES-by-traffic table, ES-mode distribution, KPI formulas, figure interpretations, and the explicit list of valid vs. unsupported claims. |
| `README.md` | ITU 5G Energy Consumption Modelling background; ANN predictor lineage (Related Work). |
| `newREADME.md` | Four-agent architecture, ZMQ ports/sockets, real-time loop, training/feature details. |
| `CODE/SIMU5G_INTEGRATION_ANALYSIS.md` | TwinMetricsAgent telemetry fields, partial actuation (which features are physically applied vs. default). Used for the Limitations section. |
| `CODE/Simu5G/.../src/TwinMetricsAgent.cc` | **Actual** load-coupled analytical energy model (lines ~236–349) and Bangalore multi-zone profiles (lines ~34–95). Source of Eq. (2) and the multi-zone topology claim. |
| `MyNetwork.ned`, NED files under `CODE/Simu5G/.../ned/` | Confirmed OrchestrationShim / OrchestrationAgent / ControllableGnbPhy module structure. |
| `OUTPUTs/experiments/analysis_seed42_final_matched/scenario_summary.csv` | Verified exact KPI values (energy, power, throughput, EE, latency, loss, PRB, sleeping-cell-time). |
| `OUTPUTs/experiments/analysis_seed42_final_matched/paired_comparison.csv` | Verified energy_saving_pct = 6.946113 and throughput_change_pct = -0.005661. |
| `OUTPUTs/experiments/analysis_seed42_final_matched/formula_metadata.json` | Machine-readable KPI formulas and 3GPP TS 28.552 / 28.554 standards notes. |
| `OUTPUTs/example_SHAPplots/shap_explanation_1_importance_bar.png` | Inspected visually to confirm the top global features before describing them. |
| `IEEE_Template/IEEE_template.txt` | IEEE `\documentclass[conference]{IEEEtran}` template; preamble, author block, table/figure environments, and `thebibliography` style adapted from it. |

---

## 2. Figures included (5)

All copied into `figures/` from the repository; originals untouched.

1. `kpi_comparison.png` — KPI comparison (energy/power down, throughput flat, EE up, latency/loss/PRB up).
2. `energy_throughput_tradeoff.png` — energy–throughput trade-off (supports EE claim).
3. `hourly_profiles.png` — temporal behavior over the logical 24-hour day.
4. `es_modes_by_traffic.png` — applied ES-mode distribution by traffic regime.
5. `shap_explanation_1_importance_bar.png` — global SHAP feature importance (main explainability figure).

Deliberately **excluded** per the brief: SHAP beeswarm, waterfall, heatmap, and
pos/neg-contributor plots (kept the draft focused). The pos/neg-contributor plot
was *not* added because there was no surplus space in a 7-page IEEE draft.

---

## 3. Numerical values used (verified against repo CSVs)

| KPI | Deterministic | AI | Change |
|---|---:|---:|---:|
| Represented energy (kWh) | 54.3601 | 50.5842 | −6.946% |
| Avg. network power (W) | 2384.21 | 2218.60 | −165.6 W |
| Avg. throughput (Mbit/s) | 66.9394 | 66.9356 | −0.006% |
| Energy efficiency (Mbit/J) | 0.028076 | 0.030170 | +7.46% |
| Traffic-weighted latency (ms) | 2.933 | 3.934 | +34.1% |
| Aggregate packet loss (%) | 0.000629 | 0.006290 | +0.00566 pp |
| Weighted PRB utilization (%) | 2.458 | 3.332 | +0.874 pp |
| Sleeping cell-time (%) | 2.749 | 3.099 | +0.350 pp |
| Avg. associated UEs | 90 | 90 | unchanged |

ES by traffic regime: very-low 8.558%, low 3.444%, medium 3.191%, peak 5.562%.
ES-mode counts (AI run, of 1,710 obs): active 70 (4.09%), light sleep 1,587
(92.81%), sleep 53 (3.10%); deterministic: 211 active, 1,452 light sleep, 47 sleep.

All values are taken verbatim from `scenario_summary.csv` and `paired_comparison.csv`
and match `RESULTS_README.md`.

---

## 4. Equations included

1. Linear load-coupled power model `P = P_idle + α·u_PRB(t)` — Eq. (1).
2. **Full implemented power model** (static + dynamic + antenna + baseband +
   carrier terms, ×overhead, ×ES factor) — Eq. (2). Transcribed from the actual
   `TwinMetricsAgent::calculateEnergy()` in `TwinMetricsAgent.cc`. Symbol names
   generalized; exact zone constants (e.g. P_static=130 W, P_max=200 W, δ=4.7 for
   CBD macro) were **not** hard-coded into the paper because they vary per zone.
3. Represented energy accumulation `E = Σ P(t)Δt ≡ ∫P dt` — Eq. (3).
4. Energy saving `ΔE% = (E_base − E_opt)/E_base ×100` — Eq. (4).
5. Energy efficiency `EE = Throughput/E` (Mbit/J, 3GPP TS 28.554 form) — Eq. (5).
6. QoS safety-guard set `a_t ∈ A_safe ⇔ {p95 thresholds}` — Eq. (6).
7. Constrained objective `min E_t(a_t) s.t. QoS_i ≤ θ_i` — Eq. (7).
8. SHAP additive attribution `f(x) = φ_0 + Σ φ_i` + mean|SHAP| ranking — Eq. (8).

**MAE/RMSE were intentionally omitted.** The repository's matched-pilot KPI
outputs do not report ANN prediction MAE/RMSE for this experiment (the ITU
notebook reports MAPE/WMAPE for the original Zindi competition, not for the
Simu5G loop), so per the brief these equations were left out rather than
fabricated. See open item below.

---

## 5. Assumptions made

- **Multi-zone topology:** "Bangalore-inspired multi-zone" is confirmed directly
  in `TwinMetricsAgent.cc` (CBD / IT-park / residential zone profiles), so it is
  stated as fact, not hedged.
- **Compressed 24-hour profile:** confirmed in `RESULTS_README.md` (1440 logical
  s per physical s; 60 s = 24 h). Stated as fact.
- **SHAP feature description:** describes the features visible in
  `shap_explanation_1_importance_bar.png` (hour_spline_2, Bandwidth,
  hour_spline_1, load_smoothed, weekday_number, load_spline_6, TXpower, …). NOTE:
  this example plot's ranking differs from the `RESULTS_README.md` §9.3 table
  (which is a *different* SHAP run topped by `RUType`). The paper uses the
  example-plot features because that is the figure included, per the brief.
- **QoS guard thresholds (Eq. 6):** the repo describes a policy-compliance score
  (≥0.95) and a critical-violation count, plus a 0.5% minimum-benefit guard. The
  p95 throughput/latency/loss threshold form is presented generically (θ values
  left symbolic) because exact per-policy numeric thresholds were not extracted
  from `policies.py`/`feedbackAGENT.py` for this draft. Marked as configurable.
- **Optimizer:** described as "bounded evolutionary / population-based search."
  The repo mentions DEAP genetic algorithms, differential evolution (pop 30,
  100 iters), and NSGA — kept general to avoid over-specifying.
- **Author block / affiliations / title footnote:** left as template placeholders
  to be filled in by the team.

---

## 6. Open items requiring manual verification before submission

1. **Fill author names, affiliations, emails** (currently template placeholders).
2. **Reference metadata:** the bibliography uses best-effort IEEE formatting.
   Verify exact details for: Simu5G (b_simu5g — IEEE Access 2020, authors/pages
   given but please confirm), 3GPP TS 28.552/28.554 release/version, ITU
   challenge citation, and SHAP (NeurIPS 2017). URLs are placeholders/landing
   pages — confirm or replace with stable links. Do not submit unverified DOIs.
3. **SHAP figure vs. table mismatch:** decide whether to (a) keep the example
   importance-bar figure (current choice) or (b) regenerate the importance bar
   from `OUTPUTs/shap/paper_seed42_final/` so the figure matches the seed-42
   §9.3 ranking (RUType-topped). If (b), update the SHAP discussion text.
4. **Eq. (2) constants:** if reviewers want concrete numbers, add a small table
   of zone parameters (P_static, P_max, δ, antennas, bandwidth) from
   `TwinMetricsAgent.cc::initializeZoneProfiles()`.
5. **QoS guard thresholds (Eq. 6):** extract real numeric thresholds from
   `policies.py` / `feedbackAGENT.py` if you want them stated explicitly.
6. **MAE/RMSE:** if the ANN's matched-loop prediction error becomes available,
   add the standard MAE/RMSE equations and report the values.
7. **Template guidance text:** confirmed removed (no IEEE boilerplate/red warning
   text remains in `paper_main.tex`).

---

## 7. Claims explicitly avoided (per the brief and `RESULTS_README.md` §12)

- No statistical significance / multi-seed generalization claimed.
- No 15–25% savings claimed (the demo loop's 15% figure in `newREADME.md` is an
  illustrative single-cycle example, **not** the validated result; only the
  matched 6.946% is used).
- Not claimed that all KPIs improved (latency, loss, PRB explicitly worsen).
- No real-world hardware energy measurement claimed ("represented energy" /
  "load-coupled analytical model" used throughout).
- No production-ready RAN deployment claimed.
- SHAP framed as explaining the energy predictor only, not the LLM or whole system.
