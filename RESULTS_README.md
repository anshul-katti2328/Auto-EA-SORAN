# Auto-EASORAN Experiment Results and Paper Notes

## 1. Purpose of this document

This document records the current experimental implementation, the results obtained from the matched deterministic and AI runs, the formulas used to calculate each KPI, and the correct interpretation of the generated figures. It is intended to give every project member a common, reproducible source of values for the research paper.

The current results demonstrate a successful end-to-end pilot. They are based on one matched random seed (`seed=42`) and therefore must not yet be presented as statistically conclusive. The principal result is that the AI-controlled system reduced represented network energy by **6.946%**, while changing aggregate throughput by only **-0.006%**. The AI run also increased latency, packet loss, and PRB utilization, so these trade-offs must be reported together with the energy improvement.

## 2. Experimental objective

The project compares two energy-conservation approaches on the same Simu5G network and traffic conditions:

1. **Deterministic/rule-based control**
   - Monitoring and Keras energy prediction remain active.
   - Energy-saving actions are selected from operator-defined rules.
   - The LLM healing path and Feedback Agent are not used to select actions.
   - The resulting configuration is sent through the Orchestration Agent and physically applied to the simulated gNB PHY.

2. **AI-based autonomous control**
   - Simu5G telemetry is processed by the Keras ANN predictor.
   - The Optimization Agent searches for energy-saving configurations.
   - The LLM healing path can construct or refine a healing plan.
   - The Feedback Agent validates safety and policy compliance.
   - The Orchestration Agent applies an approved plan to the simulated gNB PHY.
   - A minimum-benefit guard rejects changes whose predicted energy saving is below `0.5%`.

The comparison is intended to answer whether autonomous AI control can reduce RAN energy without materially sacrificing delivered traffic or violating operational policies.

## 3. Implemented simulation system

### 3.1 Network scale

- **30 genuine Simu5G NR gNBs**
- **90 genuine Simu5G NR UEs**
- **3 UEs placed around each gNB**
- Multiple city zones and radio-unit profiles
- Runtime cell association and genuine Simu5G traffic/KPI signals
- Physical control of gNB transmit power and supported energy-saving state

This is a substantial increase from the original seven-base-station topology. The 30-cell configuration was selected as a credible city-scale research pilot that remained executable on the available laptop.

### 3.2 Logical day compression

The experiment represents a complete 24-hour traffic cycle in 60 seconds of physical simulation time:

```text
logical time scale = 1440 logical seconds / physical second
60 physical seconds = 86,400 logical seconds = 24 hours
```

This compression makes repeated experiments feasible while retaining the intended daily sequence of very-low, low, medium, and peak traffic conditions. It does not mean the simulator runs in real time; OMNeT++ still executes every scheduled discrete event.

### 3.3 Traffic model

The simulated daily traffic profile includes:

- very-low overnight demand;
- morning/office-hour load growth;
- moderate afternoon demand;
- high evening demand representing events such as major sports matches;
- a return to low overnight demand.

The final matched files contain all four labels:

```text
very_low, low, medium, peak
```

### 3.4 End-to-end control path

The implemented path is:

```text
Simu5G runtime signals
    -> TwinMetricsAgent
    -> ZeroMQ telemetry
    -> Keras Prediction Agent
    -> Optimization Agent
    -> deterministic rules OR AI/LLM healing
    -> Feedback Agent validation in AI mode
    -> Orchestration Agent
    -> OrchestrationShim
    -> ControllableGnbPhy
    -> physical Simu5G state
```

The orchestration reply reports the requested and physically applied state, including transmit power, bandwidth, carrier count, sleep state, cell index, and simulation application time. This prevents the experiment from claiming savings based only on a Python-side decision that never reached the simulator.

## 4. Evaluation data and matching procedure

### 4.1 Raw final runs

The original final runs are stored in:

```text
OUTPUTs/experiments/paired_seed42_final/
```

The AI collector started one snapshot earlier than the deterministic collector. Comparing the unaligned files would include AI time `t=3 s` but start deterministic data at `t=4 s`, biasing the energy comparison.

### 4.2 Matched evaluation set

The paper-facing matched files are stored in:

```text
OUTPUTs/experiments/paired_seed42_final_matched/
```

Both scenarios contain:

- physical simulation time `4 s` through `60 s`;
- 57 common snapshots;
- 30 cells per snapshot;
- 1,710 cell rows per scenario;
- seed 42 and replicate 1;
- no unassociated UEs.

All headline comparisons in this document use this common window. The unaligned 4.215% result must not be used; the correct matched energy-saving result is **6.946%**.

## 5. Headline matched results

| KPI | Deterministic | AI | Interpretation |
|---|---:|---:|---|
| Represented energy | 54.3601 kWh | 50.5842 kWh | AI used 6.946% less represented energy |
| Average network power | 2384.21 W | 2218.60 W | AI reduced average power by approximately 165.61 W |
| Average network throughput | 66.9394 Mbit/s | 66.9356 Mbit/s | Essentially unchanged; -0.006% |
| Energy efficiency | 0.028076 Mbit/J | 0.030170 Mbit/J | AI improved aggregate data-per-energy efficiency by approximately 7.46% |
| Traffic-weighted latency | 2.933 ms | 3.934 ms | AI increased latency by approximately 34.1% |
| Aggregate packet loss | 0.000629% | 0.006290% | Increase of 0.005661 percentage points; absolute value remains low |
| Weighted PRB utilization | 2.458% | 3.332% | Higher utilization under AI; resources were used more intensively |
| Average associated UEs | 90 | 90 | UE service population remained unchanged |
| Sleeping cell-time | 2.749% | 3.099% | AI spent slightly more represented cell-time in full sleep |

### 5.1 Main conclusion

For this seed, the AI controller delivered a favorable energy/throughput trade-off: **6.946% energy saving with practically zero throughput reduction**. The increased energy efficiency indicates that the network delivered almost the same data volume using less energy.

The result is not a claim of free improvement. Latency rose by about 1 ms, packet loss increased by a small absolute amount, and PRB utilization increased. These observations suggest that the AI controller consolidated traffic and operated the remaining active radio resources more intensively. A paper should describe this as an energy/QoS trade-off rather than stating that every KPI improved.

### 5.2 Energy saving by traffic condition

| Traffic condition | AI energy saving relative to deterministic |
|---|---:|
| Very low | 8.558% |
| Low | 3.444% |
| Medium | 3.191% |
| Peak | 5.562% |

The strongest saving occurred during very-low traffic, where the controller had the greatest freedom to reduce radio energy. Positive savings were also retained during peak demand, although the controller remained constrained by service-protection requirements.

## 6. Energy-saving mode behavior

The final AI matched run contained:

| Applied AI mode | Cell-snapshot count | Share of 1,710 observations |
|---|---:|---:|
| Active | 70 | 4.09% |
| Light sleep | 1,587 | 92.81% |
| Sleep | 53 | 3.10% |

The deterministic run contained 211 active, 1,452 light-sleep, and 47 sleep observations. The AI controller therefore used light sleep more frequently and left fewer cell observations fully active.

In the AI run, all low, medium, and peak observations were represented by light sleep. The active and full-sleep observations occurred during very-low traffic. This shows that the control policy protected service availability under increased demand and reserved the deepest transitions for the lowest-load period.

### 6.1 Why BWP and Micro-DTX may be absent

BWP and Micro-DTX remain candidate modes in the optimization space. They are not guaranteed to appear in the applied-state graph. Before deployment, the system compares predicted candidate energy against the predicted current-state baseline. A candidate is rejected when:

```text
predicted saving < 0.5%
```

or when the candidate is predicted to consume more energy than the current configuration. Therefore, the absence of BWP or Micro-DTX in an applied-mode chart means that they did not pass the minimum-benefit guard in this run. It does not prove that those modes are universally ineffective.

This behavior should be described as a **prediction-backed minimum-benefit deployment constraint**. The optimizer and ANN provide the candidate evaluation, while the explicit guard prevents negative-benefit changes and configuration churn. It is intelligent pipeline behavior, but not an emergent decision learned independently by the LLM.

## 7. KPI definitions and formulas

### 7.1 Represented energy

Each physical measurement interval represents a longer logical interval. Total represented network energy is:

```text
represented energy (kWh)
    = sum(represented_energy_j) / 3,600,000
```

The same represented interval is used for both scenarios after matching.

### 7.2 Average network power

```text
average network power (W)
    = total represented energy (J)
      / represented run duration (s)
```

### 7.3 Network throughput

```text
network throughput (Mbit/s)
    = represented received data volume (bits)
      / represented duration (s)
      / 1,000,000
```

The analysis uses delivered downlink and uplink data counters rather than requested traffic alone.

### 7.4 Energy efficiency

```text
energy efficiency (Mbit/J)
    = represented received data volume (Mbit)
      / represented energy consumption (J)
```

This follows the NG-RAN data-energy-efficiency form in 3GPP TS 28.554: data volume divided by energy consumption over the same measurement period. The standardized unit is bit/J; the plots use Mbit/J for readability.

Reference: [ETSI TS 128 554 / 3GPP TS 28.554 Release 18](https://www.etsi.org/deliver/etsi_ts/128500_128599/128554/18.06.00_60/ts_128554v180600p.pdf).

### 7.5 Aggregate packet loss

```text
aggregate packet loss (%)
    = 100 * max(0, offered bits - received bits)
      / offered bits
```

This is the study's aggregate simulation formula derived from offered and delivered traffic. It should be described as a 3GPP-aligned reliability measurement, not as a direct counter collected from commercial equipment.

### 7.6 Traffic-weighted latency

```text
traffic-weighted latency (ms)
    = sum(cell latency * represented received bits)
      / sum(represented received bits)
```

Weighting by received traffic prevents an almost-idle cell from influencing the network-wide result as strongly as a busy cell.

### 7.7 PRB utilization

The network-wide time-weighted PRB utilization is:

```text
PRB utilization (%)
    = 100 * sum(used PRBs * represented interval)
      / sum(available PRBs * represented interval)
```

3GPP TS 28.552 defines NR PRB-usage measurements in terms of used PRBs relative to available PRBs and identifies them as measurements for monitoring radio physical-layer load. Our formula is a time-weighted aggregation of the corresponding Simu5G counters across cells and intervals.

Reference: [ETSI TS 128 552 / 3GPP TS 28.552 Release 18](https://www.etsi.org/deliver/etsi_ts/128500_128599/128552/18.11.00_60/ts_128552v181100p.pdf).

### 7.8 Sleeping cell-time

```text
sleeping cell-time (%)
    = 100 * represented time with radio_sleeping=true
      / total represented cell-time
```

This metric measures full sleeping time. Light sleep is reported separately in the ES-mode distribution and must not be silently counted as full radio sleep.

## 8. Generated KPI figures

The final matched analysis is stored in:

```text
OUTPUTs/experiments/analysis_seed42_final_matched/
```

### 8.1 `kpi_comparison.png`

This multi-panel bar chart compares deterministic and AI control for represented energy, average power, throughput, energy efficiency, latency, and packet loss.

Paper interpretation:

- lower AI energy and power bars demonstrate energy reduction;
- almost identical throughput bars show preserved traffic delivery;
- higher AI energy efficiency demonstrates more data delivered per joule;
- higher latency and packet-loss bars expose the QoS trade-off.

The current confidence intervals are zero because only one run per scenario is present. These are not statistical error bars and must not be described as evidence of low variance.

### 8.2 `hourly_profiles.png`

This figure shows network power, throughput, PRB utilization, and latency across the logical 24-hour day. It is the primary figure for relating network operation to the compressed daily traffic cycle.

Paper interpretation should focus on whether power follows traffic demand, whether AI reduces power during low-load periods, and whether latency or PRB utilization rises during consolidation.

### 8.3 `es_modes_by_traffic.png`

This stacked chart reports the percentage of represented cell-time spent in each applied ES mode under very-low, low, medium, and peak traffic.

It shows actual applied configurations—not optimizer candidates. Modes rejected by policy validation, physical feasibility checks, or the minimum-benefit guard are intentionally absent.

### 8.4 `energy_throughput_tradeoff.png`

This plot places the scenarios in energy-versus-throughput space. A preferable operating point moves toward lower energy without moving materially toward lower throughput. With only one seed, it contains one point per scenario and should be treated as a pilot visualization.

### 8.5 CSV and metadata files

- `run_summary.csv`: one row per scenario run;
- `scenario_summary.csv`: mean, standard deviation, and 95% CI fields;
- `paired_comparison.csv`: matched energy saving and throughput change;
- `formula_metadata.json`: machine-readable formulas and standards notes.

## 9. SHAP explainability

### 9.1 Scope

SHAP explains the **Keras ANN energy predictor**. It does not directly explain:

- the deterministic rule engine;
- the LLM's text reasoning;
- Feedback Agent policy logic;
- the entire multi-agent system as one model.

Rule decisions should be explained using rule names and decision counts. LLM plans should be explained through their plan/audit records. Applying SHAP indiscriminately to these components would be methodologically incorrect.

### 9.2 SHAP experiment configuration

The final SHAP artifacts are stored in:

```text
OUTPUTs/shap/paper_seed42_final/
```

Configuration:

- method: Kernel SHAP;
- model: `CODE/models/energy_prediction_keras.keras`;
- background set: 100 representative training rows;
- explanation cohort: 24 stratified rows;
- seed: 42;
- sampling: balanced across load and active/energy-saving strata;
- SHAP base value: approximately 29.85 W.

### 9.3 Global feature importance

The leading mean absolute SHAP values were:

| Rank | Feature | Mean absolute SHAP value |
|---:|---|---:|
| 1 | RUType | 8.384 |
| 2 | load_smoothed | 2.714 |
| 3 | TXpower | 2.190 |
| 4 | Bandwidth | 0.796 |
| 5 | ESMode6 | 0.788 |
| 6 | hour_spline_3 | 0.736 |
| 7 | Frequency | 0.685 |
| 8 | hour_spline_1 | 0.682 |
| 9 | Antennas | 0.588 |
| 10 | ESMode1 | 0.477 |

The model is most sensitive to radio-unit hardware class, followed by smoothed load and transmit power. Bandwidth, energy-saving modes, time of day, frequency, and antenna configuration also contribute.

`RUType` is a categorically encoded feature. A larger numeric category code does not mean a physically larger or better RU. The code is non-ordinal, so only the magnitude of its importance should be discussed unless individual categories are decoded separately.

### 9.4 SHAP figures

- `ann_energy_shap_1_importance_bar.png`: global mean absolute contribution of the most important features;
- `ann_energy_shap_2_beeswarm.png`: direction and distribution of feature contributions across the cohort;
- `ann_energy_shap_3_waterfall_sample1.png`: local explanation for one representative prediction;
- `ann_energy_shap_4_pos_neg_contributors.png`: features increasing and decreasing that representative prediction;
- `ann_energy_shap_5_heatmap.png`: contribution patterns across all 24 explained samples.

The beeswarm horizontal position indicates whether a feature increased or decreased the predicted energy relative to the base value. Colour indicates transformed feature value. Colour direction must not be interpreted ordinally for categorical codes.

## 10. Code and execution efficiency

### 10.1 Improvements made

The implementation now includes several efficiency and reliability mechanisms:

- 30 cells are predicted as a batch instead of as unrelated one-cell model calls;
- prediction messages are drained in bounded batches to reduce ZeroMQ backlog;
- non-blocking polls prevent feedback sockets from stalling prediction processing;
- the Optimization Agent uses bounded NSGA population and generation counts;
- AI healing plans have a finite guard, preventing unlimited repeated plans;
- the Orchestrator REQ socket is reset after timeouts, avoiding ZeroMQ EFSM lockup;
- the Simu5G shim polls more frequently for commands;
- SHAP is disabled during ordinary simulation and run offline for selected samples;
- the minimum-benefit gate avoids physically applying low-value configuration churn;
- a 24-hour traffic cycle is evaluated in 60 physical simulation seconds.

Observed prediction tests processed a complete 30-cell batch in roughly 0.4-0.5 seconds on the development laptop. This is sufficiently below the logical decision interval for the present experiment. It is an implementation observation, not a platform-independent performance guarantee.

### 10.2 Simulation cost

The Simu5G model remains computationally intensive because 90 UEs and 30 gNBs generate a large number of discrete events. Logical time compression reduces experiment duration but does not eliminate event-processing cost. Laptop feasibility should therefore be reported in terms of the tested 30-gNB configuration rather than claiming arbitrary scalability.

### 10.3 Control efficiency versus energy efficiency

These terms should not be conflated:

- **energy efficiency** is the measured data volume per joule;
- **control efficiency** refers to batching, bounded optimization, asynchronous messaging, and avoiding unnecessary actions;
- **simulation speed** is the number of discrete events or simulated seconds processed per wall-clock second.

The paper's primary quantitative efficiency claim is the measured improvement in energy efficiency, not merely faster Python execution.

## 11. Reproduction commands

Run the matched KPI analysis from `CODE`:

```bash
python -u analyze_experiments.py \
  --input-dir ../OUTPUTs/experiments/paired_seed42_final_matched \
  --output-dir ../OUTPUTs/experiments/analysis_seed42_final_matched
```

Generate the paper SHAP artifacts:

```bash
python -u generate_shap_plots.py \
  --data ../archive/merged_training_data.csv \
  --output-dir ../OUTPUTs/shap/paper_seed42_final \
  --samples 24 \
  --background-samples 100 \
  --seed 42
```

## 12. Valid and invalid claims for the paper

### Valid pilot claims

- An end-to-end autonomous energy-control loop was implemented and physically connected to Simu5G gNB state.
- The topology was scaled to 30 gNBs and 90 UEs.
- A complete logical daily traffic profile was simulated.
- For matched seed 42, AI control reduced represented energy by 6.946%.
- Throughput changed by only -0.006% in that matched run.
- Aggregate energy efficiency improved by approximately 7.46%.
- The improvement involved measurable latency, packet-loss, and PRB-utilization trade-offs.
- SHAP identified RU type, load history, and transmit power as the strongest ANN energy-prediction influences.

### Claims that are not yet supported

- The result is statistically significant.
- The 6.946% saving generalizes to every seed, topology, traffic profile, or real network.
- AI improved every KPI.
- A missing ES mode is universally ineffective.
- SHAP explains the LLM or entire multi-agent system.
- The analytical energy model is identical to a commercial base-station power meter.

## 13. Work required for enhanced results. (over-optimisation & to be checked)

1. Run at least five matched seeds per scenario.
2. Ensure identical physical timestamps and BS identifiers before every comparison.
3. Report mean, standard deviation, 95% confidence interval, and paired effect size.
4. Confirm that no runs contain unassociated UEs, orchestration timeouts, rejected physical applications, or stale telemetry.
5. Preserve raw data, configuration, model hash/version, and logs for each run.
6. Use the same ANN, topology, traffic schedule, and policy thresholds in paired scenarios.
7. Recreate the plots from the multi-seed directory and replace the single-seed pilot figures in the final paper.

## 14. Short paper-ready summary

> A 30-gNB, 90-UE Simu5G testbed was used to compare deterministic and autonomous AI-based RAN energy control over a compressed 24-hour traffic cycle. In the matched seed-42 pilot, the AI controller reduced represented energy consumption from 54.36 kWh to 50.58 kWh, corresponding to a 6.95% saving, while aggregate throughput changed by only -0.006%. Data energy efficiency improved by approximately 7.46%. The energy gain was accompanied by increased traffic-weighted latency, packet loss, and PRB utilization, demonstrating a measurable energy/QoS trade-off. Kernel SHAP analysis of the Keras energy predictor identified radio-unit type, smoothed load, transmit power, bandwidth, and energy-saving state as the most influential prediction features. These results establish end-to-end feasibility but remain a single-seed pilot pending multi-seed statistical validation.
