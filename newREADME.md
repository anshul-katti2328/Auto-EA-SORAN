Auto EA-SORAN: CODE Directory - Agent-Based 5G Energy Optimization System

Advanced Multi-Agent Framework for Real-Time 5G Base Station Energy Optimization with Policy Compliance

🏆 2nd Place Winner (Team : NoWiresAttached) - Build-a-thon 3.0: AI Native Bharat 5G Challenge, Indian Mobile Congress (IMC) 2025 🏆
🌐 Overview

The CODE directory contains a sophisticated multi-agent system for real-time 5G base station energy optimization. This system extends the ITU AI/ML baseline solution with intelligent agents that interact through ZeroMQ (ZMQ) messaging, forming a closed-loop optimization pipeline that continuously monitors, predicts, optimizes, and validates energy configurations while enforcing comprehensive 3GPP, Nokia, and hybrid policies.
Key Innovation: Agent Collaboration Pattern

Unlike traditional monolithic systems, our approach implements autonomous agents that specialize in specific tasks:

    Prediction Agent: Real-time energy consumption prediction using Keras/TensorFlow

    Optimization Agent: Multi-objective optimization with policy constraints using DEAP and FAISS

    Feedback Agent: Validation and policy compliance checking with rollback capabilities

    Orchestration Agent: Execution and OMNeT++ simulation integration

Real-Time Loop

text
Simu5G Simulation
    ↓
Prediction Agent (Energy forecast)
    ↓
Optimization Agent (Find best config)
    ↓
Feedback Agent (Validate policy compliance)
    ↓
Orchestration Agent (Apply changes)
    ↓
Simu5G Simulation (Verify results)
    ↓
[Loop continues...]

Message Flow

    Simu5G → Prediction Agent: Publishes unseen network metrics via ZMQ PUB

    Prediction Agent → Optimization Agent: Sends energy predictions via ZMQ PUSH

    Optimization Agent → Feedback Agent: Proposes optimized configurations

    Feedback Agent → Policy Manager: Validates against 3GPP/Nokia policies

    Feedback Agent → Orchestration Agent: Sends approved plans for execution

    Orchestration Agent → OrchestrationShim: Applies configuration changes to simulation

    OrchestrationShim → Simu5G: Updates base station parameters

    Loop repeats: TwinMetricsAgent measures impact, cycle continues

🤖 Agent System Overview
1. Prediction Agent (newPredictionAgent.py)

Role: Real-time energy consumption prediction on unseen simulation data

Key Features:

    Keras/TensorFlow model for energy prediction

    Handles unseen data from Simu5G TwinMetricsAgent

    SHAP-based explainability with batched processing

    Simu5G data format mapping and column name translation

    Real-time prediction with <50ms latency

Communication:

    SUB: Receives metrics from Simu5G (tcp://localhost:5556)

    REP: Responds to direct requests (tcp://0.0.0.0:5555)

    PUSH: Sends predictions to optimization agent (tcp://localhost:5558)

Workflow:

text
1. Load Keras model (energy_prediction_keras.keras)
2. Subscribe to Simu5G metrics stream
3. For each cell metric received:
   a. Map Simu5G columns to model features
   b. Extract features using FarziFeatureEngineer
   c. Make energy prediction
   d. Compute confidence score
   e. Send to optimization agent

2. Optimization Agent (OptimizationAgentwzmq.py)

Role: Multi-objective optimization of base station configurations with policy constraints

Key Features:

    Differential evolution for configuration search

    DEAP-based genetic algorithms for multi-population optimization

    FAISS similarity search for policy retrieval

    LLM-powered execution plan generation

    3GPP policy integration via PolicyManager

    Multi-objective optimization: minimize energy, minimize latency, maximize throughput

Communication:

    REP: Receives predictions (tcp://0.0.0.0:5558)

    REQ: Sends optimization responses (tcp://localhost:5560)

    PUB: Broadcasts optimized configs (tcp://0.0.0.0:5559)

Optimization Process:

text
1. Receive energy prediction from prediction agent
2. Load current base station configuration
3. Query applicable policies:
   a. 5QI QoS class policies
   b. Nokia-specific policies
   c. Energy efficiency policies
   d. Network-specific policies
4. Define optimization problem:
   - Variables: tx_power, bandwidth, carriers, es_modes
   - Objectives: minimize energy, maintain QoS
   - Constraints: policy limits, hardware limits
5. Run differential evolution:
   - Population size: 30
   - Iterations: 100
   - Mutation: 0.8
   - Crossover: 0.7
6. Extract Pareto-optimal solutions
7. Select best solution based on policy compliance
8. Generate LLM-powered execution template
9. Send to feedback agent for validation

3. Feedback Agent (feedbackAGENT.py)

Role: Validate energy conservation plans and enforce policy compliance

Key Features:

    Comprehensive 3GPP + Nokia policy validation

    Dynamic policy constraint extraction via FAISS

    Per-sample policy compliance scoring

    Automated rollback engine for policy violations

    KPI monitoring and alerting

    Critical violation detection and blocking

Communication:

    REQ: Receives optimization proposals (tcp://0.0.0.0:5560)

    PUSH: Sends validated plans (tcp://localhost:5562)

    SUB: Monitors network KPIs (tcp://localhost:5561)

Validation Pipeline:

text
1. Receive optimization proposal (OptimizationResponse)
2. Extract proposed configuration and constraints
3. Validate against applicable policies:
   a. Current state validation
   b. Predicted state validation
   c. Constraint violation detection
4. Generate validation report:
   - Policy compliance score (0-1)
   - Critical violations count
   - Per-policy violation details
5. Decision logic:
   - If policy_compliance_score >= 0.95: APPROVE
   - If critical_violations == 0: APPROVE
   - Else: REJECT with recommended adjustments
6. Send result to orchestration agent
7. Log validation metrics

4. Orchestration Agent (OrchestrationAgent.py)

Role: Execute approved configurations and manage Simu5G simulation state

Key Features:

    Direct integration with OrchestrationShim (OMNeT++)

    Command transformation for simulation parameter changes

    Healing plan execution with violation analysis

    Configuration application tracking

    Metrics logging and performance tracking

Communication:

    REP: Receives validated plans (tcp://0.0.0.0:5562)

    REQ: Sends commands to OrchestrationShim (tcp://localhost:6000)

Execution Process:

text
1. Receive healing plan from feedback agent
2. Analyze critical violations to determine actions:
   a. Power violations → reduce tx_power_dbm
   b. Latency violations → enable energy-saving mode
   c. Throughput issues → adjust carriers
3. Create OrchestrationShim-compatible command:
   {
     "bs_id": "BS_1",
     "delta_tx_db": -5.0,
     "tx_power_dbm": 43.0,
     "es_mode": "sleep"
   }
4. Send to OrchestrationShim via ZMQ REQ socket
5. Wait for acknowledgment
6. Log metrics (success/failure/timeout)
7. Return result to feedback agent

🔄 Execution Flow
Step-by-Step: Complete Optimization Cycle

Prerequisites:

    Simu5G simulation running with TwinMetricsAgent enabled

    OMNeT++ OrchestrationShim listening on port 6000

    All agents configured with correct ports

Execution:

bash
# Terminal 1: Start Simu5G simulation
cd CODE/validation/simu5g/
./run_simulation.sh

# Terminal 2: Start Prediction Agent
cd CODE/
python newPredictionAgent.py

# Terminal 3: Start Optimization Agent
python OptimizationAgentwzmq.py

# Terminal 4: Start Feedback Agent
python feedbackAGENT.py

# Terminal 5: Start Orchestration Agent
python OrchestrationAgent.py

Real-Time Loop (Single Cycle):

text
Time T0: Simu5G publishes metrics
├─ Cell A: load=45%, tx_power=30dBm, es_mode=normal
├─ Cell B: load=75%, tx_power=40dBm, es_mode=active
└─ Cell C: load=20%, tx_power=25dBm, es_mode=sleep

Time T0+10ms: Prediction Agent receives metrics
├─ Maps "prb_util_pct" → "load"
├─ Extracts features using FarziFeatureEngineer
├─ Runs Keras inference: 3 predictions
├─ Cell A: predicted_energy=1450W (confidence=0.94)
├─ Cell B: predicted_energy=1820W (confidence=0.91)
└─ Cell C: predicted_energy=620W (confidence=0.96)

Time T0+25ms: Optimization Agent receives predictions
├─ Query 3GPP 5QI policies for active services
├─ Set optimization objectives:
│  ├─ Minimize energy consumption
│  ├─ Maintain latency < 100ms
│  └─ Keep throughput > 50 Mbps
├─ Run differential evolution (100 iterations)
├─ Find Pareto-optimal solutions
└─ Select best: energy_saving=15%, policy_score=0.98

Time T0+150ms: Feedback Agent receives optimization
├─ Validate against Nokia energy policies
├─ Check 3GPP QoS requirements
├─ Compute policy_compliance_score=0.99
├─ critical_violations=0
├─ DECISION: APPROVE
└─ Generate healing plan

Time T0+160ms: Orchestration Agent receives healing plan
├─ Analyze violations for power/latency issues
├─ Create shim command: {bs_id, delta_tx_db=-3, es_mode="active"}
├─ Send to OrchestrationShim
├─ Receive: {accepted=true, applied_at_sim_s=142.5}
└─ Log: HEALING_PLAN_APPLIED

Time T0+200ms: OrchestrationShim applies changes
├─ Update Cell A: tx_power 30→27dBm, es_mode normal→active
├─ Continue simulation with new parameters
└─ TwinMetricsAgent measures impact

Time T0+500ms: Simu5G reports new metrics
└─ Cycle repeats with updated network state

🔧 Installation & Setup
Prerequisites

Hardware:

    CPU: 4+ cores

    RAM: 8GB+ (recommended 16GB)

    GPU: Optional but recommended for Keras inference

Software:

    Ubuntu 20.04+ or similar Linux

    Python 3.8+

    OMNeT++ 6.0+

    Simu5G module compiled

Step 1: Clone Repository

bash
cd /path/to/Auto-EA-SORAN/CODE/

Step 2: Create Virtual Environment

bash
python3 -m venv venv
source venv/bin/activate

Step 3: Install Dependencies

All Python dependencies are specified in requirements.txt. Install them using:

bash
pip install -r requirements.txt

This will install:

    Core ML and data processing libraries

    Deep learning frameworks (TensorFlow/Keras)

    Distributed computing and ZMQ messaging

    Optimization algorithms (DEAP, FAISS)

    Visualization and analysis tools

    Configuration management and utilities

Step 4: Model Training

Train the Keras energy prediction model:

bash
python kerasTRAINER.py

# Expected output:
# ✓ Loaded 102,705 rows
# ✓ After cleaning: 98,432 rows
# ✓ Feature engineer saved
# ✓ Model built with 845,312 parameters
# ✓ Training complete!
# ✓ Model saved to: models/energy_prediction_keras.keras

Step 5: Verify Installation

bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import zmq; print(f'ZMQ: {zmq.__version__}')"
python -c "import faiss; print('FAISS: OK')"

▶️ Running the System
Manual Agent Startup (Recommended)

Terminal 1: Prediction Agent

bash
python newPredictionAgent.py
# Output: "PredictionAgent prediction_agent_keras_001 initialized"

Terminal 2: Optimization Agent

bash
python OptimizationAgentwzmq.py
# Output: "OptimizationAgent optimization_agent_001 initialized"

Terminal 3: Feedback Agent

bash
python feedbackAGENT.py
# Output: "Enhanced Feedback Agent feedback_agent_001 initialized"

Terminal 4: Orchestration Agent

bash
python OrchestrationAgent.py
# Output: "Listening for commands on tcp://*:5562"

Verifying Agent Operation

Each agent generates logs during execution. Check the output directory (discussed in Output & Diagnostics) for detailed logs, performance metrics, and visualization artifacts.
📖 Agent Descriptions
Communication Patterns

Synchronous (REQ-REP):

    Request waits for reply

    Used between sequential agents (Prediction → Optimization → Feedback → Orchestration)

    Example: Optimization Agent requests feedback validation

Asynchronous (PUB-SUB):

    One-to-many broadcasting

    Used for metrics distribution

    Example: Simu5G broadcasts metrics to all interested agents

Task Queue (PUSH-PULL):

    Work distribution

    Used for load balancing

    Example: Batch predictions sent to optimization

Port Mapping
Agent	Component	Port	Type	Direction
Prediction	Request Listener	5555	REP	← Request
Prediction	Response Sender	(dynamic)	PUSH	→ Response
Simu5G	Metrics Publisher	5556	PUB	→ Broadcast
Optimization	Request Listener	5558	REP	← Request
Optimization	Config Publisher	5559	PUB	→ Broadcast
Feedback	Request Listener	5560	REP	← Request
Orchestration	Request Listener	5562	REP	← Request
OrchestrationShim	Command Listener	6000	REP	← Command
State Management

Each agent maintains:

    Current state: Last received configuration/metrics

    History: Previous 100 optimization cycles

    Metrics: Performance tracking (latency, success rate, violations)

    Cache: Recently used policies, models, feature engineers

🎯 Key Features
1. Multi-Policy Enforcement

The system validates against comprehensive policy sets:

3GPP 5QI Policies (QoS Class Identifiers):

    Conversational Voice (5QI-1): max latency 100ms

    Real-time Gaming (5QI-3): max latency 50ms

    URLLC Discrete Automation (5QI-82): max latency 10ms

Nokia Proprietary Policies:

    Power constraints per product type

    Thermal management limits

    MIMO-specific efficiency requirements

Energy Efficiency Policies:

    Daily energy consumption targets

    Peak power limits

    Energy-saving mode effectiveness thresholds

Custom Network Policies:

    Dense deployment constraints

    IoT service requirements

    Autonomous vehicle (V2X) latency needs

2. Real-Time Prediction

    Keras model inference: <50ms per prediction

    Batch prediction capability: 1000+ predictions/sec

    Confidence scoring for result reliability

    SHAP explainability with feature importance

3. Advanced Optimization

    Algorithm: Differential Evolution + DEAP

    Population: 30 individuals

    Generations: 100+ (adaptive)

    Multi-objective: Energy, latency, throughput

    Pareto optimization: Multiple non-dominated solutions

4. Policy-Driven Healing

When violations detected:

    Analyze root causes

    Generate recommended adjustments

    Create healing plan

    Validate against constraints

    Execute changes gradually

    Monitor impact

5. Closed-Loop Feedback

    Continuous monitoring of optimization impact

    Automatic rollback if QoS degraded

    Learning-based policy adjustment

    Trend detection for proactive optimization

💻 Technical Stack

Machine Learning:

    Keras/TensorFlow 2.12.0 (energy prediction)

    DEAP (genetic algorithms)

    FAISS (policy similarity search)

    SHAP (explainability)

    Scikit-learn (preprocessing)

Distributed Computing:

    ZeroMQ (async messaging)

    Python async/await (concurrency)

    Multi-process agents

5G Simulation:

    OMNeT++ 6.0+

    Simu5G module

    TwinMetricsAgent (real-time metrics)

    OrchestrationShim (parameter control)

Utilities:

    YAML (configuration)

    JSON (messaging)

    CSV (logging)

    NumPy/Pandas (data handling)

📊 Performance & Results
Prediction Agent
Metric	Value
Latency	<50ms
Throughput	1,200 predictions/sec (single)
Batch Throughput	52,800 predictions/sec
Model Accuracy (OOF MAPE)	2.59%
Memory	2.1GB
Optimization Agent
Metric	Value
Convergence Time	150-300ms
Population Size	30 individuals
Generations	100
Pareto Solutions	5-10 per cycle
Policy Compliance	>99%
Feedback Agent
Metric	Value
Validation Time	50-100ms
Policy Checks	40+ per proposal
Compliance Rate	99.7%
Violations Blocked	100% of critical
False Positive Rate	<0.1%
Orchestration Agent
Metric	Value
Command Latency	<100ms
Success Rate	98.5%
Applied Plans	95%+ of proposals
Rollback Rate	<1%
Downtime	0
System-Wide
Metric	Value
Full Cycle Time	500ms-1s
Energy Saving	12-18%
Latency Impact	+2-5ms
Throughput Impact	-1-3%
Policy Violations	<0.1%

📞 Contact & Support

Project Maintainer: Rishiraj Kumar
GitHub: @Rishi8520
Repository: Auto-EA-SORAN

Team Members:

    Rishiraj Kumar - Feedback and Policy Validation
    
    Anirudh Nishtala - Simu5G Integration

    Tarunika D. - Model Training & Validation

    Sreelakshmi Anil - Orchestration and back-deployment simu5G

For Issues:

    GitHub Issues: Technical problems and feature requests

    Email: Direct collaboration inquiries

    Discussion Board: Implementation questions

📄 License

MIT License - Educational and research use
Citation

text
@software{auto_ea_soran_code_2025,
  title={Auto EA-SORAN: Multi-Agent 5G Energy Optimization Framework (CODE Directory)},
  author={Rishiraj Kumar and Anirudh Nishtala and Tarunika D. and Sreelakshmi Anil},
  year={2025},
  url={https://github.com/Rishi8520/Auto-EA-SORAN},
}

Built with PyTorch, TensorFlow, ZMQ, and OMNeT++ | Optimizing 5G Energy Consumption in Real-Time

Last Updated: 16th November 2025

---

# 2026 Implementation Update: Scalable Simu5G Closed-Loop Validation

> **Status note:** This section records the implementation and measurements validated during the 2026 Simu5G integration work. Where an older example, port, topology size, runtime claim, or projected performance figure above differs from this section, the validated information below takes precedence. The earlier material has been retained as project history.

## What Was Improved

The project has moved from a prototype agent demonstration to a physically connected Simu5G closed loop with reproducible deterministic-versus-AI experiments. The principal improvements are:

- replacement of the earlier placeholder seven-cell topology with 30 genuine Simu5G NR gNodeBs;
- deployment of 90 genuine NR UEs, with three UEs positioned around every gNB;
- a city-style topology containing dense urban and lower-density zones;
- a compressed 24-hour traffic cycle containing very-low, low, medium, and peak conditions;
- real Simu5G runtime telemetry for PRB allocation, throughput, latency, packet loss, SINR, UE association, and radio state;
- a runtime-controllable gNB PHY for physical transmit-power and deep-sleep application;
- structured acknowledgements that distinguish requested configuration from physically applied configuration;
- separate deterministic/rule-based and AI-based experimental modes;
- batched 30-cell Keras inference with uncertainty estimation;
- bounded AI optimization, policy validation, and minimum-benefit deployment protection;
- experiment collection, matched-window KPI analysis, and scenario comparison plots;
- offline and live Kernel SHAP explainability workflows;
- safeguards for ZeroMQ timeout recovery, stale telemetry, message backlog, active-UE protection, and repeated healing plans.

## Canonical Simu5G Project

The maintained simulation project is:

```text
CODE/Simu5G/auto-easoran-sim/
```

Important components include:

| File | Purpose |
|---|---|
| `ned/autoeasoran/zmqonly/AutoEASORANNetwork.ned` | Defines the genuine 30-gNB, 90-UE Simu5G topology and core network. |
| `ned/autoeasoran/zmqonly/TwinMetricsAgent.ned` | Declares runtime telemetry parameters. |
| `ned/autoeasoran/zmqonly/ControllableGnbPhy.ned` | Extends the native Simu5G gNB PHY with runtime controls. |
| `src/TwinMetricsAgent.cc/.h` | Collects and publishes runtime cell metrics and represented energy. |
| `src/ControllableGnbPhy.cc/.h` | Applies transmit-power and deep-sleep state to the actual simulated PHY. |
| `src/OrchestrationShim.cc/.h` | Receives commands from Python and applies them to the requested gNB. |
| `src/ShimRegistry.h` | Shares requested/applied per-BS configuration between orchestration and telemetry. |
| `omnetpp.ini` | Configures topology scale, applications, UE placement, PHY type, traffic, and ZeroMQ endpoints. |
| `demo.xml` | Provides IPv4 interface configuration for the expanded topology. |

Obsolete prototype NED, INI, shim, and registry files that previously existed directly under `CODE/` are not the canonical simulation sources.

## Validated Network Scale

The current default experiment uses:

```text
30 gNodeBs
90 NR UEs
3 UEs per gNodeB
20 MHz physical bandwidth
30 kHz subcarrier spacing
51 configured PRBs per cell
```

The telemetry publisher emits one ordered entry per cell. BS identifiers `1001` through `1030` correspond to Simu5G cell indices `0` through `29`.

UEs use stationary mobility for reproducible paired experiments. Their positions are distributed around the corresponding gNB locations instead of being placed at a single common coordinate. Dynamic cell association is enabled and the collector records the count of unassociated UEs so invalid runs can be rejected.

## Compressed Daily Traffic Model

The simulation represents a complete logical day in 60 physical simulation seconds:

```text
logicalTimeScale = 1440
1 physical second = 1440 logical seconds = 24 logical minutes
60 physical seconds = 86,400 logical seconds = 24 hours
```

The traffic schedule models:

- very-low overnight demand;
- morning and office-hour growth;
- moderate afternoon traffic;
- evening peak demand representing matches or large public events;
- a return to low overnight demand.

The logical-time compression reduces experiment turnaround time but does not remove Simu5G discrete-event processing cost. A 60-second simulated run may still take several minutes of wall-clock time on a laptop.

## Updated End-to-End Architecture

```text
Simu5G gNBs and UEs
        |
        v
TwinMetricsAgent -- PUB 5556 --> Prediction Agent
                                         |
                                         v
                              Optimization Agent
                                /             \
                    deterministic rules       AI optimization + LLM healing
                                                    |
                                                    v
                                             Feedback Agent
                                                    |
                                                    v
                                      Orchestration Agent -- 5563
                                                    |
                                                    v
                                      OrchestrationShim -- 6000
                                                    |
                                                    v
                                      ControllableGnbPhy
                                                    |
                                                    v
                               TwinMetricsAgent measures the result
```

The control loop is considered successful only when the shim acknowledges that the target gNB was resolved and the supported physical state was applied.

## Current ZeroMQ Port Map

The active YAML and OMNeT++ configuration uses:

| Port | Component | Purpose |
|---:|---|---|
| 5555 | Prediction Agent | Direct prediction request listener. |
| 5556 | TwinMetricsAgent | Simu5G metrics PUB stream. |
| 5557 | OrchestrationShim | Physical-application status PUB stream. |
| 5558 | Optimization Agent | Prediction input receiver. |
| 5559 | Optimization Agent | Feedback result receiver. |
| 5560 | Feedback Agent | Validation request receiver. |
| 5561 | Feedback Agent | Feedback status publisher. |
| 5563 | Orchestration Agent | Approved configuration command listener. |
| 5564 | Optimization Agent | Optimization status publisher. |
| 6000 | OrchestrationShim | Simu5G command REP endpoint. |

The YAML configuration is authoritative when an older source-code default or earlier documentation table differs.

## Deterministic and AI Experimental Modes

### Deterministic/rule-based mode

Deterministic mode uses monitoring and Keras prediction but selects configurations from explicit operator rules in `rule_based_controller.py`. It does not use LLM plan generation or Feedback Agent validation for decision selection.

The rule result includes:

- selected transmit power;
- bandwidth and carrier request;
- named energy-saving mode;
- six ES-mode flags;
- rule name;
- human-readable reason.

The controller protects cells with associated UEs from inappropriate deep sleep and prioritizes QoS recovery when latency or loss exceeds configured limits.

### AI-based mode

AI mode uses:

- Keras energy predictions and confidence;
- bounded multi-objective configuration search;
- LLM-assisted healing-plan construction;
- Feedback Agent policy and QoS validation;
- physical orchestration through the shim;
- post-application monitoring.

The current optimizer uses bounded population and generation settings suitable for the laptop experiment. It also limits repeated healing plans so one telemetry snapshot cannot trigger an unbounded sequence of LLM calls.

### Minimum-benefit deployment guard

AI candidates are not automatically deployed. The predicted saving is compared with the predicted current-state baseline:

```text
deploy only when predicted energy saving >= 0.5%
```

Candidates such as BWP adaptation or Micro-DTX may therefore be evaluated but skipped when they provide negative or negligible predicted benefit. This is a prediction-backed safety/economic guard, not evidence that those modes are universally ineffective.

## Physical Application Semantics

The physical Simu5G implementation currently supports:

- bounded gNB transmit-power control;
- physical deep-sleep state;
- restoration to active operation.

Bandwidth, carrier-count, MIMO-reduction, BWP, Micro-DTX, and RF-chain/light-sleep requests are retained in control and telemetry state. Where Simu5G 1.4.1 does not expose an equivalent runtime PHY mutation, replies and telemetry distinguish the requested state from the fixed physical bandwidth/carrier state.

An acknowledgement contains fields such as:

```json
{
  "accepted": true,
  "applied_count": 1,
  "bs_id": "1001",
  "cell_index": 0,
  "tx_power_dbm": 41.0,
  "physical_tx_power_dbm": 41.0,
  "radio_sleeping": false,
  "physical_bandwidth_mhz": 20.0,
  "physical_carrier_count": 1,
  "es_mode": "active",
  "es_mode_physically_applied": true
}
```

## Telemetry and Energy Accounting

`TwinMetricsAgent` publishes one snapshot containing all 30 cells. Important fields include:

- physical and logical simulation time;
- BS and cell index;
- zone, gNB type, RU profile, carrier and antenna metadata;
- available and mean used PRBs;
- normalized load and PRB utilization;
- generated and delivered traffic;
- active UE count;
- SINR, latency, packet loss, and spectral efficiency;
- requested and physical transmit power;
- radio sleeping state and six ES-mode flags;
- interval power and energy;
- represented interval and represented energy;
- data energy efficiency;
- traffic level and traffic multiplier.

Mean used PRBs may be fractional because allocation is averaged across TTIs in the reporting interval. This is more accurate than rounding every interval to an integer before calculating utilization.

The current energy source is explicitly labelled `analytical_load_model`; it must not be described as a direct commercial power-meter measurement.

## Keras Prediction Improvements

The Prediction Agent now:

- receives complete Simu5G cell snapshots;
- maps runtime telemetry to the trained Farzi-team ANN schema;
- assigns one of 30 validated hardware profiles to each gNB;
- performs one 30-cell Keras batch during normal operation;
- reports MC-dropout confidence/uncertainty;
- discards stale snapshots;
- preserves request and simulation-time identifiers;
- provides optional live Kernel SHAP explanations.

Observed 30-cell prediction batches completed in approximately 0.4-0.5 seconds on the development laptop. This observation replaces the earlier synthetic throughput claims above and is not a platform-independent guarantee.

## SHAP Explainability

### Offline model-level SHAP

`generate_shap_plots.py` generates seeded, reproducible Kernel SHAP explanations for the saved Keras ANN. The validated paper run used:

```text
24 explained samples
100 sampled background records
45 transformed ANN features
seed 42
```

These plots explain general ANN behavior on a stratified modelling cohort. They do not compare AI control with deterministic control.

### Live Simu5G SHAP validation

Live SHAP was also enabled for a short Simu5G-only smoke test using:

```yaml
features:
  use_shap: true
  shap_batch_size: 30
```

Two complete 30-cell runtime batches were successfully explained. Both batches consistently ranked RU type, load spline, antenna count, smoothed load, lagged load, and transmit power among the strongest ANN drivers.

Live SHAP explains real-time Simu5G samples, while its reference distribution still comes from `merged_training_data.csv`. The current live initializer reads the first 100 modelling rows and uses a 50-row subset for Kernel SHAP, producing a reference prediction near 66.53 W in the smoke test. This validates integration but is less representative than the seeded offline background.

Live Kernel SHAP takes roughly 2-3 minutes per 30-cell snapshot and disables normal batch acceleration. It should therefore remain disabled during KPI experiments:

```yaml
features:
  use_shap: false
  shap_batch_size: 30
```

For future scenario-specific SHAP, matched AI and deterministic Simu5G observations should be explained offline against one common background. Different scenario-specific backgrounds must not be used because their SHAP values would not be directly comparable.

### SHAP interpretation cautions

- SHAP explains the Keras predictor, not the LLM, rule engine, or whole multi-agent system.
- RU type and operating mode are categorical codes and have no ordinal high/low meaning.
- Lag, smoothed-load, and spline features are correlated transformations of traffic load and should be discussed as a load-feature family.
- A SHAP contribution is a model attribution relative to a background, not proof of physical causality.

## Experiment Collection and Analysis

`experiment_metrics_collector.py` records scenario-labelled telemetry as cell CSV and raw JSONL files. Each row includes scenario, run ID, replicate, seed, timestamps, cell identity, KPI source, energy source, radio metrics, energy fields, and ES mode.

`analyze_experiments.py` generates:

- `kpi_comparison.png`;
- `hourly_profiles.png`;
- `es_modes_by_traffic.png`;
- `energy_throughput_tradeoff.png`;
- `run_summary.csv`;
- `scenario_summary.csv`;
- `paired_comparison.csv`;
- `formula_metadata.json`.

The final pilot comparison uses only common `(physical_sim_time_s, BS)` observations. The unmatched first AI snapshot was removed so one scenario could not gain or lose represented energy merely because its collector started earlier.

## KPI Definitions

### Represented energy

```text
represented energy (kWh)
  = sum(represented_energy_j) / 3,600,000
```

### Average network power

```text
average network power (W)
  = total represented energy (J) / represented duration (s)
```

### Network throughput

```text
network throughput (Mbit/s)
  = represented received bits / represented duration / 1,000,000
```

### Data energy efficiency

```text
energy efficiency (Mbit/J)
  = represented received data volume (Mbit)
    / represented energy consumption (J)
```

This follows the 3GPP TS 28.554 form of data volume divided by energy consumption over the same period.

### Aggregate packet loss

```text
packet loss (%)
  = 100 * max(0, offered bits - received bits) / offered bits
```

### Traffic-weighted latency

```text
traffic-weighted latency (ms)
  = sum(cell latency * represented received bits)
    / sum(represented received bits)
```

### Time-weighted PRB utilization

```text
PRB utilization (%)
  = 100 * sum(used PRBs * represented interval)
    / sum(available PRBs * represented interval)
```

This is a time-weighted aggregation of Simu5G counters aligned with the used-to-available PRB relationship described by 3GPP TS 28.552.

## Validated Matched Pilot Results

The final matched seed-42 comparison contains:

```text
57 common snapshots per scenario
30 cells per snapshot
1,710 cell observations per scenario
physical simulation time 4-60 seconds
90 associated UEs
0 unassociated UEs
```

| KPI | Deterministic | AI | Observed change |
|---|---:|---:|---:|
| Represented energy | 54.3601 kWh | 50.5842 kWh | **-6.946%** |
| Average network power | 2384.21 W | 2218.60 W | -6.946% |
| Average network throughput | 66.9394 Mbit/s | 66.9356 Mbit/s | **-0.006%** |
| Energy efficiency | 0.028076 Mbit/J | 0.030170 Mbit/J | approximately +7.46% |
| Traffic-weighted latency | 2.933 ms | 3.934 ms | approximately +34.1% |
| Aggregate packet loss | 0.000629% | 0.006290% | +0.005661 percentage points |
| Weighted PRB utilization | 2.458% | 3.332% | approximately +35.5% |
| Average associated UEs | 90 | 90 | unchanged |
| Full sleeping cell-time | 2.749% | 3.099% | +0.350 percentage points |

The validated energy result replaces the earlier projected `12-18%` system-wide saving shown above. The current evidence supports **6.946% energy saving for one matched seed**, not a generalized saving range.

### Energy saving by traffic level

| Traffic condition | AI saving relative to deterministic |
|---|---:|
| Very low | 8.558% |
| Low | 3.444% |
| Medium | 3.191% |
| Peak | 5.562% |

### Applied ES-mode distribution

| Scenario | Active | Light sleep | Full sleep |
|---|---:|---:|---:|
| AI | 70 | 1,587 | 53 |
| Deterministic | 211 | 1,452 | 47 |

The AI controller spent more represented cell-time in light sleep and less time fully active. All low, medium, and peak AI observations used light sleep; active and full-sleep observations occurred during very-low traffic.

## Correct Interpretation of the Result

For seed 42, AI control achieved a favorable energy-throughput trade-off:

```text
6.946% lower represented energy
0.006% lower aggregate throughput
approximately 7.46% higher data energy efficiency
```

The improvement was not free. AI also increased latency, packet loss, and PRB utilization. The likely interpretation is that AI consolidated operation and used active radio resources more intensively. The paper must report these QoS trade-offs rather than claiming that every KPI improved.

## Reproduction Commands

### Python agents

Run from `CODE/` in separate terminals as required by the selected mode:

```bash
python -u OrchestrationAgent.py
python -u feedbackAGENT.py config/agentCONFIG.yaml
python -u OptimizationAgentwzmq.py --mode ai_based --seed 42
python -u newPredictionAgent.py --config config/prediction_agent_config.yaml
```

For deterministic control:

```bash
python -u OptimizationAgentwzmq.py --mode rule_based --seed 42
```

Feedback Agent and LLM plan generation are not required for deterministic decision selection.

### Simu5G command-line run

Run from the OMNeT++ environment:

```bash
cd ~/default_workspace/auto-easoran-sim

./auto-easoran-sim -u Cmdenv \
-n "ned:/home/opp_env/default_workspace/inet-4.5.4/src:/home/opp_env/default_workspace/simu5g-1.4.1/src" \
-c General \
--seed-set=42 \
--sim-time-limit=60s \
omnetpp.ini
```

### Qtenv topology visualization

```bash
./auto-easoran-sim -u Qtenv \
-n "ned:/home/opp_env/default_workspace/inet-4.5.4/src:/home/opp_env/default_workspace/simu5g-1.4.1/src" \
-c General \
--seed-set=42 \
--sim-time-limit=1s \
omnetpp.ini
```

### Matched KPI analysis

```bash
python -u analyze_experiments.py \
  --input-dir ../OUTPUTs/experiments/paired_seed42_final_matched \
  --output-dir ../OUTPUTs/experiments/analysis_seed42_final_matched
```

### Reproducible offline SHAP

```bash
python -u generate_shap_plots.py \
  --data ../archive/merged_training_data.csv \
  --output-dir ../OUTPUTs/shap/paper_seed42_final \
  --samples 24 \
  --background-samples 100 \
  --seed 42
```

## Reliability Improvements

The validated implementation includes:

- bounded draining of prediction messages to reduce backlog;
- non-blocking feedback/status polling;
- REQ-socket recreation after orchestration timeout or protocol failure;
- active-UE protection against unsafe deep sleep;
- physical feasibility constraints for candidate ES modes;
- minimum predicted-saving deployment threshold;
- bounded LLM healing-plan count;
- stale-snapshot rejection;
- requested-versus-physical state reporting;
- matched-window analysis to eliminate collector-start bias;
- explicit KPI and energy-source metadata;
- offline SHAP generation to avoid disturbing KPI experiments.

## Current Limitations

- The headline comparison uses one seed and one replicate per scenario.
- Confidence intervals are zero in the single-run summary and must not be interpreted as low variability.
- At least five matched seeds are required before statistical generalization.
- The energy calculation is an analytical model driven by runtime Simu5G state.
- Some requested ES modes do not have a direct mutable Simu5G 1.4.1 PHY equivalent.
- Live Kernel SHAP is synchronous and too expensive for normal closed-loop experiments.
- The current offline SHAP cohort explains the ANN, not the LLM or deterministic rules.
- Scenario-specific AI-versus-deterministic SHAP still requires matched runtime samples and one common background.
- The 30-gNB topology is the validated laptop-scale configuration; larger deployments require a separate scalability study.

## Valid and Invalid Research Claims

### Supported by the current pilot

- A genuine 30-gNB, 90-UE Simu5G topology was integrated with the agents.
- Real runtime telemetry drives Keras prediction and controller decisions.
- Supported power and sleep commands are physically applied to the target Simu5G gNB PHY.
- Deterministic and AI scenarios were compared over one matched logical-day run.
- AI saved 6.946% represented energy with a -0.006% throughput change for seed 42.
- AI improved aggregate data energy efficiency while increasing latency, loss, and PRB utilization.
- Both offline and live ANN SHAP generation were validated.

### Not yet supported

- The result is statistically significant or generalizes to all seeds.
- AI improves every KPI.
- The observed saving applies to real commercial networks without calibration.
- A candidate mode skipped by the benefit guard is universally ineffective.
- SHAP explains the LLM, policy engine, or complete multi-agent system.
- Earlier projected performance values above are validated measurements of the current implementation.

## Paper-Ready Pilot Summary

> A 30-gNB, 90-UE Simu5G testbed was integrated with a multi-agent RAN energy-control pipeline and evaluated over a compressed 24-hour traffic cycle. In the matched seed-42 pilot, AI-based control reduced represented energy from 54.36 kWh to 50.58 kWh, corresponding to a 6.95% saving, while aggregate throughput changed by only -0.006%. Data energy efficiency improved by approximately 7.46%. The energy improvement was accompanied by higher traffic-weighted latency, packet loss, and PRB utilization, demonstrating a measurable energy-QoS trade-off. Kernel SHAP analysis identified radio-unit class, traffic-load history, antenna configuration, and transmit power as influential ANN energy-prediction features. These results establish end-to-end feasibility but remain single-seed evidence pending multi-seed statistical validation.

## Updated Documentation Date

This implementation update was added on **28 June 2026**. Detailed result notes and formulas are also maintained in `RESULTS_README.md`.
