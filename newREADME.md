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

🏗 System Architecture
Component Stack

text
┌─────────────────────────────────────────────────────────────┐
│                   Simu5G Simulation (OMNeT++)               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  TwinMetricsAgent (provides real-time network metrics) │ │
│  │  OrchestrationShim (applies configuration changes)     │ │
│  └────────────────────────────────────────────────────────┘ │
└────┬──────────────────────────────────────┬─────────────────┘
     │ ZMQ Pub/Sub                          │ ZMQ REQ/REP
     │ Metrics Feed                         │ Commands
     ↓                                      ↑
┌─────────────────────────────────────────────────────────────┐
│               Distributed Agent System (Python)              │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ Prediction Agent │  │ Optimization Agnt│               │
│  │ (Keras/TF)       │  │ (DEAP + FAISS)   │               │
│  │ Port: 5555       │  │ Port: 5558       │               │
│  └────────┬─────────┘  └────────┬─────────┘               │
│           │                      │                         │
│  ┌────────▼──────────────────────▼─────────┐             │
│  │     Policy Manager (3GPP + Nokia)       │             │
│  │     FAISS-based Policy Similarity       │             │
│  │     Constraint Extraction & Validation  │             │
│  └────────┬─────────────────────┬──────────┘             │
│           │                     │                         │
│  ┌────────▼──────────┐  ┌──────▼──────────┐             │
│  │ Feedback Agent    │  │ Orchestration   │             │
│  │ (Validation)      │  │ Agent (Exec)    │             │
│  │ Port: 5560        │  │ Port: 5562      │             │
│  └───────────────────┘  └────────┬────────┘             │
│                                  │                       │
│  Communication Layer (ZMQ):       │                       │
│  - REQ/REP: Sync request-reply   │                       │
│  - PUB/SUB: Async metrics stream │                       │
│  - PUSH/PULL: Task queue         │                       │
└──────────────────────────────────┼───────────────────────┘
                                   │ ZMQ Cmd
                                   ↓
                          ┌─────────────────┐
                          │ OrchestrationShim│
                          │ (OMNeT++)        │
                          └─────────────────┘

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
