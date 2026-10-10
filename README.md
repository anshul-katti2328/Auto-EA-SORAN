Auto EA-SORAN
Autonomous Energy-Aware Self-Optimizing RAN
ITU AI for Good Build-a-Thon 5.0
Team CutTheCord
Rajat Duggal (Mentor) – rajat.duggal@nokia.com
Saanvi Chauhan (Team Lead) – saanvi.mitblr2023@learner.manipal.edu
Nysa Sahu – nysa.mitblr2024@learner.manipal.edu
Anshul Katti – anshul1.mitblr2024@learner.manipal.edu

Digital-twin feasibility study for the hackathon — the current implementation is evaluated in Simu5G and does not claim production-network deployment.
________________________________________
Executive Summary
The problem. The rapid deployment of 5G networks has increased the energy footprint of the Radio Access Network (RAN), driven by network densification, massive-MIMO deployments, increasing data traffic and the continuous operation of base-station infrastructure. Although 5G New Radio provides improved energy efficiency per transmitted bit compared with earlier generations, overall energy consumption can continue to increase as capacity and traffic demand grow. Always-on operation of base stations can leave network resources active during periods of low traffic, while aggressive energy-saving actions during high-load periods can degrade throughput or latency.
The solution. Auto EA-SORAN (Autonomous Energy-Aware Self-Optimizing RAN) is an AI-native closed-loop architecture for autonomous energy optimization of 5G base stations using Simu5G as a high-fidelity digital twin. The architecture consists of four cooperative agents: a Prediction Agent that forecasts traffic and energy behaviour using a Keras ANN-based regression model, an Optimization Agent that formulates energy-saving actions such as transmit-power adjustment and micro/deep-sleep transitions, a Feedback Agent that validates proposed actions against predefined QoS and safety guard-rails, and an Orchestration Agent that executes validated actions within the simulated RAN. The system continuously operates through a prediction → optimization → feedback → actuation cycle.
What makes it distinct. Auto EA-SORAN combines AI-based energy prediction, multi-agent coordination, policy-controlled actuation, explainable safety validation and high-fidelity 5G simulation within one closed-loop system. Unlike approaches that optimize isolated RAN parameters or rely only on static thresholds, the proposed architecture jointly considers energy consumption and QoS and can reject or roll back an action when predefined performance guard-rails are violated. The architecture is mapped to relevant 3GPP, ITU and O-RAN technical frameworks; these mappings represent technical alignment and architectural correspondence, not formal certification or compliance claims. [1][2][3][4][5][6][7][9]
Knowledge base. The knowledge base comprises the BSinfo.csv, CLdata.csv and ECdata.csv challenge datasets, live Simu5G telemetry, the Keras ANN tabular-regression model, temporal and lag-based feature engineering, SHAP-based explainability, Simu5G/INET simulation components, and relevant 5G energy-efficiency, energy-saving and QoS technical documents. [15][16][17]
________________________________________
Team
•	Saanvi — Team Leader — Manipal Institute of Technology, Bengaluru
•	Anshul — Team Member — Manipal Institute of Technology, Bengaluru
•	Nysa — Team Member — Manipal Institute of Technology, Bengaluru
Mentors / Internal Guides.
•	Rajat Duggal
•	Ashutosh Anand
Contact.
•	Anshul — anshul1.mitblr2024@learner.manipal.edu
•	Nysa — nysa.mitblr2024@learner.manipal.edu
•	Saanvi — saanvi.mitblr2023@learner.manipal.edu
________________________________________
Submission
•	Submission title — Autonomous Energy-Aware Self-Optimizing RAN (Auto EA-SORAN)
•	Institution — Manipal Institute of Technology, Bengaluru
•	Repository — ⟨add link if applicable⟩
•	Demo video — ⟨add link if applicable⟩
________________________________________
Problem, and Why Existing Tools Do Not Solve It
The RAN energy problem
The rapid deployment of 5G networks has increased the energy footprint of the RAN. Network densification, massive-MIMO deployments, increasing traffic and continuously active base-station infrastructure create periods in which network resources are not matched to instantaneous demand.
5G also supports heterogeneous services such as eMBB, URLLC and mMTC, each with different throughput, latency and reliability requirements. Consequently, aggressive energy-saving actions can adversely affect QoS, while conservative policies can miss potential energy-saving opportunities.
What existing approaches do, and what remains open
Conventional network-management systems rely heavily on predefined rules, static configurations and threshold-based control. Common energy-saving mechanisms include cell/base-station sleep modes, transmit-power adaptation and carrier on/off control.
Academic research has extended these approaches using traffic/load prediction, machine learning, heuristic optimization, reinforcement learning for base-station sleep control, multi-agent systems and federated learning. Commercial telecom platforms have also introduced AI-based energy-management capabilities.
However, several gaps remain:
•	many solutions optimize individual control parameters rather than coordinating multiple energy-control actions;
•	static or network-wide QoS-energy thresholds may be insufficient for heterogeneous 5G services;
•	simplified simulation environments may not capture complete protocol-stack interactions;
•	black-box AI decisions can create explainability and operator-trust challenges;
•	a prediction model alone does not provide a complete safe actuation loop.
Auto EA-SORAN connects prediction to optimization, safety validation and runtime actuation rather than treating energy prediction as the end of the problem.
________________________________________
Architecture — ITU-T Y.3172
                 ┌───────────────────────┐
                 │      Knowledge Base   │
                 │ datasets + standards  │
                 └───────────┬───────────┘
                             │
Network telemetry ──→ Prediction ──→ Optimization
                           Agent          Agent
                                              │
                                              ▼
                                      Feedback Agent
                                      QoS / safety
                                        validation
                                              │
                                              ▼
                                      Orchestration
                                          Agent
                                              │
                                              ▼
                                        Simu5G RAN
                                              │
                                              └────→ updated telemetry
The architecture follows the ML pipeline and management/orchestration concepts described in ITU-T Y.3172 and the AI-based network-automation context of Y.3177. [1][2]
Component	Role	Auto EA-SORAN implementation
Data / telemetry source	Network and configuration inputs	BSinfo.csv, CLdata.csv, ECdata.csv and Simu5G runtime telemetry
Prediction Agent	Estimate future network/energy behaviour	Keras ANN-based tabular regression with temporal and lag-based features
Optimization Agent	Generate candidate energy-saving actions	Transmit-power adjustment and micro/deep-sleep state selection
Feedback Agent	Validate actions against safety/QoS constraints	Throughput and latency guard-rails, with rejection/rollback
Orchestration Agent	Execute validated actions	Applies the selected action in the Simu5G environment
Digital twin	Controlled execution environment	Simu5G/INET 5G network simulation
Explainability	Explain model behaviour	SHAP-based feature attribution
The closed-loop controller
The system operates as:
Network telemetry → Prediction → Optimization → Feedback/Safety validation → Orchestration → Updated network state.
The separation between prediction, decision-making, validation and actuation is central to the architecture. The Feedback Agent provides a safety boundary between AI-generated decisions and network control.
________________________________________
Knowledge Base and Challenge Datasets
Dataset coverage
Dataset	Content	Role in Auto EA-SORAN
BSinfo.csv	Static base-station/cell configuration: BS, CellName, RUType, operating mode, frequency, bandwidth, number of antennas and TX power	Describes the radio configuration associated with each cell/base station
CLdata.csv	Hourly cell-level operating data: time, BS, CellName, normalized traffic load and FSMode1–FSMode6	Represents changing traffic conditions and activation of energy-saving mechanisms
ECdata.csv	Hourly base-station energy-consumption values: time, BS and Energy	Provides the energy-consumption target used by the prediction pipeline
The challenge data therefore combines static configuration, dynamic operating conditions and energy-consumption observations. [17]
Dataset-unit note. The published challenge dataset supplies an Energy value; the current report does not assign a physical unit such as kWh unless that unit is explicitly defined by the dataset documentation.
Additional knowledge sources
The evaluation also uses:
•	Simu5G/INET runtime telemetry;
•	3GPP energy-efficiency, intent-driven management, SON and network-resource specifications;
•	ITU-T ML and AI-native networking frameworks;
•	O-RAN technical material;
•	TRAI AI/Big Data recommendations;
•	responsible-AI and Indian data-governance documents;
•	Simu5G technical literature. [1][2][3][4][5][6][7][9][10][11][12][13][14][15][16]
________________________________________
Uniqueness of the Proposed Solution
The uniqueness of Auto EA-SORAN lies in integrating AI-based prediction, multi-agent autonomous decision-making, policy-guarded control and high-fidelity 5G network simulation into a single closed-loop energy-optimization framework.
The Prediction Agent uses a Keras ANN-based model trained on historical base-station configuration, cell-level operating and energy-consumption data. The Optimization Agent converts predicted network conditions into structured energy-saving actions. The Feedback Agent validates every proposed action against throughput and latency guard-rails. The Orchestration Agent then applies validated actions directly within the simulated RAN.
A second distinguishing aspect is the use of Simu5G as a controllable digital twin, coupled with a custom state-based gNB energy model containing active, micro-sleep and deep-sleep states. Active-state power consumption is dynamically linked to runtime PRB utilization. The agent layer communicates with the simulator through low-latency asynchronous IPC.
The framework additionally incorporates SHAP-based explainability, policy-based validation, automatic rollback when post-actuation QoS guard-rails are breached, and operator-intent constraints. The proposed uniqueness is therefore the end-to-end integration of prediction, optimization, explainability, safety validation and runtime orchestration, rather than a claim of a new standalone ML algorithm. [15][16]
________________________________________
Mapping to Standards
The proposed architecture is mapped to standards and technical frameworks relevant to ML-enabled 5G networks, AI-based network automation, energy efficiency, intent-driven management and self-organizing networks. These mappings identify architectural correspondence and implementation gaps between the current proof-of-concept and standardized deployment. [1][2][3][4][5][6][7][9]
Standard / framework	Relevant concept	Auto EA-SORAN correspondence	Current gap
ITU-T Y.3172	ML architecture for future networks	Prediction pipeline, data/model separation and management context	Current implementation remains a Simu5G proof-of-concept
ITU-T Y.3177	AI-based network automation for resource/fault management	Closed-loop prediction, optimization, validation and orchestration	Standardized production automation interfaces are not yet implemented
3GPP TS 28.310	Energy efficiency of 5G	Energy-aware RAN control and energy-saving actions	Full 3GPP management integration remains future work
3GPP TS 28.312	Intent-driven management	Operator objectives/constraints influence optimization	Intent interface is represented at architecture level
3GPP TS 28.313	Self-Organizing Networks	Autonomous optimization and feedback	Current implementation is simulation-based
3GPP TS 28.541	5G Network Resource Model	RAN/resource concepts used in modelling	Direct standardized NRM integration is future work
ITU FG-AINN	AI-native telecom architecture	AI agents and closed-loop network control	Full standards realization is not claimed
O-RAN	Near-RT RIC / E2-based control	Proposed deployment direction for orchestration	Current implementation does not claim live RIC/E2 integration
________________________________________
Policy Gaps and ITU AI Readiness Mapping
Policy and governance gaps
The project identifies deployment requirements and gaps rather than claiming that the prototype is legally compliant.
#	Area	Observation / implementation gap
1	AI governance	Production deployment would require explicit model governance, auditability and lifecycle management
2	Human oversight	Energy-saving actions that can affect QoS require defined operator controls and safety guard-rails
3	Privacy	The current feasibility study uses network/base-station and simulated telemetry rather than subscriber-identifying information; future personal-data use would require applicable data-protection controls
4	Accountability	Production deployment would require clear responsibility for model decisions, policy validation and actuation
5	Security	Network-control interfaces would require security assessment before field deployment
6	Standards integration	Production deployment would require standardized interfaces and interoperability with RAN management platforms
Relevant Indian policy and regulatory material includes TRAI's 2023 recommendations on AI and Big Data in telecom, NITI Aayog's Responsible AI work, the Digital Personal Data Protection Act 2023, the Telecommunications Act 2023 and the India AI Governance Guidelines 2025. [10][11][12][13][14]
AI Readiness dimensions
Dimension	Why it matters	Evidence / current status
Data / model readiness	Reliable prediction requires structured historical and runtime data	BSinfo, CLdata, ECdata plus Simu5G telemetry and a trained ANN
AI integration	AI must participate in a complete operational loop	Prediction and optimization are connected to feedback and orchestration
Automation	The target is movement from static control toward autonomous optimization	Observe → predict → optimize → validate → actuate → observe
Human collaboration / oversight	AI-generated actions must remain bounded by explicit constraints	Feedback Agent, QoS guard-rails and rollback
Explainability	Operators need visibility into model behaviour	SHAP-based feature attribution
Policy / governance	Deployment requires auditability, security and accountability	Policy/safety layer is implemented conceptually; production governance remains a deployment requirement
________________________________________
Evaluation
Feasibility study / digital-twin evaluation
The current implementation is evaluated as a 5G RAN energy-optimization feasibility study using a high-fidelity Simu5G digital twin rather than as a live commercial-network deployment.
Historical network and energy data are used to train the Prediction Agent, while the digital twin generates dynamic network conditions and runtime telemetry.
The complete autonomous loop is evaluated:
Network telemetry → Prediction → Optimization → Feedback/Safety validation → Orchestration → Updated network state.
Multiple traffic conditions are considered to examine whether the system can identify energy-saving opportunities while maintaining required QoS. The evaluation therefore considers the interaction between traffic/load, PRB utilization, energy consumption, throughput and latency, rather than treating energy consumption independently.
Problem scenario
The primary scenario is a dynamic 5G traffic environment in which base-station utilization varies over time.
During high-load periods, aggressive energy-saving actions may cause throughput degradation or increased latency. During low-load periods, keeping all radio resources continuously active can result in avoidable energy consumption.
The control problem is:
How can a 5G RAN autonomously reduce energy consumption during periods of low or changing network load while maintaining predefined QoS requirements?
The system evaluates whether AI-based prediction can anticipate network conditions sufficiently early for the Optimization Agent to select an appropriate energy-saving action, and whether the Feedback Agent can prevent actions that violate throughput or latency constraints.
Evaluation dimensions
Models. The Prediction Agent uses a Keras ANN-based tabular-regression model trained using historical base-station, cell-level operating and energy-consumption data. Temporal and lag-based features represent changing network conditions. SHAP-based feature attribution is used for interpretability. [15][16]
Learning. The ML pipeline learns the relationship between network/base-station characteristics and energy consumption from historical data. Runtime inference uses current Simu5G telemetry to generate predictions for the present network state.
Agentic architecture. Four cooperating agents provide separation between prediction, decision-making, safety validation and actuation.
Sandbox. Simu5G functions as a controlled digital-twin environment in which energy-saving policies can be tested without affecting a production network. [15][16]
Automation. The intended sequence is observe → predict → optimize → validate → actuate → observe again.
Privacy. The current feasibility study primarily uses base-station, cell-level, energy and simulated network telemetry rather than subscriber-identifying information. Future deployment involving personal data would require applicable data-classification, access-control, retention and privacy-preserving controls.
Responsible AI. The Feedback Agent prevents proposed actions from being executed when predefined QoS constraints are violated. Post-actuation monitoring and rollback provide an additional safeguard. SHAP analysis provides interpretability into the contribution of input features.
Reliability. Reliability is evaluated by observing whether the autonomous loop maintains throughput and latency within predefined guard-rails while performing energy-saving actions.
Energy conservation. Energy conservation is the primary optimization objective. The evaluation compares operation under energy-saving actions against a baseline without autonomous optimization while simultaneously observing QoS metrics.
Openness and explainability. The model pipeline is separated into preprocessing, trained-model and inference components. SHAP analysis identifies influential input features. Future deployment would require model cards, standardized metadata, versioning and interoperability interfaces.
Collaboration with AI. Operator-level objectives and constraints can influence AI-based optimization. The Feedback Agent acts as a policy/safety boundary between AI-generated decisions and network actuation.
Trust and adoption. Predictable behaviour, explainability, QoS protection and recovery from incorrect decisions are addressed through explicit guard-rails, feedback monitoring, rollback mechanisms and explainability.
________________________________________
Deployment of the Solution
Following successful digital-twin evaluation, the proposed deployment pathway is a controlled progression from simulation to laboratory/field pilot and finally to production RAN integration.
Phase 1 — Digital twin. Validate the Prediction, Optimization, Feedback and Orchestration agents under reproducible traffic and network conditions in Simu5G. [15][16]
Phase 2 — Controlled RAN testbed. Deploy the prediction and optimization services against a controlled 5G/O-RAN test environment. Introduce real network telemetry while maintaining policy guard-rails and rollback mechanisms.
Phase 3 — Near-RT RIC / O-RAN integration. Map optimization and orchestration functions toward a near-real-time RIC architecture, with standardized interfaces such as E2 providing controlled interaction with the RAN. The current implementation does not claim this integration; it is the next deployment stage. [9]
Phase 4 — Pilot deployment. Conduct a limited pilot in a controlled network region or selected base-station cluster. Compare the AI-enabled system against the existing energy-management baseline using energy consumption, throughput, latency, availability and reliability KPIs.
Phase 5 — Production deployment. Subject to successful pilot validation, security assessment, standards/interface integration and policy approval, extend the solution to larger RAN domains with continuous monitoring, model governance, drift detection, audit logging and controlled rollback.
The deployment objective is therefore:
validated AI-native digital-twin feasibility study → controlled 5G/O-RAN testbed → limited pilot → production-scale autonomous energy optimization, while preserving QoS, operator control and safety throughout the transition.
________________________________________
References
[1] ITU-T Recommendation Y.3172 (06/2019), “Architectural framework for machine learning in future networks including IMT-2020.”
[2] ITU-T Recommendation Y.3177 (02/2021), “Architectural framework for artificial intelligence-based network automation for resource and fault management in future networks including IMT-2020.”
[3] 3GPP TS 28.310, “Management and orchestration; Energy efficiency of 5G.”
[4] 3GPP TS 28.312, “Management and orchestration; Intent driven management services for mobile networks.”
[5] 3GPP TS 28.313, “Management and orchestration; Self-Organizing Networks (SON) for 5G networks.”
[6] 3GPP TS 28.541, “Management and orchestration; 5G Network Resource Model (NRM); Stage 2 and Stage 3.”
[7] ITU-T Focus Group on AI Native for Telecommunication Networks (FG-AINN), relevant architecture and standardization-gap material.
[8] ITU, “AI Readiness Framework,” including dimensions covering data/model readiness, AI integration, collaboration, policy and digital infrastructure.
[9] O-RAN Alliance, near-RT RIC and E2-based RAN control architecture documentation.
[10] Telecom Regulatory Authority of India (TRAI), “Recommendations on Leveraging Artificial Intelligence and Big Data in Telecommunication Sector,” 20 July 2023.
[11] NITI Aayog, “Towards Responsible AI – Enforcement of Principles,” 2021.
[12] Government of India, “The Digital Personal Data Protection Act, 2023,” India Code, Act No. 22 of 2023.
[13] Government of India, “The Telecommunications Act, 2023,” India Code, Act No. 44 of 2023.
[14] Ministry of Electronics and Information Technology (MeitY), “India AI Governance Guidelines,” 5 November 2025.
[15] G. Nardini, G. Stea and A. Virdis, “Scalable Real-Time Emulation of 5G Networks With Simu5G,” IEEE Access, vol. 9, pp. 148504–148520, 2021, doi:10.1109/ACCESS.2021.3123873.
[16] C. Boeira et al., “A Calibrated and Automated Simulator for Innovations in 5G,” arXiv:2404.10643, 2024.
[17] Build-a-thon challenge training datasets supplied with the project: BSinfo.csv, CLdata.csv and ECdata.csv.
[18] India — AI Regulatory & Policy Landscape, compiled reference list for ITU AI for Good Sandbox — India, 26 September 2026.

