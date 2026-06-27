#!/usr/bin/env python3
"""
Auto EA-SORAN Feedback Agent
Main agent implementation for validating energy conservation plans
and monitoring network performance KPIs with comprehensive policy enforcement.

Enhanced with Nokia, 3GPP, and hybrid policies for 5G network optimization.
"""

import asyncio
import logging
import time
import zmq
import zmq.asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
from datetime import datetime, timedelta

# Import comprehensive policy system
from policies import (
    POLICY_MANAGER, 
    validate_energy_plan_policies,
    get_critical_policy_violations,
    get_policy_summary,
    PolicySeverity
)

#from .feedbackCOMM import CommunicationHandler, MessageType
#from .validation.validator import ValidationEngine
#from .monitoring.kpi_monitor import KPIMonitor
#from .rollback.rollback_engine import RollbackEngine
#from .utils.logger import get_logger
#from .utils.exceptions import (
#    ValidationError, 
#    RollbackError, 
#    ConfigurationError,
#    CommunicationError
#)
from feedbackCOMM import CommunicationHandler
from optimizationCOMM import OptimizationMessageType as MessageType, OptimizationMessage as Message
import logging

class ValidationEngine:
    """Analytical validation using live KPIs and proposed settings."""

    def __init__(self, config):
        self.config = config

        self.max_tx_power = float(
            config.get("max_tx_power_dbm", 43.0)
        )
        self.min_tx_power = float(
            config.get("min_tx_power_dbm", 5.0)
        )
        self.max_latency = float(
            config.get("max_latency_ms", 20.0)
        )
        self.max_packet_loss = float(
            config.get("max_packet_loss_pct", 1.0)
        )
        self.min_throughput = float(
            config.get("min_throughput_mbps", 1.0)
        )
        self.min_throughput_retention = min(
            1.0,
            max(
                0.0,
                float(
                    config.get(
                        "min_throughput_retention", 0.80
                    )
                ),
            ),
        )
        self.min_confidence = float(
            config.get("min_confidence", 0.50)
        )

    async def predict_impact(
        self,
        proposed_config,
        current_kpis,
    ):
        """Estimate the KPI impact of a proposed configuration."""

        current_tx_power = float(current_kpis.tx_power)
        proposed_tx_power = float(
            proposed_config.get(
                "tx_power_dbm",
                proposed_config.get(
                    "tx_power", current_tx_power
                ),
            )
        )

        es_mode = str(
            proposed_config.get("es_mode", "active")
        ).lower()

        # Approximate savings aligned with TwinMetricsAgent.
        es_saving_pct = {
            "active": 0.0,
            "normal": 0.0,
            "sleep": 65.0,
            "light_sleep": 40.0,
            "carrier_off": 25.0,
            "mimo_reduce": 10.0,
            "micro_dtx": 15.0,
        }.get(es_mode, 0.0)

        latency_penalty_ms = {
            "active": 0.0,
            "normal": 0.0,
            "sleep": 15.0,
            "light_sleep": 5.0,
            "carrier_off": 1.0,
            "mimo_reduce": 1.5,
            "micro_dtx": 2.0,
        }.get(es_mode, 0.0)

        # Conservative service-retention estimates for modes that
        # keep the radio available. Deep sleep remains disruptive.
        throughput_factor = {
            "active": 1.00,
            "normal": 1.00,
            "sleep": 0.15,
            "deep_sleep": 0.15,
            "light_sleep": 0.90,
            "rf_chain_sleep": 0.90,
            "carrier_off": 0.80,
            "mimo_reduce": 0.90,
            "micro_dtx": 0.95,
            "bwp_adaptation": 0.90,
        }.get(es_mode, 1.00)

        tx_reduction_db = max(
            0.0,
            current_tx_power - proposed_tx_power,
        )

        # TX-power reduction has a smaller supply-power effect
        # than a complete ES-mode transition.
        tx_saving_pct = min(
            20.0,
            tx_reduction_db * 1.5,
        )

        # Combine independent percentage reductions without
        # simply adding percentages beyond 100%.
        combined_energy_saving = 100.0 * (
            1.0
            - (1.0 - es_saving_pct / 100.0)
            * (1.0 - tx_saving_pct / 100.0)
        )

        current_throughput = max(
            0.0,
            float(current_kpis.avg_throughput),
        )

        predicted_throughput = (
            current_throughput
            * throughput_factor
        )

        throughput_change = (
            predicted_throughput - current_throughput
        )

        # Reducing TX power can modestly increase loss.
        packet_loss_change = tx_reduction_db * 0.01

        return {
            "energy_saving": combined_energy_saving,
            "latency_change": latency_penalty_ms,
            "throughput_change": throughput_change,
            "packet_loss_change": packet_loss_change,
            "power_reduction": tx_reduction_db,
            "prb_utilization_change": 0.0,
            "predicted_tx_power": proposed_tx_power,
            "selected_es_mode": es_mode,
        }

    async def validate_plan(
        self,
        plan,
        current_kpis,
        predicted_impact,
    ):
        """Run deterministic safety checks before policy checks."""

        proposed_tx_power = float(
            plan.proposed_config.get(
                "tx_power_dbm",
                plan.proposed_config.get(
                    "tx_power",
                    current_kpis.tx_power,
                ),
            )
        )

        proposed_es_mode = str(
            plan.proposed_config.get(
                "es_mode", "active"
            )
        ).lower()

        active_users = max(
            0,
            int(current_kpis.active_users),
        )

        current_throughput = max(
            0.0,
            float(current_kpis.avg_throughput),
        )

        throughput_floor = (
            current_throughput
            * self.min_throughput_retention
        )

        if current_throughput >= self.min_throughput:
            throughput_floor = max(
                self.min_throughput,
                throughput_floor,
            )

        predicted_latency = (
            float(current_kpis.avg_latency)
            + float(
                predicted_impact.get(
                    "latency_change", 0.0
                )
            )
        )

        predicted_throughput = (
            current_throughput
            + float(
                predicted_impact.get(
                    "throughput_change", 0.0
                )
            )
        )

        predicted_packet_loss = (
            float(current_kpis.packet_loss_rate)
            + float(
                predicted_impact.get(
                    "packet_loss_change", 0.0
                )
            )
        )

        predicted_energy_saving = float(
            predicted_impact.get(
                "energy_saving", 0.0
            )
        )

        return [
            {
                "name": "tx_power_upper_bound",
                "passed": (
                    proposed_tx_power
                    <= self.max_tx_power
                ),
                "critical": True,
                "severity": "critical",
                "message": (
                    f"TX power {proposed_tx_power:.2f} dBm; "
                    f"maximum is {self.max_tx_power:.2f} dBm"
                ),
            },
            {
                "name": "tx_power_lower_bound",
                "passed": (
                    proposed_tx_power
                    >= self.min_tx_power
                ),
                "critical": True,
                "severity": "critical",
                "message": (
                    f"TX power {proposed_tx_power:.2f} dBm; "
                    f"minimum is {self.min_tx_power:.2f} dBm"
                ),
            },

            {
                "name": "latency_limit",
                "passed": (
                    predicted_latency
                    <= self.max_latency
                ),
                "critical": True,
                "severity": "critical",
                "message": (
                    f"Predicted latency "
                    f"{predicted_latency:.2f} ms; maximum is "
                    f"{self.max_latency:.2f} ms"
                ),
            },
            {
                "name": "packet_loss_limit",
                "passed": (
                    predicted_packet_loss
                    <= self.max_packet_loss
                ),
                "critical": True,
                "severity": "critical",
                "message": (
                    f"Predicted packet loss "
                    f"{predicted_packet_loss:.4f}%; "
                    f"maximum is "
                    f"{self.max_packet_loss:.4f}%"
                ),
            },
            {
                "name": "throughput_retention",
                "passed": (
                    predicted_throughput
                    >= throughput_floor
                ),
                "critical": True,
                "severity": "critical",
                "message": (
                    f"Predicted throughput "
                    f"{predicted_throughput:.3f} Mbps; "
                    f"demand-aware floor is "
                    f"{throughput_floor:.3f} Mbps "
                    f"({self.min_throughput_retention:.0%} "
                    f"of the current baseline)"
                ),
            },
            {
                "name": "active_user_sleep_protection",
                "passed": not (
                    proposed_es_mode in {
                        "sleep",
                        "deep_sleep",
                    }
                    and active_users > 0
                ),
                "critical": True,
                "severity": "critical",
                "message": (
                    f"Requested ES mode is "
                    f"{proposed_es_mode}; active users: "
                    f"{active_users}"
                ),
            },
            {
                "name": "positive_energy_saving",
                "passed": predicted_energy_saving >= 0.0,
                "critical": False,
                "severity": "medium",
                "message": (
                    f"Predicted energy saving is "
                    f"{predicted_energy_saving:.2f}%"
                ),
            },
            {
                "name": "prediction_confidence",
                "passed": (
                    float(plan.confidence_score)
                    >= self.min_confidence
                ),
                "critical": False,
                "severity": "medium",
                "message": (
                    f"Plan confidence is "
                    f"{float(plan.confidence_score):.3f}; "
                    f"minimum preferred confidence is "
                    f"{self.min_confidence:.3f}"
                ),
            },
        ]

class KPIMonitor:
    """Maintain the latest live KPI snapshot for every BS."""

    def __init__(self, config):
        self.config = config
        self.metrics_endpoint = config.get(
            "metrics_endpoint",
            "tcp://127.0.0.1:5556",
        )
        self.snapshot_timeout = float(
            config.get("snapshot_timeout", 5.0)
        )

        self.context = zmq.asyncio.Context.instance()
        self.socket = None
        self.listener_task = None
        self.latest_snapshots = {}
        self.running = False

    async def start(self):
        if self.running:
            return

        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.connect(self.metrics_endpoint)

        self.running = True
        self.listener_task = asyncio.create_task(
            self._listen_for_metrics()
        )

    async def stop(self):
        self.running = False

        if self.listener_task is not None:
            self.listener_task.cancel()

            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass

            self.listener_task = None

        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None

    async def _listen_for_metrics(self):
        while self.running:
            try:
                frames = await self.socket.recv_multipart()

                if not frames:
                    continue

                payload = json.loads(
                    frames[-1].decode("utf-8")
                )

                sim_time_s = float(
                    payload.get("sim_time_s", 0.0)
                )

                for cell in payload.get("cells", []):
                    self._store_cell_snapshot(
                        cell,
                        sim_time_s,
                    )

            except asyncio.CancelledError:
                raise

            except Exception as error:
                logging.getLogger(
                    "KPIMonitor"
                ).error(
                    "Failed to process metrics: %s",
                    error,
                    exc_info=True,
                )
                await asyncio.sleep(0.1)

    def _store_cell_snapshot(
        self,
        cell: Dict[str, Any],
        sim_time_s: float,
    ):
        bs_id = str(cell.get("bs_id", "unknown"))

        power_w = max(
            0.0,
            float(cell.get("energy_consumption_w", 0.0)),
        )
        dl_mbps = max(
            0.0,
            float(cell.get("dl_mbps", 0.0)),
        )
        ul_mbps = max(
            0.0,
            float(cell.get("ul_mbps", 0.0)),
        )

        raw_es_modes = cell.get("es_modes", [0] * 6)
        if not isinstance(raw_es_modes, (list, tuple)):
            raw_es_modes = []

        es_modes = []
        for value in raw_es_modes[:6]:
            try:
                es_modes.append(
                    1 if int(value) != 0 else 0
                )
            except (TypeError, ValueError):
                es_modes.append(0)

        es_modes.extend([0] * (6 - len(es_modes)))

        efficiency_mbit_per_j = max(
            0.0,
            float(
                cell.get(
                    "energy_efficiency_mbit_per_j",
                    0.0,
                )
            ),
        )

        if efficiency_mbit_per_j > 0.0:
            energy_efficiency = (
                efficiency_mbit_per_j * 1_000_000.0
            )
        elif power_w > 0.0:
            energy_efficiency = (
                (dl_mbps + ul_mbps)
                * 1_000_000.0
                / power_w
            )
            efficiency_mbit_per_j = (
                energy_efficiency / 1_000_000.0
            )
        else:
            energy_efficiency = 0.0

        packet_loss_pct = max(
            0.0,
            min(
                100.0,
                float(cell.get("packet_loss_pct", 0.0)),
            ),
        )

        raw_sleeping = cell.get(
            "radio_sleeping",
            False,
        )
        if isinstance(raw_sleeping, bool):
            radio_sleeping = raw_sleeping
        else:
            radio_sleeping = (
                str(raw_sleeping).strip().lower()
                in {"1", "true", "yes"}
            )

        snapshot = KPISnapshot(
            bs_id=bs_id,
            timestamp=datetime.now(),
            energy_consumption=power_w,
            avg_throughput=dl_mbps,
            avg_latency=max(
                0.0,
                float(cell.get("avg_latency_ms", 0.0)),
            ),
            packet_loss_rate=packet_loss_pct,
            mac_tb_loss_rate=packet_loss_pct,
            active_users=max(
                0,
                int(cell.get("active_ues", 0)),
            ),
            prb_utilization=max(
                0.0,
                min(
                    100.0,
                    float(cell.get("prb_util_pct", 0.0)),
                ),
            ),
            tx_power=float(
                cell.get("tx_power_dbm", 0.0)
            ),
            energy_efficiency=energy_efficiency,
            sim_time_s=float(sim_time_s),
            measurement_interval_s=max(
                0.0,
                float(
                    cell.get(
                        "measurement_interval_s",
                        0.0,
                    )
                ),
            ),
            ul_throughput=ul_mbps,
            generated_dl_throughput=max(
                0.0,
                float(
                    cell.get(
                        "generated_dl_mbps",
                        0.0,
                    )
                ),
            ),
            available_prbs=max(
                0,
                int(cell.get("available_prbs", 0)),
            ),
            used_prbs=max(
                0.0,
                float(cell.get("used_prbs", 0.0)),
            ),
            interval_energy_j=max(
                0.0,
                float(
                    cell.get("interval_energy_j", 0.0)
                ),
            ),
            represented_interval_s=max(
                0.0,
                float(
                    cell.get(
                        "represented_interval_s",
                        0.0,
                    )
                ),
            ),
            represented_energy_j=max(
                0.0,
                float(
                    cell.get(
                        "represented_energy_j",
                        0.0,
                    )
                ),
            ),            
            energy_efficiency_mbit_per_j=(
                efficiency_mbit_per_j
            ),
            sinr=float(
                cell.get("avg_sinr_db", 0.0)
            ),
            es_mode=str(
                cell.get("es_mode", "active")
            ),
            es_modes=es_modes,
            physical_tx_power=float(
                cell.get(
                    "physical_tx_power_dbm",
                    cell.get("tx_power_dbm", 0.0),
                )
            ),
            radio_sleeping=radio_sleeping,
            kpi_source=str(
                cell.get("kpi_source", "unknown")
            ),
            energy_source=str(
                cell.get(
                    "energy_source",
                    "analytical_load_model",
                )
            ),
            loss_source="offered_vs_received_bytes",
        )

        self.latest_snapshots[bs_id] = snapshot

    async def get_current_kpis(self, bs_id):
        bs_id = str(bs_id)
        deadline = time.monotonic() + self.snapshot_timeout

        while time.monotonic() < deadline:
            snapshot = self.latest_snapshots.get(bs_id)

            if snapshot is not None:
                return snapshot

            await asyncio.sleep(0.05)

        raise ValidationError(
            f"No live KPI snapshot available for BS {bs_id}"
        )

class RollbackEngine:
    def __init__(self, config): pass
    async def execute_rollback(self, bs_id, reason): 
        return {'success': True}

class ValidationError(Exception): pass
class RollbackError(Exception): pass  
class ConfigurationError(Exception): pass
class CommunicationError(Exception): pass

# Lines 75-76 - Replace get_logger function

def get_logger(name: str, config: Dict = None):
    """Setup logger with basic configuration"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Only add handler if none exists
        # Create console handler
        handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

# ADD THESE MISSING CLASSES/FUNCTIONS:
class CommunicationError(Exception):
    """Communication-related errors"""
    pass

class AgentState(Enum):
    """Agent operational states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    VALIDATING = "validating"
    ROLLING_BACK = "rolling_back"
    RECOVERING = "recovering"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class EnergyPlan:
    """Energy conservation plan from Optimization Agent"""
    plan_id: str
    target_bs_id: str
    proposed_config: Dict[str, Any]
    expected_energy_saving: float
    confidence_score: float
    timestamp: datetime
    optimization_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Enhanced result of energy plan validation with policy compliance"""
    plan_id: str
    is_valid: bool
    validation_score: float
    violations: List[Dict[str, Any]]
    recommended_adjustments: Optional[Dict[str, Any]]
    validation_metadata: Dict[str, Any]
    timestamp: datetime
    
    # New policy-specific fields
    policy_compliance_score: float = 0.0
    critical_policy_violations: int = 0
    policy_blocks_optimization: bool = False
    
@dataclass
class KPISnapshot:
    """Current network performance snapshot."""

    bs_id: str
    timestamp: datetime

    energy_consumption: float  # Average electrical power, W
    avg_throughput: float      # DL throughput, Mbps
    avg_latency: float         # ms
    packet_loss_rate: float    # percentage
    mac_tb_loss_rate: float    # percentage
    active_users: int
    prb_utilization: float     # percentage
    tx_power: float            # configured dBm
    energy_efficiency: float   # bits per joule

    sim_time_s: float = 0.0
    measurement_interval_s: float = 0.0

    ul_throughput: float = 0.0
    generated_dl_throughput: float = 0.0

    available_prbs: int = 0
    used_prbs: float = 0.0

    interval_energy_j: float = 0.0
    represented_interval_s: float = 0.0
    represented_energy_j: float = 0.0
    energy_efficiency_mbit_per_j: float = 0.0

    sinr: float = 0.0

    es_mode: str = "active"
    es_modes: List[int] = field(
        default_factory=lambda: [0, 0, 0, 0, 0, 0]
    )

    physical_tx_power: float = 0.0
    radio_sleeping: bool = False

    kpi_source: str = "unknown"
    energy_source: str = "unknown"
    loss_source: str = "offered_vs_received_bytes"
class FeedbackAgent:
    """
    Main Feedback Agent class implementing validation, monitoring,
    and rollback functionality for Auto EA-SORAN system with comprehensive
    Nokia and 3GPP policy enforcement.
    """
    
    def __init__(self, config_path: str):
        """Initialize the Feedback Agent with comprehensive policy system"""
        self.config = self._load_config(config_path)
        self.logger = get_logger(__name__, self.config.get('logging', {}))
        
        # Agent state management
        self.state = AgentState.INITIALIZING
        self.agent_id = self.config['agent']['id']
        self.startup_time = datetime.now()
        
        # Core components
        self.communication = CommunicationHandler(self.config['communication'])
        self.validator = ValidationEngine(self.config['validation'])
        self.kpi_monitor = KPIMonitor(self.config['monitoring'])
        self.rollback_engine = RollbackEngine(self.config['rollback'])
        
        # Initialize comprehensive policy system
        self.policy_manager = POLICY_MANAGER
        policy_summary = get_policy_summary()
        self.logger.info(f"Policy system initialized: {policy_summary}")
        
        # State tracking
        self.current_configs: Dict[str, Dict] = {}  # BS_ID -> current config
        self.pending_validations: Dict[str, EnergyPlan] = {}
        self.validation_history: List[ValidationResult] = []
        self.rollback_history: List[Dict] = []
        
        # Enhanced performance tracking with comprehensive policy metrics
        self.validation_metrics = {
            # Original metrics
            'total_validations': 0,
            'successful_validations': 0,
            'rejected_plans': 0,
            'rollbacks_triggered': 0,
            'avg_validation_time': 0.0,
            
            # Comprehensive policy metrics
            'policy_violations_total': 0,
            'policy_violations_3gpp': 0,
            'policy_violations_nokia': 0,
            'policy_violations_hybrid': 0,
            'policy_violations_energy': 0,
            'critical_violations_blocked': 0,
            'policy_compliant_optimizations': 0,
            
            # Severity breakdown
            'critical_policy_violations': 0,
            'high_policy_violations': 0,
            'medium_policy_violations': 0,
            'low_policy_violations': 0,
            
            # 5QI specific metrics
            'qci_5qi_violations': 0,
            'urllc_violations': 0,
            'embb_violations': 0,
            'mmtc_violations': 0,
            
            # Nokia specific metrics
            'nokia_energy_violations': 0,
            'nokia_thermal_violations': 0,
            'nokia_mimo_violations': 0,
            
            # Performance optimization
            'avg_policy_check_time': 0.0,
            'policy_compliance_rate': 1.0
        }
        
        # BS type mapping for policy applicability
        self.bs_type_mapping = {
            'default': 'nokia_airscale',
            # Can be extended with more specific BS types
        }
        
        # Active service tracking per BS
        self.active_services_per_bs: Dict[str, List[str]] = {}
        
        self.logger.info(f"Enhanced Feedback Agent {self.agent_id} initialized with comprehensive policy enforcement")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load agent configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except:
            config = {}
        # Add default communication configuration if not present
        if 'communication' not in config:
            config['communication'] = {
                'agent_id': 'feedback_agent_001',
                'reply_port': 5560,  # Must match OptimizationAgent's feedback_reply_port
                'pub_port': 5561,
                'endpoints': {
                    'optimization_agent': {
                        'host': 'localhost',
                        'port': 5558,      # OptimizationAgent's REP socket
                        'pub_port': 5559   # OptimizationAgent's PUB socket
                    },
                    'prediction_agent': {
                        'host': 'localhost',
                        'port': 5555,      # PredictionAgent's REP socket
                    },
                    'orchestration_agent': {
                        'host': 'localhost',
                        'port': 5562       # ✅ FIXED: OrchestrationAgent's port
                    }
                }
            }
        
        # Ensure required sections exist
        config.setdefault('validation', {'timeout': 300})
        config.setdefault('monitoring', {'check_interval': 10, 'post_change_window': 300})
        config.setdefault('rollback', {'policies': []})
        config.setdefault('agent', {'id': 'feedback_agent_001', 'heartbeat_interval': 30})
        
        # Add default policy configuration if not present
        if 'policies' not in config:
            config['policies'] = {
                'enforcement_levels': {
                    'CRITICAL': 0.0,  # No tolerance
                    'HIGH': 0.1,      # 10% tolerance
                    'MEDIUM': 0.2,    # 20% tolerance
                    'LOW': 0.5        # 50% tolerance
                    }
            }
        return config
        
    # Lines 297-330 - Replace the start() method

    async def start(self):
        """Start the Enhanced Feedback Agent"""
        self.logger.info("Starting Enhanced Feedback Agent with comprehensive policy enforcement...")
    
        try:
        # Initialize communication with detailed error handling
            self.logger.info("Initializing communication handler...")
            await self.communication.initialize()
            self.logger.info("✅ Communication handler initialized successfully")
        
        # Start KPI monitoring
            self.logger.info("Starting KPI monitoring...")
            await self.kpi_monitor.start()
            self.logger.info("✅ KPI monitoring started")
        
        # Initialize policy system
            self.logger.info("Initializing policy enforcement...")
            self._initialize_policy_enforcement()
            self.logger.info("✅ Policy enforcement initialized")
        
        # Register message handlers
            self.logger.info("Registering message handlers...")
            self._register_message_handlers()
            self.logger.info("✅ Message handlers registered")
        
        # Start main event loop
            self.state = AgentState.ACTIVE
            self.logger.info("Enhanced Feedback Agent started successfully")
        
        # Start background tasks
            self.logger.info("Starting background tasks...")
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._monitoring_loop())
            asyncio.create_task(self._cleanup_loop())
            asyncio.create_task(self._policy_metrics_loop())
            self.logger.info("✅ Background tasks started")
        
        # Main message processing loop
            self.logger.info("Starting main message processing loop...")
            await self._main_loop()
        
        except Exception as e:
            self.logger.error(f"Failed to start enhanced agent: {e}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            self.state = AgentState.ERROR
            raise
    
    def _initialize_policy_enforcement(self):
        """Initialize policy enforcement system"""
        try:
            # Log policy system initialization
            policy_summary = get_policy_summary()
            self.logger.info(f"Initializing policy enforcement system:")
            self.logger.info(f"  - Total policies: {policy_summary['total_policies']}")
            self.logger.info(f"  - 3GPP policies: {policy_summary['3gpp_policies']}")
            self.logger.info(f"  - Nokia policies: {policy_summary['nokia_policies']}")
            self.logger.info(f"  - Hybrid policies: {policy_summary['hybrid_policies']}")
            self.logger.info(f"  - Critical policies: {policy_summary['critical_policies']}")
            
            # Set default active services for all BSs (can be updated dynamically)
            default_services = ['voice', 'data', 'video', 'iot', 'urllc']
            
            self.logger.info("Policy enforcement system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize policy enforcement: {e}")
            raise ConfigurationError(f"Policy system initialization failed: {e}")
    
    def _register_message_handlers(self):
        """Register handlers for different message types"""
        handlers = {
        MessageType.OPTIMIZATION_RESPONSE: self._handle_validation_request,

        MessageType.VALIDATE_ENERGY_PLAN: self._handle_validation_request,
        MessageType.HEARTBEAT: self._handle_heartbeat,
        MessageType.STATUS_REQUEST: self._handle_status_request,
        # Keep existing handlers but use string keys for non-standard types
        'CONFIG_APPLIED': self._handle_config_applied,
        'PERFORMANCE_ALERT': self._handle_performance_alert,
        'ROLLBACK_REQUEST': self._handle_rollback_request,
        'UPDATE_BS_SERVICES': self._handle_bs_services_update,
        'POLICY_OVERRIDE_REQUEST': self._handle_policy_override
        }
        self.communication.register_handlers(handlers)

    
    async def _main_loop(self):
        """Enhanced main agent event loop"""
        while self.state not in [AgentState.SHUTDOWN, AgentState.ERROR]:
            try:
                # Process incoming messages
                await self.communication.process_messages(timeout=1.0)
                
                # Check for timeouts and cleanup
                await self._check_timeouts()
                
                # Update agent metrics
                self._update_metrics()
                
                # Update policy compliance metrics
                self._update_policy_compliance_metrics()
                
            except Exception as e:
                self.logger.error(f"Error in enhanced main loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _handle_validation_request(self, message: Dict) -> Dict:
        """Enhanced energy plan validation with comprehensive policy checking"""
        start_time = time.time()
        policy_check_start = time.time()
        
        try:
            # ✅ ADD: Debug logging to see what fields we're receiving
            #self.logger.info(f"📥 Received validation request with fields: {list(message.keys())}")
            #self.logger.info(f"📥 Message payload: {message}")
        
        # Check for required fields
            #required_fields = ['plan_id', 'target_bs_id', 'proposed_config', 'expected_energy_saving', 'confidence_score', 'timestamp']
            #missing_fields = [field for field in required_fields if field not in message]
            if 'energy_saving_percentage' in message and 'expected_energy_saving' not in message:
                message['expected_energy_saving'] = message['energy_saving_percentage']
        
        # Debug logging (you can remove this after it works)
            self.logger.info(f"📥 Received validation request with fields: {list(message.keys())}")
        
        # Check for required fields
            required_fields = ['plan_id', 'target_bs_id', 'proposed_config', 'expected_energy_saving', 'confidence_score', 'timestamp']
            missing_fields = [field for field in required_fields if field not in message]
            if missing_fields:
                self.logger.error(f"❌ Missing required fields: {missing_fields}")
                self.logger.error(f"❌ Available fields: {list(message.keys())}")
                return {
                'plan_id': message.get('plan_id', 'unknown'),
                'is_valid': False,
                'error': f"Missing required fields: {missing_fields}",
                'validation_time': time.time() - start_time,
                'policy_check_failed': True
                }
            # Parse energy plan
            plan = EnergyPlan(
                plan_id=message['plan_id'],
                target_bs_id=message['target_bs_id'],
                proposed_config=message['proposed_config'],
                expected_energy_saving=message['expected_energy_saving'],
                confidence_score=message['confidence_score'],
                timestamp=datetime.fromisoformat(message['timestamp']),
                optimization_metadata=message.get('metadata', {})
            )
            
            self.logger.info(f"Validating energy plan {plan.plan_id} for BS {plan.target_bs_id} with comprehensive policy enforcement")
            self.pending_validations[plan.plan_id] = plan
            self.state = AgentState.VALIDATING
            
            # Perform enhanced validation with policy checking
            result = await self._validate_energy_plan_comprehensive(plan)
            
            # Update comprehensive metrics
            self.validation_metrics['total_validations'] += 1
            if result.is_valid:
                self.validation_metrics['successful_validations'] += 1
                if not result.policy_blocks_optimization:
                    self.validation_metrics['policy_compliant_optimizations'] += 1
            else:
                self.validation_metrics['rejected_plans'] += 1
            
            # Update policy-specific metrics
            self._update_policy_violation_metrics(result)
            
            validation_time = time.time() - start_time
            policy_check_time = time.time() - policy_check_start
            self._update_avg_validation_time(validation_time)
            self._update_avg_policy_check_time(policy_check_time)
            
            # Store validation result
            self.validation_history.append(result)
            
            # Clean up
            del self.pending_validations[plan.plan_id]
            self.state = AgentState.ACTIVE
            
            # Enhanced response with policy information
            response = {
                'plan_id': result.plan_id,
                'is_valid': result.is_valid,
                'validation_score': result.validation_score,
                'policy_compliance_score': result.policy_compliance_score,
                'violations': result.violations,
                'critical_policy_violations': result.critical_policy_violations,
                'policy_blocks_optimization': result.policy_blocks_optimization,
                'recommended_adjustments': result.recommended_adjustments,
                'metadata': result.validation_metadata,
                'validation_time': validation_time,
                'policy_check_time': policy_check_time
            }
            
            status = "APPROVED" if result.is_valid else "REJECTED"
            if result.policy_blocks_optimization:
                status += " (BLOCKED BY CRITICAL POLICIES)"
                
            self.logger.info(f"Enhanced validation completed for plan {plan.plan_id}: {status}")
            return response
            
        except Exception as e:
            self.logger.error(f"Enhanced validation request failed: {e}")
            self.state = AgentState.ACTIVE
            return {
                'plan_id': message.get('plan_id', 'unknown'),
                'is_valid': False,
                'error': str(e),
                'validation_time': time.time() - start_time,
                'policy_check_failed': True
            }
    
    async def _validate_energy_plan_comprehensive(self, plan: EnergyPlan) -> ValidationResult:
        """
        Comprehensive validation logic with Nokia, 3GPP, and hybrid policies
        This is the core enhancement with full policy integration
        """
        
        # Get current KPI snapshot
        current_kpis = await self.kpi_monitor.get_current_kpis(plan.target_bs_id)
        
        # Predict impact of proposed changes using existing validator
        predicted_impact = await self.validator.predict_impact(
            plan.proposed_config, 
            current_kpis
        )
        
        # Convert KPISnapshot to dictionary for policy validation
        current_kpi_dict = {
            'avg_latency': current_kpis.avg_latency,
            'avg_throughput': current_kpis.avg_throughput,
            'packet_loss_rate': current_kpis.packet_loss_rate,
            'energy_consumption': current_kpis.energy_consumption,
            'energy_efficiency': current_kpis.energy_efficiency,
            'prb_utilization': current_kpis.prb_utilization,
            'tx_power': current_kpis.tx_power,
            'mac_tb_loss_rate': current_kpis.mac_tb_loss_rate,
            'active_users': current_kpis.active_users
        }
        
        # Calculate predicted KPI values after optimization
        predicted_kpi_dict = self._calculate_predicted_kpis(
            current_kpi_dict, predicted_impact, plan.proposed_config
        )
        
        # === COMPREHENSIVE POLICY VALIDATION ===
        bs_type = self.bs_type_mapping.get(plan.target_bs_id, 'nokia_airscale')
        active_services = self.active_services_per_bs.get(
            plan.target_bs_id, 
            ['voice', 'data', 'video', 'iot']  # Default services
        )
        
        policy_validation = validate_energy_plan_policies(
            bs_id=plan.target_bs_id,
            current_kpis=current_kpi_dict,
            predicted_kpis=predicted_kpi_dict,
            bs_type=bs_type,
            active_services=active_services,
        )
        
        # Extract critical violations that block optimization
        critical_policy_violations = get_critical_policy_violations(policy_validation)
        
        # Run traditional validation checks (backward compatibility)
        validation_checks = await self.validator.validate_plan(
            plan, 
            current_kpis, 
            predicted_impact
        )
        
        # === COMBINE ALL VALIDATION RESULTS ===
        all_violations = []
        
        # Add policy violations with enhanced metadata
        for violation in policy_validation['all_violations']:
            all_violations.append({
                'type': 'policy_violation',
                'policy_id': violation.get('policy_id', 'unknown'),
                'policy_type': violation.get('type', 'unknown'),
                'severity': violation.get('severity', 'medium'),
                'message': violation.get('message', 'Policy violation detected'),
                'blocks_optimization': violation.get('blocks_optimization', False),
                'source': 'comprehensive_policy_engine',
                'policy_category': violation.get('category', 'general'),
                'standard': violation.get('standard', 'unknown')
            })
        
        # Add traditional validation failures
        for check in validation_checks:
            if not check['passed']:
                all_violations.append({
                    'type': 'validation_check_failure',
                    'severity': check.get('severity', 'medium'),
                    'message': check.get('message', 'Validation check failed'),
                    'blocks_optimization': check.get('critical', False),
                    'source': 'traditional_validator'
                })
        
        # === DETERMINE OVERALL VALIDATION RESULT ===
        blocking_violations = [v for v in all_violations if v.get('blocks_optimization', False)]
        is_valid = len(blocking_violations) == 0
        
        # Calculate comprehensive validation scores
        total_policies_checked = policy_validation['total_policies_checked']
        policies_compliant = policy_validation['policies_compliant']
        traditional_checks_passed = len([c for c in validation_checks if c['passed']])
        total_traditional_checks = len(validation_checks)
        
        # Policy compliance score (0.0 to 1.0)
        policy_compliance_score = (
            policies_compliant / total_policies_checked 
            if total_policies_checked > 0 else 1.0
        )
        
        # Traditional validation score (0.0 to 1.0)
        traditional_score = (
            traditional_checks_passed / total_traditional_checks 
            if total_traditional_checks > 0 else 1.0
        )
        
        # Weighted overall validation score: 70% policy, 30% traditional
        overall_validation_score = (0.7 * policy_compliance_score) + (0.3 * traditional_score)
        
        # Count critical policy violations
        critical_violations_count = len([
            v for v in all_violations 
            if v.get('severity') == 'critical' and v.get('type') == 'policy_violation'
        ])
        
        # Determine if policies block optimization
        policy_blocks_optimization = len(critical_policy_violations) > 0
        
        # === GENERATE ENHANCED RECOMMENDATIONS ===
        recommended_adjustments = await self._generate_comprehensive_recommendations(
            plan, all_violations, policy_validation, predicted_impact
        )
        
        # === CREATE ENHANCED VALIDATION RESULT ===
        validation_result = ValidationResult(
            plan_id=plan.plan_id,
            is_valid=is_valid,
            validation_score=overall_validation_score,
            policy_compliance_score=policy_compliance_score,
            critical_policy_violations=critical_violations_count,
            policy_blocks_optimization=policy_blocks_optimization,
            violations=all_violations,
            recommended_adjustments=recommended_adjustments,
            validation_metadata={
                # Traditional validation data
                'validation_checks': validation_checks,
                'predicted_impact': predicted_impact,
                'current_kpis': current_kpis.__dict__,
                
                # Comprehensive policy validation data
                'policy_validation': policy_validation,
                'total_policies_checked': total_policies_checked,
                'policies_compliant': policies_compliant,
                'critical_policy_violations': len(critical_policy_violations),
                'blocking_violations': len(blocking_violations),
                'policy_score': policy_compliance_score,
                'traditional_score': traditional_score,
                
                # Enhanced metadata
                'bs_type': bs_type,
                'active_services': active_services,
                'predicted_kpis': predicted_kpi_dict,
                'validation_timestamp': datetime.now().isoformat(),
                'validator_version': '3.0_comprehensive_policy_enforcement',
                'policy_framework': 'Nokia+3GPP+Hybrid'
            },
            timestamp=datetime.now()
        )
        
        # Log comprehensive validation details
        self._log_comprehensive_validation_details(plan, validation_result, policy_validation)
        
        return validation_result
    
    def _calculate_predicted_kpis(
        self,
        current_kpis: Dict[str, float],
        predicted_impact: Dict[str, Any],
        proposed_config: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate post-change KPIs from live measurements."""

        predicted_kpis = current_kpis.copy()

        predicted_kpis["avg_latency"] = (
            float(current_kpis["avg_latency"])
            + float(
                predicted_impact.get(
                    "latency_change", 0.0
                )
            )
        )

        predicted_kpis["avg_throughput"] = (
            float(current_kpis["avg_throughput"])
            + float(
                predicted_impact.get(
                    "throughput_change", 0.0
                )
            )
        )

        predicted_kpis["packet_loss_rate"] = (
            float(current_kpis["packet_loss_rate"])
            + float(
                predicted_impact.get(
                    "packet_loss_change", 0.0
                )
            )
        )

        predicted_kpis["prb_utilization"] = (
            float(current_kpis["prb_utilization"])
            + float(
                predicted_impact.get(
                    "prb_utilization_change", 0.0
                )
            )
        )

        energy_saving_pct = max(
            0.0,
            min(
                100.0,
                float(
                    predicted_impact.get(
                        "energy_saving", 0.0
                    )
                ),
            ),
        )

        predicted_kpis["energy_consumption"] = (
            float(current_kpis["energy_consumption"])
            * (1.0 - energy_saving_pct / 100.0)
        )

        predicted_kpis["tx_power"] = float(
            proposed_config.get(
                "tx_power_dbm",
                proposed_config.get(
                    "tx_power",
                    current_kpis["tx_power"],
                ),
            )
        )

        # Apply only physical lower bounds. Do not clamp invalid
        # proposals into compliance before policy validation.
        predicted_kpis["avg_latency"] = max(
            0.1,
            predicted_kpis["avg_latency"],
        )
        predicted_kpis["avg_throughput"] = max(
            0.0,
            predicted_kpis["avg_throughput"],
        )
        predicted_kpis["packet_loss_rate"] = max(
            0.0,
            predicted_kpis["packet_loss_rate"],
        )
        predicted_kpis["prb_utilization"] = max(
            0.0,
            min(
                100.0,
                predicted_kpis["prb_utilization"],
            ),
        )
        predicted_kpis["energy_consumption"] = max(
            0.1,
            predicted_kpis["energy_consumption"],
        )

        # Throughput in bit/s divided by power in joule/s
        # produces energy efficiency in bit/joule.
        predicted_kpis["energy_efficiency"] = (
            predicted_kpis["avg_throughput"]
            * 1_000_000.0
            / predicted_kpis["energy_consumption"]
        )

        return predicted_kpis
    
    async def _generate_comprehensive_recommendations(self, plan: EnergyPlan, 
                                                    violations: List[Dict],
                                                    policy_validation: Dict[str, Any],
                                                    predicted_impact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate comprehensive recommendations addressing policy violations"""
        
        if not violations:
            return None
        
        recommendations = {
            'config_modifications': {},
            'policy_compliance_actions': [],
            'risk_mitigation': [],
            'alternative_optimizations': [],
            'phased_implementation': [],
            'nokia_specific_optimizations': [],
            'compliance_monitoring': []
        }
        
        # Analyze violations by type and severity
        critical_violations = [v for v in violations if v.get('severity') == 'critical']
        high_violations = [v for v in violations if v.get('severity') == 'high']
        policy_violations = [v for v in violations if v.get('type') == 'policy_violation']
        nokia_violations = [v for v in policy_violations if v.get('standard') == 'nokia']
        gpp_violations = [v for v in policy_violations if v.get('standard') == '3gpp']
        
        # Handle critical violations (blocking)
        for violation in critical_violations:
            message = violation.get('message', '').lower()
            
            if 'latency' in message:
                recommendations['config_modifications']['enable_low_latency_mode'] = True
                recommendations['config_modifications']['reduce_processing_delay'] = True
                recommendations['policy_compliance_actions'].append(
                    "Enable Nokia ultra-low latency algorithms for critical services"
                )
                recommendations['phased_implementation'].append(
                    "Phase 1: Implement latency optimizations before energy savings"
                )
                
            if 'energy' in message or 'power' in message:
                recommendations['config_modifications']['gradual_power_reduction'] = True
                recommendations['policy_compliance_actions'].append(
                    "Implement gradual energy optimization with continuous monitoring"
                )
                recommendations['alternative_optimizations'].append(
                    "Consider antenna-level power optimization instead of full BS power reduction"
                )
        
        # Handle Nokia-specific violations
        if nokia_violations:
            recommendations['nokia_specific_optimizations'].extend([
                "Apply Nokia AirScale energy efficiency algorithms",
                "Enable Nokia thermal management features",
                "Utilize Nokia Massive MIMO optimization"
            ])
            
            for violation in nokia_violations:
                if 'thermal' in violation.get('message', '').lower():
                    recommendations['config_modifications']['enable_thermal_management'] = True
                    recommendations['risk_mitigation'].append(
                        "Monitor BS temperature during optimization"
                    )
                    
                if 'mimo' in violation.get('message', '').lower():
                    recommendations['config_modifications']['optimize_mimo_configuration'] = True
        
        # Handle 3GPP violations
        if gpp_violations:
            recommendations['policy_compliance_actions'].extend([
                "Ensure 3GPP QoS requirements are maintained",
                "Implement 3GPP-compliant energy saving procedures"
            ])
            
            for violation in gpp_violations:
                if '5qi' in violation.get('message', '').lower():
                    recommendations['config_modifications']['prioritize_critical_bearers'] = True
                    recommendations['compliance_monitoring'].append(
                        "Monitor 5QI compliance during energy optimization"
                    )
        
        # Handle high-severity violations (non-blocking but important)
        for violation in high_violations:
            message = violation.get('message', '').lower()
            
            if 'throughput' in message:
                recommendations['risk_mitigation'].append(
                    "Monitor throughput degradation during energy optimization"
                )
                recommendations['alternative_optimizations'].append(
                    "Consider load-based dynamic energy optimization"
                )
                
            if 'efficiency' in message:
                recommendations['phased_implementation'].append(
                    "Phase 2: Gradual efficiency improvements with performance monitoring"
                )
        
        # Add general recommendations based on energy saving target
        energy_saving = plan.expected_energy_saving
        if energy_saving > 30:  # High energy saving target
            recommendations['phased_implementation'].extend([
                "Phase 1: Implement 10% energy saving with monitoring",
                "Phase 2: Gradually increase to target based on performance",
                "Phase 3: Full optimization with continuous validation"
            ])
            
        if energy_saving > 20:  # Moderate energy saving
            recommendations['compliance_monitoring'].extend([
                "Monitor KPIs for 1 hour after each optimization step",
                "Implement automatic rollback if violations occur",
                "Generate hourly compliance reports"
            ])
        
        # Add Nokia product-specific recommendations
        recommendations['nokia_specific_optimizations'].extend([
            "Leverage Nokia AirScale BaseFlow traffic steering",
            "Enable Nokia ReefShark beamforming optimization",
            "Utilize Nokia energy-aware load balancing"
        ])
        
        return recommendations if any(recommendations.values()) else None
    
    def _update_policy_violation_metrics(self, result: ValidationResult):
        """Update detailed policy violation metrics"""
        
        # Update policy-specific counters
        self.validation_metrics['critical_policy_violations'] += result.critical_policy_violations
        
        # Count violations by type
        for violation in result.violations:
            if violation.get('type') == 'policy_violation':
                standard = violation.get('standard', 'unknown')
                severity = violation.get('severity', 'medium')
                
                # Count by standard
                if standard == '3gpp':
                    self.validation_metrics['policy_violations_3gpp'] += 1
                elif standard == 'nokia':
                    self.validation_metrics['policy_violations_nokia'] += 1
                elif standard == 'hybrid':
                    self.validation_metrics['policy_violations_hybrid'] += 1
                    
                # Count by severity
                if severity == 'critical':
                    self.validation_metrics['critical_policy_violations'] += 1
                elif severity == 'high':
                    self.validation_metrics['high_policy_violations'] += 1
                elif severity == 'medium':
                    self.validation_metrics['medium_policy_violations'] += 1
                elif severity == 'low':
                    self.validation_metrics['low_policy_violations'] += 1
                
                # Count specific policy types
                policy_category = violation.get('policy_category', '')
                if 'energy' in policy_category.lower():
                    self.validation_metrics['policy_violations_energy'] += 1
                if 'qos' in policy_category.lower() or '5qi' in policy_category.lower():
                    self.validation_metrics['qci_5qi_violations'] += 1
        
        # Update total violations
        policy_violations = [v for v in result.violations if v.get('type') == 'policy_violation']
        self.validation_metrics['policy_violations_total'] += len(policy_violations)
        
        # Track if optimization was blocked
        if result.policy_blocks_optimization:
            self.validation_metrics['critical_violations_blocked'] += 1
    
    def _log_comprehensive_validation_details(self, plan: EnergyPlan, result: ValidationResult, 
                                           policy_validation: Dict[str, Any]):
        """Log detailed comprehensive validation information"""
        
        self.logger.info(f"=== COMPREHENSIVE VALIDATION DETAILS FOR PLAN {plan.plan_id} ===")
        self.logger.info(f"Target BS: {plan.target_bs_id}")
        self.logger.info(f"Expected Energy Saving: {plan.expected_energy_saving:.2f}%")
        self.logger.info(f"Confidence Score: {plan.confidence_score:.3f}")
        
        # Overall validation results
        self.logger.info(f"Overall Valid: {result.is_valid}")
        self.logger.info(f"Overall Validation Score: {result.validation_score:.3f}")
        self.logger.info(f"Policy Compliance Score: {result.policy_compliance_score:.3f}")
        
        # Policy validation summary
        self.logger.info(f"=== POLICY VALIDATION SUMMARY ===")
        self.logger.info(f"Total Policies Checked: {policy_validation['total_policies_checked']}")
        self.logger.info(f"Policies Compliant: {policy_validation['policies_compliant']}")
        self.logger.info(f"Total Policy Violations: {policy_validation['total_violations']}")
        self.logger.info(f"Critical Policy Violations: {result.critical_policy_violations}")
        self.logger.info(f"Blocks Optimization: {result.policy_blocks_optimization}")
        
        # Violation breakdown by severity
        violation_by_severity = {}
        for violation in result.violations:
            sev = violation.get('severity', 'unknown')
            violation_by_severity[sev] = violation_by_severity.get(sev, 0) + 1
        
        self.logger.info(f"Violations by Severity: {violation_by_severity}")
        
        # Violation breakdown by standard
        violation_by_standard = {}
        for violation in result.violations:
            if violation.get('type') == 'policy_violation':
                std = violation.get('standard', 'unknown')
                violation_by_standard[std] = violation_by_standard.get(std, 0) + 1
        
        if violation_by_standard:
            self.logger.info(f"Policy Violations by Standard: {violation_by_standard}")
        
        # Log specific critical violations
        critical_violations = [v for v in result.violations if v.get('severity') == 'critical']
        if critical_violations:
            self.logger.warning(f"=== CRITICAL VIOLATIONS (BLOCKING) ===")
            for violation in critical_violations:
                self.logger.warning(f"  - {violation.get('message', 'Unknown violation')}")
        
        # Log Nokia-specific violations
        nokia_violations = [v for v in result.violations 
                           if v.get('type') == 'policy_violation' and v.get('standard') == 'nokia']
        if nokia_violations:
            self.logger.info(f"=== NOKIA POLICY VIOLATIONS ===")
            for violation in nokia_violations:
                self.logger.info(f"  - {violation.get('message', 'Unknown Nokia violation')}")
        
        # Log 3GPP violations
        gpp_violations = [v for v in result.violations 
                         if v.get('type') == 'policy_violation' and v.get('standard') == '3gpp']
        if gpp_violations:
            self.logger.info(f"=== 3GPP POLICY VIOLATIONS ===")
            for violation in gpp_violations:
                self.logger.info(f"  - {violation.get('message', 'Unknown 3GPP violation')}")
        
        # Log recommendations summary
        if result.recommended_adjustments:
            rec_count = len(result.recommended_adjustments.get('policy_compliance_actions', []))
            rec_count += len(result.recommended_adjustments.get('nokia_specific_optimizations', []))
            self.logger.info(f"Generated {rec_count} comprehensive recommendations")
        
        self.logger.info(f"=== END COMPREHENSIVE VALIDATION DETAILS ===")
    
    # Enhanced message handlers
    async def _handle_bs_services_update(self, message: Dict) -> Dict:
        """Handle updates to active services per base station"""
        bs_id = message['bs_id']
        active_services = message['active_services']
        
        self.active_services_per_bs[bs_id] = active_services
        self.logger.info(f"Updated active services for BS {bs_id}: {active_services}")
        
        return {'status': 'services_updated'}
    
    async def _handle_policy_override(self, message: Dict) -> Dict:
        """Handle temporary policy override requests"""
        # This could be implemented for emergency situations
        # For now, just log and acknowledge
        self.logger.warning(f"Policy override request received: {message}")
        return {'status': 'override_logged', 'action': 'manual_review_required'}
    
    # Enhanced monitoring and background tasks
    async def _policy_metrics_loop(self):
        """Background task for updating policy compliance metrics"""
        while self.state not in [AgentState.SHUTDOWN, AgentState.ERROR]:
            try:
                # Update policy compliance rate
                if self.validation_metrics['total_validations'] > 0:
                    compliant_validations = (
                        self.validation_metrics['policy_compliant_optimizations']
                    )
                    total_validations = self.validation_metrics['total_validations']
                    self.validation_metrics['policy_compliance_rate'] = (
                        compliant_validations / total_validations
                    )
                
                # Log policy metrics every 10 minutes
                if int(time.time()) % 600 == 0:  # Every 10 minutes
                    self._log_policy_metrics_summary()
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Policy metrics loop error: {e}")
                await asyncio.sleep(60)
    
    def _log_policy_metrics_summary(self):
        """Log comprehensive policy metrics summary"""
        metrics = self.validation_metrics
        
        self.logger.info(f"=== POLICY ENFORCEMENT METRICS SUMMARY ===")
        self.logger.info(f"Total Validations: {metrics['total_validations']}")
        self.logger.info(f"Policy Compliance Rate: {metrics['policy_compliance_rate']:.3f}")
        self.logger.info(f"Critical Violations Blocked: {metrics['critical_violations_blocked']}")
        
        # Violations by standard
        self.logger.info(f"Policy Violations - 3GPP: {metrics['policy_violations_3gpp']}")
        self.logger.info(f"Policy Violations - Nokia: {metrics['policy_violations_nokia']}")
        self.logger.info(f"Policy Violations - Hybrid: {metrics['policy_violations_hybrid']}")
        
        # Violations by severity
        self.logger.info(f"Critical Violations: {metrics['critical_policy_violations']}")
        self.logger.info(f"High Violations: {metrics['high_policy_violations']}")
        self.logger.info(f"Medium Violations: {metrics['medium_policy_violations']}")
        
        self.logger.info(f"Average Policy Check Time: {metrics['avg_policy_check_time']:.3f}s")
        self.logger.info(f"=== END POLICY METRICS SUMMARY ===")
    
    def _update_avg_policy_check_time(self, policy_check_time: float):
        """Update average policy check time metric"""
        current_avg = self.validation_metrics['avg_policy_check_time']
        total_validations = self.validation_metrics['total_validations']
        
        if total_validations > 0:
            new_avg = ((current_avg * (total_validations - 1)) + policy_check_time) / total_validations
            self.validation_metrics['avg_policy_check_time'] = new_avg
    
    def _update_policy_compliance_metrics(self):
        """Update policy compliance metrics"""
        # This can be extended for real-time policy compliance tracking
        pass
    
    # Enhanced status request with comprehensive policy information
    async def _handle_status_request(self, message: Dict) -> Dict:
        """Enhanced status request with comprehensive policy metrics"""
        base_status = {
            'agent_id': self.agent_id,
            'state': self.state.value,
            'uptime': (datetime.now() - self.startup_time).total_seconds(),
            'pending_validations': len(self.pending_validations),
            'monitored_bs': list(self.current_configs.keys()),
            'last_activity': datetime.now().isoformat()
        }
        
        # Add comprehensive metrics including policy metrics
        base_status['metrics'] = self.validation_metrics
        
        # Add policy system status
        base_status['policy_system'] = get_policy_summary()
        
        # Add active services tracking
        base_status['active_services_per_bs'] = self.active_services_per_bs
        
        # Add recent validation summary
        recent_validations = [v for v in self.validation_history 
                             if (datetime.now() - v.timestamp).total_seconds() < 3600]  # Last hour
        base_status['recent_validations'] = {
            'total': len(recent_validations),
            'approved': len([v for v in recent_validations if v.is_valid]),
            'policy_blocked': len([v for v in recent_validations if v.policy_blocks_optimization])
        }
        
        return base_status
    
    # Keep all existing methods (they remain unchanged)
    async def _handle_config_applied(self, message: Dict) -> Dict:
        """Handle notification that configuration was applied to BS"""
        bs_id = message['bs_id']
        applied_config = message['applied_config']
        plan_id = message.get('plan_id', 'unknown')
        
        self.logger.info(f"Configuration applied to BS {bs_id} (plan: {plan_id})")
        
        # Update current config tracking
        self.current_configs[bs_id] = {
            'config': applied_config,
            'applied_at': datetime.now(),
            'plan_id': plan_id
        }
        
        # Start monitoring for performance impact
        await self.kpi_monitor.start_monitoring_window(
            bs_id, 
            duration=self.config['monitoring']['post_change_window']
        )
        
        return {'status': 'acknowledged'}
    
    async def _handle_performance_alert(self, message: Dict) -> Dict:
        """Handle performance degradation alerts from monitoring"""
        bs_id = message['bs_id']
        alert_type = message['alert_type']
        severity = message['severity']
        metrics = message['metrics']
        
        self.logger.warning(f"Performance alert for BS {bs_id}: {alert_type} (severity: {severity})")
        
        # Determine if rollback is needed
        if severity in ['HIGH', 'CRITICAL']:
            rollback_decision = await self._evaluate_rollback_need(bs_id, metrics)
            
            if rollback_decision['should_rollback']:
                self.logger.warning(f"Triggering rollback for BS {bs_id}")
                await self._trigger_rollback(bs_id, rollback_decision['reason'])
        
        return {'status': 'processed'}
    
    async def _evaluate_rollback_need(self, bs_id: str, metrics: Dict) -> Dict:
        """Enhanced rollback evaluation with policy awareness"""
        
        # Get rollback policies
        policies = self.config['rollback']['policies']
        
        # Check each policy
        for policy in policies:
            metric_name = policy['metric']
            threshold = policy['threshold']
            comparison = policy['comparison']  # 'gt', 'lt', 'eq'
            
            if metric_name in metrics:
                metric_value = metrics[metric_name]
                
                should_rollback = False
                if comparison == 'gt' and metric_value > threshold:
                    should_rollback = True
                elif comparison == 'lt' and metric_value < threshold:
                    should_rollback = True
                elif comparison == 'eq' and abs(metric_value - threshold) < 0.001:
                    should_rollback = True
                
                if should_rollback:
                    return {
                        'should_rollback': True,
                        'reason': f"{metric_name} violated: {metric_value} {comparison} {threshold}",
                        'violated_policy': policy,
                        'trigger_type': 'traditional_policy'
                    }
        
        # Enhanced: Check against comprehensive policies
        current_kpis = {
            'avg_latency': metrics.get('avg_latency', 0),
            'avg_throughput': metrics.get('avg_throughput', 0),
            'packet_loss_rate': metrics.get('packet_loss_rate', 0),
            'energy_consumption': metrics.get('energy_consumption', 0),
            'energy_efficiency': metrics.get('energy_efficiency', 0),
            'prb_utilization': metrics.get('prb_utilization', 0),
            'tx_power': metrics.get('tx_power', 0),
            'mac_tb_loss_rate': metrics.get('mac_tb_loss_rate', 0),
            'active_users': metrics.get('active_users', 0)
        }
        
        # Use same KPIs for current and predicted (rollback evaluation)
        policy_check = validate_energy_plan_policies(bs_id, current_kpis, current_kpis)
        
        # Check for critical policy violations
        critical_violations = get_critical_policy_violations(policy_check)
        
        if critical_violations:
            return {
                'should_rollback': True,
                'reason': f"Critical policy violations detected: {len(critical_violations)} violations",
                'violated_policies': critical_violations,
                'trigger_type': 'comprehensive_policy_violation'
            }
        
        return {'should_rollback': False, 'reason': 'No policy violations detected'}
    
    async def _trigger_rollback(self, bs_id: str, reason: str):
        """Trigger rollback for a specific base station"""
        try:
            self.state = AgentState.ROLLING_BACK
            
            # Execute rollback
            rollback_result = await self.rollback_engine.execute_rollback(bs_id, reason)
            
            # Record rollback
            rollback_record = {
                'bs_id': bs_id,
                'reason': reason,
                'timestamp': datetime.now(),
                'success': rollback_result['success'],
                'details': rollback_result
            }
            
            self.rollback_history.append(rollback_record)
            self.validation_metrics['rollbacks_triggered'] += 1
            
            # Notify other agents
            await self.communication.broadcast('ROLLBACK_COMPLETED', {
                'bs_id': bs_id,
                'success': rollback_result['success'],
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info(f"Rollback completed for BS {bs_id}: {'SUCCESS' if rollback_result['success'] else 'FAILED'}")
            
        except Exception as e:
            self.logger.error(f"Rollback failed for BS {bs_id}: {e}")
        finally:
            self.state = AgentState.ACTIVE
    
    async def _handle_rollback_request(self, message: Dict) -> Dict:
        """Handle manual rollback request"""
        bs_id = message['bs_id']
        reason = message.get('reason', 'Manual rollback request')
        
        self.logger.info(f"Manual rollback requested for BS {bs_id}")
        
        await self._trigger_rollback(bs_id, reason)
        
        return {'status': 'rollback_initiated'}
    
    async def _handle_heartbeat(self, message: Dict) -> Dict:
        """Handle heartbeat from other agents"""
        return {
            'agent_id': self.agent_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'alive'
        }
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to other agents"""
        while self.state not in [AgentState.SHUTDOWN, AgentState.ERROR]:
            try:
                await self.communication.send_heartbeat()
                await asyncio.sleep(self.config['agent']['heartbeat_interval'])
            except Exception as e:
                self.logger.error(f"Heartbeat failed: {e}")
                await asyncio.sleep(5.0)
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop for performance tracking"""
        while self.state not in [AgentState.SHUTDOWN, AgentState.ERROR]:
            try:
                # Check all monitored base stations
                for bs_id in self.current_configs.keys():
                    await self.kpi_monitor.check_performance(bs_id)
                
                await asyncio.sleep(self.config['monitoring']['check_interval'])
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10.0)
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old data and logs"""
        while self.state not in [AgentState.SHUTDOWN, AgentState.ERROR]:
            try:
                # Cleanup old validation history
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.validation_history = [
                    v for v in self.validation_history 
                    if v.timestamp > cutoff_time
                ]
                
                # Cleanup old rollback history
                self.rollback_history = [
                    r for r in self.rollback_history 
                    if r['timestamp'] > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _check_timeouts(self):
        """Check for and handle validation timeouts"""
        timeout_duration = self.config['validation']['timeout']
        current_time = datetime.now()
        
        timed_out_plans = []
        for plan_id, plan in self.pending_validations.items():
            if (current_time - plan.timestamp).total_seconds() > timeout_duration:
                timed_out_plans.append(plan_id)
        
        for plan_id in timed_out_plans:
            self.logger.warning(f"Validation timeout for plan {plan_id}")
            del self.pending_validations[plan_id]
    
    def _update_avg_validation_time(self, validation_time: float):
        """Update average validation time metric"""
        current_avg = self.validation_metrics['avg_validation_time']
        total_validations = self.validation_metrics['total_validations']
        
        if total_validations > 0:
            new_avg = ((current_avg * (total_validations - 1)) + validation_time) / total_validations
            self.validation_metrics['avg_validation_time'] = new_avg
    
    def _update_metrics(self):
        """Update internal performance metrics"""
        # This can be extended to update more complex metrics
        pass
    
    async def shutdown(self):
        """Graceful shutdown of the enhanced agent"""
        self.logger.info("Shutting down Enhanced Feedback Agent...")
        self.state = AgentState.SHUTDOWN
        
        # Log final policy metrics
        self._log_policy_metrics_summary()
        
        # Stop monitoring
        await self.kpi_monitor.stop()
        
        # Close communication
        await self.communication.close()
        
        self.logger.info("Enhanced Feedback Agent shut down complete")


def main():
    """Main entry point for running the Enhanced Feedback Agent"""
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/agentCONFIG.yaml"
    
    agent = FeedbackAgent(config_path)
    
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("Shutdown requested by user")
    except Exception as e:
        print(f"Enhanced agent failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()