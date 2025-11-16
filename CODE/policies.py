"""
5G Network Policies for Auto EA-SORAN System
Comprehensive policy definitions from Nokia and 3GPP standards
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

class PolicyStandard(Enum):
    """Policy standard classification"""
    NOKIA_PROPRIETARY = "nokia"
    THREE_GPP = "3gpp"
    ITU = "itu"
    ETSI = "etsi"
    HYBRID = "hybrid"  # Nokia implementation of 3GPP standard

class PolicySeverity(Enum):
    """Policy violation severity levels"""
    CRITICAL = "critical"  # Service disruption, immediate rollback
    HIGH = "high"         # SLA violation, needs attention
    MEDIUM = "medium"     # Performance degradation
    LOW = "low"           # Minor optimization opportunity
    INFO = "info"         # Informational only

class PolicyCategory(Enum):
    """Policy category classification"""
    QOS_5QI = "qos_5qi"              # QoS Class Identifier policies
    ENERGY_EFFICIENCY = "energy"     # Energy consumption policies
    RADIO_RESOURCES = "radio"        # RAN resource management
    LATENCY_PERFORMANCE = "latency"  # Delay and response time
    THROUGHPUT = "throughput"        # Data rate policies
    RELIABILITY = "reliability"      # Error rates and availability
    NOKIA_PRODUCT = "nokia_product"  # Nokia-specific enforcement
    NETWORK_SLICING = "network_slicing"  # Network slice policies
    DENSE_DEPLOYMENT = "dense_deployment"  # Dense urban deployment policies
    IOT_SERVICES = "iot_services"    # IoT-specific policies
    AUTONOMOUS_VEHICLE = "autonomous_vehicle"  # V2X and autonomous vehicle policies

@dataclass
class Policy:
    """Universal policy definition structure"""
    policy_id: str
    name: str
    description: str
    standard: PolicyStandard
    category: PolicyCategory
    severity: PolicySeverity
    
    # Thresholds and limits
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    
    # Metadata
    unit: str = ""
    nokia_product: Optional[str] = None
    standard_reference: Optional[str] = None
    enforcement_notes: Optional[str] = None
    
    def validate(self, current_value: float, predicted_value: float) -> Dict[str, Any]:
        """Validate current and predicted values against policy"""
        violations = []
        
        # Check current value
        if self.min_value is not None and current_value < self.min_value:
            violations.append({
                'type': 'current_min_violation',
                'message': f"Current {self.name} ({current_value}{self.unit}) below minimum ({self.min_value}{self.unit})",
                'severity': self.severity.value
            })
            
        if self.max_value is not None and current_value > self.max_value:
            violations.append({
                'type': 'current_max_violation', 
                'message': f"Current {self.name} ({current_value}{self.unit}) exceeds maximum ({self.max_value}{self.unit})",
                'severity': self.severity.value
            })
        
        # Check predicted value (after energy optimization)
        if self.min_value is not None and predicted_value < self.min_value:
            violations.append({
                'type': 'predicted_min_violation',
                'message': f"Predicted {self.name} ({predicted_value}{self.unit}) would fall below minimum ({self.min_value}{self.unit})",
                'severity': self.severity.value,
                'blocks_optimization': True
            })
            
        if self.max_value is not None and predicted_value > self.max_value:
            violations.append({
                'type': 'predicted_max_violation',
                'message': f"Predicted {self.name} ({predicted_value}{self.unit}) would exceed maximum ({self.max_value}{self.unit})", 
                'severity': self.severity.value,
                'blocks_optimization': True
            })
        
        return {
            'policy_id': self.policy_id,
            'policy_name': self.name,
            'standard': self.standard.value,
            'category': self.category.value,
            'violations': violations,
            'is_compliant': len(violations) == 0,
            'current_value': current_value,
            'predicted_value': predicted_value,
            'target_value': self.target_value
        }

# =============================================================================
# 3GPP 5QI POLICIES (5G QoS Class Identifier Standards)
# =============================================================================

QCI_5QI_POLICIES = {
    # Real-time services
    1: Policy(
        policy_id="5QI_001_CONVERSATIONAL_VOICE",
        name="Conversational Voice",
        description="Voice calls over 5G with ultra-low latency requirements",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.QOS_5QI,
        severity=PolicySeverity.CRITICAL,
        max_value=100.0,  # 100ms max delay
        target_value=50.0,  # 50ms target
        unit="ms",
        standard_reference="3GPP TS 23.501"
    ),
    
    2: Policy(
        policy_id="5QI_002_CONVERSATIONAL_VIDEO",
        name="Conversational Video (Live)",
        description="Real-time video calls and conferencing",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.QOS_5QI,
        severity=PolicySeverity.CRITICAL,
        max_value=150.0,  # 150ms max delay
        target_value=100.0,  # 100ms target
        unit="ms",
        standard_reference="3GPP TS 23.501"
    ),
    
    3: Policy(
        policy_id="5QI_003_REAL_TIME_GAMING",
        name="Real-time Gaming",
        description="Interactive gaming with strict latency requirements",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.QOS_5QI,
        severity=PolicySeverity.HIGH,
        max_value=50.0,   # 50ms max delay
        target_value=20.0,  # 20ms target
        unit="ms",
        standard_reference="3GPP TS 23.501"
    ),
    
    # Non-real-time services
    4: Policy(
        policy_id="5QI_004_NON_RT_VIDEO",
        name="Non-real-time Video",
        description="Video streaming, buffered video content",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.QOS_5QI,
        severity=PolicySeverity.MEDIUM,
        max_value=300.0,  # 300ms max delay
        target_value=200.0,  # 200ms target
        unit="ms",
        standard_reference="3GPP TS 23.501"
    ),
    
    5: Policy(
        policy_id="5QI_005_IMS_SIGNALING",
        name="IMS Signaling",
        description="IP Multimedia Subsystem signaling traffic",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.QOS_5QI,
        severity=PolicySeverity.HIGH,
        max_value=100.0,  # 100ms max delay
        target_value=60.0,   # 60ms target
        unit="ms",
        standard_reference="3GPP TS 23.501"
    ),
    
    # Mission critical services
    65: Policy(
        policy_id="5QI_065_MISSION_CRITICAL_USER_PLANE",
        name="Mission Critical User Plane",
        description="Emergency services, public safety communications",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.QOS_5QI,
        severity=PolicySeverity.CRITICAL,
        max_value=75.0,   # 75ms max delay
        target_value=50.0,  # 50ms target
        unit="ms",
        standard_reference="3GPP TS 23.501"
    ),
    
    # Ultra-reliable low-latency (URLLC)
    82: Policy(
        policy_id="5QI_082_URLLC_DISCRETE_AUTOMATION",
        name="URLLC Discrete Automation",
        description="Industrial automation, discrete control",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.QOS_5QI,
        severity=PolicySeverity.CRITICAL,
        max_value=10.0,   # 10ms max delay
        target_value=5.0,   # 5ms target
        unit="ms",
        standard_reference="3GPP TS 23.501"
    )
}

# =============================================================================
# NOKIA PROPRIETARY POLICIES
# =============================================================================

NOKIA_POLICIES = {
    "NOKIA_ENERGY_EFFICIENCY": Policy(
        policy_id="NOK_001_ENERGY_EFFICIENCY",
        name="Nokia Energy Efficiency",
        description="Nokia-specific energy efficiency requirements for base stations",
        standard=PolicyStandard.NOKIA_PROPRIETARY,
        category=PolicyCategory.ENERGY_EFFICIENCY,
        severity=PolicySeverity.HIGH,
        min_value=1000.0,  # Minimum 1000 bytes per joule
        target_value=2000.0,  # Target 2000 bytes per joule
        unit="bytes/joule",
        nokia_product="Nokia AirScale Base Station",
        enforcement_notes="Energy savings must not compromise service quality"
    ),
    
    "NOKIA_RADIO_EFFICIENCY": Policy(
        policy_id="NOK_002_RADIO_EFFICIENCY",
        name="Nokia Radio Resource Efficiency",
        description="Optimal utilization of radio resources in Nokia equipment",
        standard=PolicyStandard.NOKIA_PROPRIETARY,
        category=PolicyCategory.RADIO_RESOURCES,
        severity=PolicySeverity.MEDIUM,
        min_value=70.0,    # Minimum 70% efficiency
        max_value=95.0,    # Maximum 95% to avoid overload
        target_value=85.0, # Target 85% efficiency
        unit="%",
        nokia_product="Nokia AirScale",
        enforcement_notes="Balance between efficiency and headroom for traffic spikes"
    ),
    
    "NOKIA_THERMAL_MANAGEMENT": Policy(
        policy_id="NOK_003_THERMAL_MGMT",
        name="Nokia Thermal Management",
        description="Temperature-based power scaling for Nokia base stations",
        standard=PolicyStandard.NOKIA_PROPRIETARY,
        category=PolicyCategory.ENERGY_EFFICIENCY,
        severity=PolicySeverity.HIGH,
        max_value=65.0,    # Maximum 65°C operating temperature
        target_value=50.0, # Target 50°C
        unit="°C",
        nokia_product="Nokia AirScale Base Station",
        enforcement_notes="Automatic power reduction above threshold temperatures"
    ),
    
    "NOKIA_MASSIVE_MIMO_EFFICIENCY": Policy(
        policy_id="NOK_004_MASSIVE_MIMO",
        name="Nokia Massive MIMO Efficiency",
        description="Energy-efficient massive MIMO operations",
        standard=PolicyStandard.NOKIA_PROPRIETARY,
        category=PolicyCategory.ENERGY_EFFICIENCY,
        severity=PolicySeverity.MEDIUM,
        min_value=80.0,    # Minimum 80% antenna utilization for efficiency
        target_value=90.0, # Target 90% utilization
        unit="%",
        nokia_product="Nokia AirScale mMIMO",
        enforcement_notes="Dynamic antenna switching based on traffic patterns"
    )
}

# =============================================================================
# HYBRID POLICIES (Nokia implementation of 3GPP standards)
# =============================================================================

HYBRID_POLICIES = {
    "NOKIA_3GPP_THROUGHPUT": Policy(
        policy_id="HYB_001_THROUGHPUT_3GPP_NOK",
        name="Nokia 3GPP Throughput Implementation",
        description="Nokia's implementation of 3GPP throughput requirements",
        standard=PolicyStandard.HYBRID,
        category=PolicyCategory.THROUGHPUT,
        severity=PolicySeverity.HIGH,
        min_value=100.0,   # Minimum 100 Mbps per user
        target_value=500.0, # Target 500 Mbps per user
        unit="Mbps",
        nokia_product="Nokia AirScale",
        standard_reference="3GPP TS 38.306",
        enforcement_notes="Nokia-optimized algorithms for 3GPP compliance"
    ),
    
    "NOKIA_3GPP_LATENCY_URLLC": Policy(
        policy_id="HYB_002_LATENCY_URLLC_NOK",
        name="Nokia URLLC Latency Optimization",
        description="Nokia's enhanced URLLC latency implementation",
        standard=PolicyStandard.HYBRID,
        category=PolicyCategory.LATENCY_PERFORMANCE,
        severity=PolicySeverity.CRITICAL,
        max_value=1.0,     # Maximum 1ms latency for URLLC
        target_value=0.5,  # Target 0.5ms
        unit="ms",
        nokia_product="Nokia AirScale Ultra Low Latency",
        standard_reference="3GPP TS 38.913",
        enforcement_notes="Nokia-specific optimizations for sub-1ms latency"
    )
}

# =============================================================================
# ENERGY EFFICIENCY SPECIFIC POLICIES
# =============================================================================

ENERGY_POLICIES = {
    "POWER_CONSUMPTION_LIMIT": Policy(
        policy_id="ENE_001_POWER_LIMIT",
        name="Base Station Power Consumption Limit",
        description="Maximum power consumption per base station",
        standard=PolicyStandard.NOKIA_PROPRIETARY,
        category=PolicyCategory.ENERGY_EFFICIENCY,
        severity=PolicySeverity.HIGH,
        max_value=2000.0,  # Maximum 2000W per BS
        target_value=1500.0, # Target 1500W
        unit="watts",
        nokia_product="Nokia AirScale Base Station",
        enforcement_notes="Dynamic power scaling based on traffic load"
    ),
    
    "ENERGY_SAVING_MODE": Policy(
        policy_id="ENE_002_ENERGY_SAVING",
        name="Energy Saving Mode Activation",
        description="Automatic energy saving during low traffic periods",
        standard=PolicyStandard.NOKIA_PROPRIETARY,
        category=PolicyCategory.ENERGY_EFFICIENCY,
        severity=PolicySeverity.MEDIUM,
        min_value=10.0,    # Minimum 10% energy saving
        target_value=30.0, # Target 30% energy saving
        unit="%",
        nokia_product="Nokia AirScale",
        enforcement_notes="Must maintain minimum service quality levels"
    ),
    
    "RENEWABLE_ENERGY_USAGE": Policy(
        policy_id="ENE_003_RENEWABLE",
        name="Renewable Energy Integration",
        description="Percentage of renewable energy usage requirement",
        standard=PolicyStandard.ITU,
        category=PolicyCategory.ENERGY_EFFICIENCY,
        severity=PolicySeverity.LOW,
        min_value=25.0,    # Minimum 25% renewable energy
        target_value=50.0, # Target 50% renewable
        unit="%",
        standard_reference="ITU-T L.1310",
        enforcement_notes="Gradual increase in renewable energy adoption"
    )
}

# =============================================================================
# ADDITIONAL 3GPP POLICIES FROM OPTIMIZATIONAGENT.PY
# =============================================================================

ADDITIONAL_3GPP_POLICIES = {
    "3GPP_TS_28541_NETWORK_SLICE_LATENCY": Policy(
        policy_id="3GPP_001_NS_LATENCY",
        name="Network Slice Latency Requirements",
        description="Network slice instances shall have maximum end-to-end latency < 10 ms for URLLC services",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.NETWORK_SLICING,
        severity=PolicySeverity.CRITICAL,
        max_value=10.0,  # 10ms max for URLLC
        target_value=5.0,  # Target 5ms
        unit="ms",
        standard_reference="3GPP TS 28.541",
        enforcement_notes="Critical for ultra-reliable low-latency communications"
    ),
    
    "3GPP_TS_22261_EMBB_DATA_RATE": Policy(
        policy_id="3GPP_002_EMBB_RATE",
        name="Enhanced Mobile Broadband Data Rate",
        description="Enhanced mobile broadband minimum user data rate shall be > 100 Mbps in dense urban areas",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.THROUGHPUT,
        severity=PolicySeverity.HIGH,
        min_value=100.0,   # Minimum 100 Mbps
        target_value=500.0, # Target 500 Mbps
        unit="Mbps",
        standard_reference="3GPP TS 22.261",
        enforcement_notes="Dense urban area deployment requirement"
    ),
    
    "3GPP_TS_38300_PACKET_LOSS": Policy(
        policy_id="3GPP_003_PKT_LOSS",
        name="Packet Loss Rate for eMBB",
        description="Packet loss rate for eMBB services must not exceed 0.1%",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.RELIABILITY,
        severity=PolicySeverity.HIGH,
        max_value=0.001,   # 0.1% max packet loss
        target_value=0.0001, # Target 0.01%
        unit="ratio",
        standard_reference="3GPP TS 38.300",
        enforcement_notes="Enhanced mobile broadband quality requirement"
    ),
    
    "3GPP_TS_28530_ENERGY_QOS": Policy(
        policy_id="3GPP_004_ENERGY_QOS",
        name="Energy Efficiency with QoS Maintenance",
        description="Energy efficiency optimization should maintain QoS constraints while reducing power consumption",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.ENERGY_EFFICIENCY,
        severity=PolicySeverity.HIGH,
        min_value=1000.0,  # Minimum energy efficiency
        target_value=2500.0, # Target efficiency
        unit="bytes/joule",
        standard_reference="3GPP TS 28.530",
        enforcement_notes="Balance between energy savings and service quality"
    ),
    
    "3GPP_TR_38901_MAX_TX_POWER": Policy(
        policy_id="3GPP_005_MAX_TX_PWR",
        name="Maximum Transmission Power",
        description="Maximum transmission power for gNB shall not exceed 43 dBm for macro cells",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.RADIO_RESOURCES,
        severity=PolicySeverity.CRITICAL,
        max_value=43.0,    # 43 dBm max
        target_value=30.0, # Target 30 dBm for efficiency
        unit="dBm",
        standard_reference="3GPP TR 38.901",
        enforcement_notes="Macro cell power limitation for interference control"
    )
}

# =============================================================================
# SERVICE-SPECIFIC POLICIES FROM OPTIMIZATIONAGENT.PY
# =============================================================================

SERVICE_SPECIFIC_POLICIES = {
    "AUTONOMOUS_VEHICLE_NETWORK_SLICE": Policy(
        policy_id="SVC_001_AUTO_VEHICLE",
        name="Autonomous Vehicle Network Slice",
        description="Network slice for autonomous vehicles requires latency < 5 ms and reliability > 99.999%",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.AUTONOMOUS_VEHICLE,
        severity=PolicySeverity.CRITICAL,
        max_value=5.0,     # 5ms max latency
        target_value=2.0,  # Target 2ms
        unit="ms",
        standard_reference="3GPP TS 22.261",
        enforcement_notes="Ultra-reliable communication for autonomous vehicles"
    ),
    
    "INDUSTRIAL_IOT_APPLICATIONS": Policy(
        policy_id="SVC_002_IND_IOT",
        name="Industrial IoT Applications",
        description="Industrial IoT applications need minimum throughput 1 Mbps with maximum latency 50 ms",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.IOT_SERVICES,
        severity=PolicySeverity.HIGH,
        min_value=1.0,     # Minimum 1 Mbps throughput
        max_value=50.0,    # Maximum 50ms latency (context-dependent)
        target_value=25.0, # Target 25ms latency
        unit="Mbps", # Note: This policy has dual constraints (throughput + latency)
        standard_reference="3GPP TS 22.261",
        enforcement_notes="Industrial automation requirements"
    ),
    
    "MASSIVE_IOT_CONNECTIONS": Policy(
        policy_id="SVC_003_MASSIVE_IOT",
        name="Massive IoT Connections",
        description="Massive IoT connections should have packet loss < 0.01% and support up to 1M devices per km²",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.IOT_SERVICES,
        severity=PolicySeverity.MEDIUM,
        max_value=0.0001,  # 0.01% max packet loss
        target_value=0.00001, # Target 0.001%
        unit="ratio",
        standard_reference="3GPP TS 22.261",
        enforcement_notes="Massive machine-type communications"
    ),
    
    "DENSE_URBAN_DEPLOYMENT": Policy(
        policy_id="SVC_004_DENSE_URBAN",
        name="Dense Urban Area Requirements",
        description="Dense urban deployments require enhanced capacity and coverage optimization",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.DENSE_DEPLOYMENT,
        severity=PolicySeverity.HIGH,
        min_value=200.0,   # Minimum 200 Mbps in dense areas
        target_value=1000.0, # Target 1 Gbps
        unit="Mbps",
        standard_reference="3GPP TS 22.261",
        enforcement_notes="High-capacity urban deployment requirements"
    ),
    
    "URLLC_GENERAL_SERVICES": Policy(
        policy_id="SVC_005_URLLC_GENERAL",
        name="URLLC Services General",
        description="Ultra-reliable low-latency communication services with strict performance requirements",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.LATENCY_PERFORMANCE,
        severity=PolicySeverity.CRITICAL,
        max_value=1.0,     # 1ms max latency
        target_value=0.5,  # Target 0.5ms
        unit="ms",
        standard_reference="3GPP TS 38.913",
        enforcement_notes="General URLLC service requirements"
    ),
    
    "EMBB_GENERAL_SERVICES": Policy(
        policy_id="SVC_006_EMBB_GENERAL",
        name="eMBB Services General",
        description="Enhanced mobile broadband services with high data rate requirements",
        standard=PolicyStandard.THREE_GPP,
        category=PolicyCategory.THROUGHPUT,
        severity=PolicySeverity.HIGH,
        min_value=100.0,   # Minimum 100 Mbps
        target_value=1000.0, # Target 1 Gbps
        unit="Mbps",
        standard_reference="3GPP TS 38.913",
        enforcement_notes="Enhanced mobile broadband specifications"
    )
}

# =============================================================================
# POLICY AGGREGATION AND MANAGEMENT
# =============================================================================

class PolicyManager:
    """Central policy management system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Aggregate all policies including new ones from OptimizationAgent.py
        self.all_policies = {}
        self.all_policies.update(QCI_5QI_POLICIES)
        self.all_policies.update(NOKIA_POLICIES)
        self.all_policies.update(HYBRID_POLICIES)
        self.all_policies.update(ENERGY_POLICIES)
        self.all_policies.update(ADDITIONAL_3GPP_POLICIES)  # NEW
        self.all_policies.update(SERVICE_SPECIFIC_POLICIES)  # NEW
        
        self.logger.info(f"Loaded {len(self.all_policies)} policies")
        self._log_policy_summary()
    
    def _log_policy_summary(self):
        """Log summary of loaded policies"""
        by_standard = {}
        by_severity = {}
        by_category = {}
        
        for policy in self.all_policies.values():
            # Count by standard
            std = policy.standard.value
            by_standard[std] = by_standard.get(std, 0) + 1
            
            # Count by severity
            sev = policy.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
            
            # Count by category
            cat = policy.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
        
        self.logger.info(f"Policies by standard: {by_standard}")
        self.logger.info(f"Policies by severity: {by_severity}")
        self.logger.info(f"Policies by category: {by_category}")
    
    def get_applicable_policies(self, bs_type: str, service_types: List[str]) -> Dict[str, Policy]:
        """Get policies applicable to specific BS and service types"""
        applicable = {}
        
        for policy_id, policy in self.all_policies.items():
            # Include all energy policies
            if policy.category == PolicyCategory.ENERGY_EFFICIENCY:
                applicable[policy_id] = policy
                continue
            
            # Include 5QI policies for active services
            if policy.category == PolicyCategory.QOS_5QI:
                if any(service in policy.description.lower() for service in service_types):
                    applicable[policy_id] = policy
                continue
            
            # Include service-specific policies
            service_categories = [
                PolicyCategory.AUTONOMOUS_VEHICLE,
                PolicyCategory.IOT_SERVICES,
                PolicyCategory.NETWORK_SLICING,
                PolicyCategory.DENSE_DEPLOYMENT
            ]
            if policy.category in service_categories:
                # Check if service type matches policy
                for service in service_types:
                    if (service.lower() in policy.description.lower() or
                        service.lower() in policy.name.lower()):
                        applicable[policy_id] = policy
                        break
            
            # Include Nokia policies for Nokia equipment
            if policy.standard == PolicyStandard.NOKIA_PROPRIETARY and "nokia" in bs_type.lower():
                applicable[policy_id] = policy
                continue
            
            # Include all other high/critical policies
            if policy.severity in [PolicySeverity.CRITICAL, PolicySeverity.HIGH]:
                applicable[policy_id] = policy
        
        return applicable
    
    def validate_kpi_values(self, bs_id: str, current_kpis: Dict[str, float], 
                          predicted_kpis: Dict[str, float], 
                          bs_type: str = "nokia_airscale",
                          active_services: List[str] = None) -> Dict[str, Any]:
        """Validate current and predicted KPI values against all applicable policies"""
        
        if active_services is None:
            active_services = ["voice", "data", "video"]  # Default services
        
        # Get applicable policies
        applicable_policies = self.get_applicable_policies(bs_type, active_services)
        
        validation_results = {}
        all_violations = []
        critical_violations = []
        
        for policy_id, policy in applicable_policies.items():
            # Map policy to KPI values
            current_value, predicted_value = self._map_policy_to_kpi(
                policy, current_kpis, predicted_kpis
            )
            
            if current_value is not None and predicted_value is not None:
                # Validate policy
                result = policy.validate(current_value, predicted_value)
                validation_results[policy_id] = result
                
                # Aggregate violations
                all_violations.extend(result['violations'])
                
                # Track critical violations that block optimization
                critical_viols = [v for v in result['violations'] 
                                if v['severity'] == 'critical' and v.get('blocks_optimization', False)]
                critical_violations.extend(critical_viols)
        
        return {
            'bs_id': bs_id,
            'total_policies_checked': len(applicable_policies),
            'policies_compliant': len([r for r in validation_results.values() if r['is_compliant']]),
            'total_violations': len(all_violations),
            'critical_violations': len(critical_violations),
            'blocks_optimization': len(critical_violations) > 0,
            'validation_results': validation_results,
            'all_violations': all_violations,
            'critical_violations': critical_violations,
            'applicable_policies': list(applicable_policies.keys())
        }
    
    def _map_policy_to_kpi(self, policy: Policy, current_kpis: Dict[str, float], 
                          predicted_kpis: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
        """Map policy to corresponding KPI values"""
        
        # Enhanced KPI mapping with new policy categories
        kpi_mappings = {
            # Latency mappings
            PolicyCategory.QOS_5QI: 'avg_latency',
            PolicyCategory.LATENCY_PERFORMANCE: 'avg_latency',
            PolicyCategory.NETWORK_SLICING: 'avg_latency',
            PolicyCategory.AUTONOMOUS_VEHICLE: 'avg_latency',
            
            # Energy mappings  
            PolicyCategory.ENERGY_EFFICIENCY: self._get_energy_kpi(policy),
            
            # Throughput mappings
            PolicyCategory.THROUGHPUT: 'avg_throughput',
            PolicyCategory.DENSE_DEPLOYMENT: 'avg_throughput',
            PolicyCategory.IOT_SERVICES: self._get_iot_kpi(policy),
            
            # Radio resource mappings
            PolicyCategory.RADIO_RESOURCES: 'prb_utilization',
            
            # Reliability mappings
            PolicyCategory.RELIABILITY: 'packet_loss_rate'
        }
        
        kpi_name = kpi_mappings.get(policy.category)
        
        if isinstance(kpi_name, str):
            current_value = current_kpis.get(kpi_name)
            predicted_value = predicted_kpis.get(kpi_name)
        else:
            # Handle complex mappings
            current_value = kpi_name
            predicted_value = kpi_name
        
        return current_value, predicted_value
    
    def _get_energy_kpi(self, policy: Policy) -> str:
        """Get appropriate energy KPI based on policy"""
        if "efficiency" in policy.name.lower():
            return 'energy_efficiency'
        elif "consumption" in policy.name.lower():
            return 'energy_consumption'
        elif "power" in policy.name.lower():
            return 'tx_power'
        else:
            return 'energy_efficiency'  # Default
    
    def _get_iot_kpi(self, policy: Policy) -> str:
        """Get appropriate IoT KPI based on policy"""
        if "throughput" in policy.description.lower():
            return 'avg_throughput'
        elif "latency" in policy.description.lower():
            return 'avg_latency'
        elif "packet loss" in policy.description.lower():
            return 'packet_loss_rate'
        else:
            return 'avg_throughput'  # Default for IoT
    
    def get_policies_by_service(self, service_type: str) -> List[Policy]:
        """Get all policies applicable to a specific service type"""
        applicable_policies = []
        
        for policy in self.all_policies.values():
            if (service_type.lower() in policy.description.lower() or
                service_type.lower() in policy.name.lower()):
                applicable_policies.append(policy)
        
        return applicable_policies
    
    def get_policies_by_standard(self, standard: PolicyStandard) -> List[Policy]:
        """Get all policies from a specific standard"""
        return [policy for policy in self.all_policies.values() 
                if policy.standard == standard]

# Create global policy manager instance
POLICY_MANAGER = PolicyManager()

# =============================================================================
# UTILITY FUNCTIONS FOR EASY ACCESS
# =============================================================================

def validate_energy_plan_policies(bs_id: str, current_kpis: Dict[str, float], 
                                predicted_kpis: Dict[str, float]) -> Dict[str, Any]:
    """Simplified function to validate energy plan against all policies"""
    return POLICY_MANAGER.validate_kpi_values(bs_id, current_kpis, predicted_kpis)

def get_critical_policy_violations(validation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract critical violations that would block energy optimization"""
    return validation_result.get('critical_violations', [])

def get_policy_summary() -> Dict[str, Any]:
    """Get summary of all loaded policies"""
    return {
        'total_policies': len(POLICY_MANAGER.all_policies),
        '3gpp_policies': len([p for p in POLICY_MANAGER.all_policies.values() 
                             if p.standard == PolicyStandard.THREE_GPP]),
        'nokia_policies': len([p for p in POLICY_MANAGER.all_policies.values() 
                              if p.standard == PolicyStandard.NOKIA_PROPRIETARY]),
        'hybrid_policies': len([p for p in POLICY_MANAGER.all_policies.values() 
                               if p.standard == PolicyStandard.HYBRID]),
        'critical_policies': len([p for p in POLICY_MANAGER.all_policies.values() 
                                 if p.severity == PolicySeverity.CRITICAL])
    }

def get_policies_by_service_type(service_type: str) -> List[Policy]:
    """Get policies applicable to a specific service type"""
    return POLICY_MANAGER.get_policies_by_service(service_type)

def get_policies_by_standard(standard_name: str) -> List[Policy]:
    """Get policies by standard name (string)"""
    standard_map = {
        'nokia': PolicyStandard.NOKIA_PROPRIETARY,
        '3gpp': PolicyStandard.THREE_GPP,
        'hybrid': PolicyStandard.HYBRID,
        'itu': PolicyStandard.ITU,
        'etsi': PolicyStandard.ETSI
    }
    
    standard = standard_map.get(standard_name.lower())
    if standard:
        return POLICY_MANAGER.get_policies_by_standard(standard)
    return []