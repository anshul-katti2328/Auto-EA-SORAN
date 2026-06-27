"""
Enhanced Multi-Objective Optimization Agent with 3GPP Policy Integration using FAISS
WITH ENHANCED ZEROMQ COMMUNICATION HANDLER
"""
import sys
import asyncio
import zmq
import zmq.asyncio
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from deap import base, creator, tools, algorithms
import json
import logging
import time
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import yaml
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import faiss
from dotenv import load_dotenv
import time
import random
os.makedirs(os.path.join(os.path.dirname(__file__), 'config'), exist_ok=True)
# Load environment variables from .env file
load_dotenv()

# Import the communication handler and message types
from optimizationCOMM import OptimizationCommunicationHandler, OptimizationMessageType, OptimizationMessage
from rule_based_controller import (
    RuleBasedDecision,
    select_rule_based_config,
)
# LLM Integration
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings('ignore')
# PyZMQ requires a selector-based asyncio loop on Windows.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

# Ensure symbols and log messages work in Windows terminals.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(
            encoding="utf-8",
            errors="replace",
        )
    except (AttributeError, OSError):
        pass
# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            'optimization_agent.log',
            encoding='utf-8',
        ),
        logging.StreamHandler()
    ]
)

class OptimizationObjective(Enum):
    MINIMIZE_ENERGY = "minimize_energy"
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    PARETO_OPTIMAL = "pareto_optimal"

@dataclass
class OptimizationRequest:
    """Data structure for optimization requests"""
    request_id: str
    timestamp: float
    bs_id: str
    current_config: Dict[str, Any]
    current_metrics: Dict[str, Any]
    predicted_energy: float
    confidence_score: float
    constraints: Dict[str, Any]
    objective: OptimizationObjective
    policy_text: Optional[str] = None # NEW: Policy text input

@dataclass
class OptimizedConfiguration:
    """Data structure for optimized configuration"""
    tx_power: float
    bandwidth: int
    carrier_count: int
    es_mode_config: List[int]
    predicted_energy: float
    predicted_qos_metrics: Dict[str, float]
    optimization_score: float
    constraint_violations: Dict[str, float]

@dataclass
class OptimizationResponse:
    """Data structure for optimization responses"""
    request_id: str
    timestamp: float
    bs_id: str
    original_config: Dict[str, Any]
    optimized_config: OptimizedConfiguration
    energy_saving_percent: float
    qos_impact_analysis: Dict[str, float]
    pareto_solutions: List[OptimizedConfiguration]
    optimization_method: str
    processing_time_ms: float
    explanation: str
    confidence: float
    execution_template: Dict[str, Any] = None
    applied_policies: List[str] = None # NEW: Applied policy texts

class SimpleEmbeddingModel:
    """Simple count-based text embedding for FAISS"""
    def __init__(self):
        self.vocab = {}
        self.dim = 0

    def fit(self, texts: List[str]):
        words = set()
        for text in texts:
            words.update(text.lower().split())
        self.vocab = {word: idx for idx, word in enumerate(sorted(words))}
        self.dim = len(self.vocab)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.vocab:
            self.fit(texts)
        
        embeddings = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for word in text.lower().split():
                if word in self.vocab:
                    vec[self.vocab[word]] += 1
            
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec)
        
        return np.array(embeddings)

class PolicyManager:
    """
    3GPP Policy Manager with FAISS-based similarity search
    Parses policy text to extract QoS thresholds dynamically
    """
    def __init__(self, policy_texts: List[str]):
        self.logger = logging.getLogger('PolicyManager')
        self.policy_texts = policy_texts
        self.embedder = SimpleEmbeddingModel()
        
        # Build FAISS index
        self.embedder.fit(policy_texts)
        self.embeddings = self.embedder.encode(policy_texts)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        
        self.logger.info(f"PolicyManager initialized with {len(policy_texts)} policies")

    def query_policies(self, query_text: str, top_k: int = 5) -> List[str]:
        """Query FAISS index for most similar policies"""
        query_embedding = self.embedder.encode([query_text])
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.policy_texts):
                results.append(self.policy_texts[idx])
        
        self.logger.debug(f"Policy query '{query_text}' matched: {results}")
        return results

    def parse_constraints_from_policies(self, policies: List[str]) -> Dict[str, float]:
        """Extract QoS constraints from policy texts using regex"""
        constraints = {}
        
        for policy in policies:
            # Extract latency constraints
            latency_patterns = [
                r"latency.*?(\d+(?:\.\d+)?)\s*ms",
                r"delay.*?(\d+(?:\.\d+)?)\s*ms",
                r"response time.*?(\d+(?:\.\d+)?)\s*ms"
            ]
            for pattern in latency_patterns:
                match = re.search(pattern, policy, re.I)
                if match:
                    val = float(match.group(1))
                    constraints['max_latency'] = min(constraints.get('max_latency', float('inf')), val)
                    break
            
            # Extract throughput constraints
            throughput_patterns = [
                r"throughput.*?(\d+(?:\.\d+)?)\s*mbps",
                r"bandwidth.*?(\d+(?:\.\d+)?)\s*mbps",
                r"data rate.*?(\d+(?:\.\d+)?)\s*mbps"
            ]
            for pattern in throughput_patterns:
                match = re.search(pattern, policy, re.I)
                if match:
                    val = float(match.group(1))
                    constraints['min_throughput'] = max(constraints.get('min_throughput', 0), val)
                    break
            
            # Extract packet loss constraints
            packet_loss_patterns = [
                r"packet loss.*?(\d+(?:\.\d+)?)\s*%",
                r"loss rate.*?(\d+(?:\.\d+)?)\s*%",
                r"error rate.*?(\d+(?:\.\d+)?)\s*%"
            ]
            for pattern in packet_loss_patterns:
                match = re.search(pattern, policy, re.I)
                if match:
                    val = float(match.group(1))
                    constraints['max_packet_loss'] = min(constraints.get('max_packet_loss', float('inf')), val / 100.0)
                    break
            
            # Extract power constraints
            power_patterns = [
                r"(?:tx|transmit|transmission)\s*power.*?(\d+(?:\.\d+)?)\s*dbm",
                r"power.*?(\d+(?:\.\d+)?)\s*dbm"
            ]
            for pattern in power_patterns:
                match = re.search(pattern, policy, re.I)
                if match:
                    val = float(match.group(1))
                    constraints['max_tx_power'] = min(constraints.get('max_tx_power', float('inf')), val)
                    break
        
        self.logger.info(f"Extracted constraints from policies: {constraints}")
        return constraints

    def update_constraints_with_policy(self, policy_text: str, base_constraints: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Update base constraints with policy-derived constraints
        
        Returns:
            Tuple of (updated_constraints, applied_policies)
        """
        # Query similar policies
        matched_policies = self.query_policies(policy_text, top_k=5)
        
        # Parse constraints from matched policies
        policy_constraints = self.parse_constraints_from_policies(matched_policies)
        
        # Update base constraints
        updated_constraints = base_constraints.copy()
        applied_policies = []
        
        for constraint_key, policy_value in policy_constraints.items():
            if constraint_key in ['max_latency', 'max_packet_loss', 'max_tx_power']:
                # For maximum constraints, take the more restrictive (minimum) value
                current_value = updated_constraints.get(constraint_key, float('inf'))
                if policy_value < current_value:
                    updated_constraints[constraint_key] = policy_value
                    applied_policies.extend(matched_policies)
            elif constraint_key in ['min_throughput']:
                # For minimum constraints, take the more restrictive (maximum) value
                current_value = updated_constraints.get(constraint_key, 0)
                if policy_value > current_value:
                    updated_constraints[constraint_key] = policy_value
                    applied_policies.extend(matched_policies)
        
        # Remove duplicates from applied policies
        applied_policies = list(set(applied_policies))
        
        self.logger.info(f"Updated constraints: {updated_constraints}")
        self.logger.info(f"Applied {len(applied_policies)} policies")
        
        return updated_constraints, applied_policies

class LLMTemplateGenerator:
    """LLM-powered template generator for optimization execution plans"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.logger = logging.getLogger('LLMTemplateGenerator')
        self.config = llm_config
        self.llm_client = None
        self.openai_model = str(
            self.config.get(
                "model",
                "gpt-5-mini-2025-08-07",
            )
        )
        self.initialize_llm()

    def initialize_llm(self):
        """Initialize LLM client based on configuration"""
        try:
            llm_type = self.config.get('type', 'openai')
            if llm_type == "openai":
                api_key = os.getenv(
                    "OPENAI_API_KEY",
                    self.config.get("api_key", ""),
                )

                if not api_key:
                    self.logger.warning(
                        "OPENAI_API_KEY is not configured; "
                        "LLM healing will be unavailable"
                    )
                    self.llm_client = None
                    return

                self.llm_client = OpenAI(
                    api_key=api_key,
                    timeout=float(
                        self.config.get(
                            "timeout_seconds",
                            30,
                        )
                    ),
                    max_retries=int(
                        self.config.get(
                            "retry_attempts",
                            3,
                        )
                    ),
                )
            elif llm_type == 'huggingface':
                model_name = self.config.get('model_name', 'microsoft/DialoGPT-medium')
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.llm_client = pipeline('text-generation',
                                         model=self.model,
                                         tokenizer=self.tokenizer)
            elif llm_type == 'local':
                # For local LLM deployment
                self.llm_client = pipeline('text-generation',
                                         model=self.config.get('model_path', 'gpt2'))
            self.logger.info(f"LLM client initialized: {llm_type}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None

    def generate_execution_template(self, request: OptimizationRequest,
                                  solution: OptimizedConfiguration,
                                  energy_saving: float) -> Dict[str, Any]:
        """Generate detailed execution template with LLM assistance"""
        
        base_template = {
            "execution_id": f"OPT_{request.bs_id}_{int(time.time())}",
            "target_base_station": request.bs_id,
            "optimization_timestamp": datetime.now().isoformat(),
            "policy_context": request.policy_text if hasattr(request, 'policy_text') else None, # NEW
            "configuration_changes": {
                "current": request.current_config,
                "optimized": {
                    "tx_power": solution.tx_power,
                    "bandwidth": solution.bandwidth,
                    "carrier_count": solution.carrier_count,
                    "es_modes": solution.es_mode_config
                }
            },
            "expected_outcomes": {
                "energy_savings_watts": request.predicted_energy - solution.predicted_energy,
                "energy_savings_percent": energy_saving,
                "qos_metrics": solution.predicted_qos_metrics,
                "constraint_status": "COMPLIANT" if not solution.constraint_violations else "VIOLATIONS_DETECTED"
            },
            "execution_plan": {},
            "monitoring_strategy": {},
            "rollback_plan": {},
            "llm_insights": ""
        }

        # Generate LLM insights if available
        if (
            self.config.get(
                "execution_insights_enabled",
                False,
            )
            and self.llm_client is not None
        ):
            try:
                llm_response = (
                    self._generate_llm_insights(
                        request,
                        solution,
                        energy_saving,
                    )
                )
                base_template["llm_insights"] = (
                    llm_response
                )
                base_template["execution_plan"] = (
                    self._extract_execution_plan(
                        llm_response,
                        request,
                        solution,
                    )
                )
            except Exception as error:
                self.logger.error(
                    "LLM generation failed: %s",
                    error,
                )
                base_template["llm_insights"] = (
                    "LLM template generation unavailable"
                )
        else:
            base_template["llm_insights"] = (
                "Per-cell LLM insights disabled; "
                "LLM reserved for feedback-triggered healing"
            )

        # Add structured execution details
        base_template["execution_plan"].update({
            "pre_execution_checks": [
                f"Verify base station {request.bs_id} operational status",
                "Confirm no active emergency traffic sessions",
                "Validate current QoS metrics within acceptable range",
                f"Check energy baseline: {request.predicted_energy:.1f}W"
            ],
            "phase_1_energy_modes": {
                "duration_minutes": 5,
                "actions": [f"Activate ES Mode {i}" for i, mode in enumerate(solution.es_mode_config) if mode == 1],
                "monitoring": ["energy_consumption", "active_connections", "signal_quality"]
            },
            "phase_2_power_adjustment": {
                "duration_minutes": 5,
                "current_tx_power": request.current_config.get('tx_power', 20),
                "target_tx_power": solution.tx_power,
                "adjustment_steps": max(1, abs(solution.tx_power - request.current_config.get('tx_power', 20)) // 2),
                "monitoring": ["sinr", "throughput", "packet_loss"]
            },
            "phase_3_bandwidth_optimization": {
                "duration_minutes": 5,
                "current_bandwidth": request.current_config.get('bandwidth', 20),
                "target_bandwidth": solution.bandwidth,
                "resource_reallocation": solution.bandwidth != request.current_config.get('bandwidth', 20)
            }
        })

        # Monitoring strategy
        base_template["monitoring_strategy"] = {
            "monitoring_period_minutes": 30,
            "key_metrics": ["energy_consumption", "throughput", "latency", "packet_loss", "user_satisfaction"],
            "alert_thresholds": {
                "energy_increase": f"> {request.predicted_energy * 1.05:.1f}W",
                "throughput_degradation": f"< {solution.predicted_qos_metrics.get('throughput', 50) * 0.95:.1f} Mbps",
                "latency_increase": f"> {solution.predicted_qos_metrics.get('latency', 20) * 1.1:.1f} ms"
            }
        }

        # Rollback plan
        base_template["rollback_plan"] = {
            "trigger_conditions": [
                "Energy consumption increases > 10% from baseline",
                "QoS constraint violation detected",
                "User complaints > 5% increase",
                "Network instability detected"
            ],
            "rollback_steps": [
                "Immediately restore TX power to previous level",
                "Deactivate new energy saving modes",
                "Restore bandwidth allocation",
                "Monitor for 15 minutes for stabilization"
            ],
            "estimated_rollback_time": "5 minutes"
        }

        return base_template

    def _generate_llm_insights(self, request: OptimizationRequest,
                             solution: OptimizedConfiguration,
                             energy_saving: float) -> str:
        """Generate LLM insights for the optimization"""
        prompt = self._create_optimization_prompt(request, solution, energy_saving)
        
        try:
            if self.llm_client is None:
                return "LLM client not available"
            
            if self.config.get("type") == "openai":
                response = (
                    self.llm_client.responses.create(
                        model=self.openai_model,
                        instructions=(
                            "You are a 5G network optimization "
                            "expert. Provide detailed technical "
                            "execution guidance for base-station "
                            "energy optimization."
                        ),
                        input=prompt,
                        max_output_tokens=int(
                            self.config.get(
                                "max_output_tokens",
                                1500,
                            )
                        ),
                    )
                )
                return response.output_text
            else:
                response = self.llm_client(prompt,
                                         max_length=1000,
                                         num_return_sequences=1,
                                         temperature=0.3)
                return response[0]['generated_text']
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            return f"LLM generation failed: {str(e)}"

    def _create_optimization_prompt(self, request: OptimizationRequest,
                                  solution: OptimizedConfiguration,
                                  energy_saving: float) -> str:
        """Create detailed prompt for LLM"""
        prompt = f"""
5G Base Station Energy Optimization Execution Plan

SCENARIO:
Base Station ID: {request.bs_id}
Current Energy Consumption: {request.predicted_energy:.2f}W
Target Energy Consumption: {solution.predicted_energy:.2f}W
Expected Energy Savings: {energy_saving:.1f}% ({solution.predicted_energy - request.predicted_energy:.2f}W)

POLICY CONTEXT: {getattr(request, 'policy_text', 'No specific policy provided')}

CURRENT CONFIGURATION:
- TX Power: {request.current_config.get('tx_power', 'N/A')} dBm
- Bandwidth: {request.current_config.get('bandwidth', 'N/A')} MHz
- ES Modes: {request.current_config.get('es_modes', 'N/A')}

OPTIMIZED CONFIGURATION:
- TX Power: {solution.tx_power} dBm
- Bandwidth: {solution.bandwidth} MHz
- ES Modes: {solution.es_mode_config}

PREDICTED QOS IMPACT:
- Throughput: {solution.predicted_qos_metrics.get('throughput', 'N/A')} Mbps
- Latency: {solution.predicted_qos_metrics.get('latency', 'N/A')} ms
- Packet Loss: {solution.predicted_qos_metrics.get('packet_loss', 'N/A')}

CONSTRAINTS:
{json.dumps(request.constraints, indent=2)}

Please provide:
1. Detailed step-by-step execution guidance
2. Potential risks and mitigation strategies
3. Critical monitoring points during transition
4. Success criteria for each phase
5. Specific technical recommendations for this optimization

Focus on practical implementation details that a network engineer would need.
"""
        return prompt

    def _extract_execution_plan(self, llm_response: str,
                              request: OptimizationRequest,
                              solution: OptimizedConfiguration) -> Dict[str, Any]:
        """Extract structured execution plan from LLM response"""
        execution_plan = {
            "llm_generated_steps": llm_response,
            "risk_assessment": self._extract_risks(llm_response),
            "success_criteria": self._extract_success_criteria(llm_response),
            "technical_recommendations": self._extract_recommendations(llm_response)
        }
        return execution_plan

    def _extract_risks(self, text: str) -> List[str]:
        """Extract risk information from LLM response"""
        risks = []
        lines = text.split('\n')
        in_risk_section = False
        
        for line in lines:
            if 'risk' in line.lower() or 'potential' in line.lower():
                in_risk_section = True
            elif in_risk_section and line.strip() and line.startswith(('•', '-', '*', '1.', '2.')):
                risks.append(line.strip())
            elif in_risk_section and not line.strip():
                in_risk_section = False
        
        return risks[:5] # Limit to top 5 risks

    def _extract_success_criteria(self, text: str) -> List[str]:
        """Extract success criteria from LLM response"""
        criteria = []
        lines = text.split('\n')
        in_success_section = False
        
        for line in lines:
            if 'success' in line.lower() or 'criteria' in line.lower():
                in_success_section = True
            elif in_success_section and line.strip() and line.startswith(('•', '-', '*', '1.', '2.')):
                criteria.append(line.strip())
            elif in_success_section and not line.strip():
                in_success_section = False
        
        return criteria[:5] # Limit to top 5 criteria

    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract technical recommendations from LLM response"""
        recommendations = []
        lines = text.split('\n')
        in_rec_section = False
        
        for line in lines:
            if 'recommend' in line.lower() or 'technical' in line.lower():
                in_rec_section = True
            elif in_rec_section and line.strip() and line.startswith(('•', '-', '*', '1.', '2.')):
                recommendations.append(line.strip())
            elif in_rec_section and not line.strip():
                in_rec_section = False
        
        return recommendations[:5] # Limit to top 5 recommendations

class OrchestratorInterface:
    """Interface for communicating with OMNeT++ Orchestrator Agent"""
    
    def __init__(self, orchestrator_config: Dict[str, Any]):
        self.logger = logging.getLogger('OrchestratorInterface')
        self.config = orchestrator_config
        
        # [OK] CORRECT INITIALIZATION
        self.host = orchestrator_config.get('orchestrator_host', 'localhost')
        self.port = orchestrator_config.get('orchestrator_port', 5563)
        
        print(f"[DEBUG] OrchestratorInterface: Connecting to {self.host}:{self.port}")
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        
        connection_string = f"tcp://{self.host}:{self.port}"
        print(f"[DEBUG] Connection string: {connection_string}")
        
        self.socket.connect(connection_string)
        
        # Set timeouts
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second timeout
        
        print(f"[DEBUG] Connected to OrchestrationAgent successfully!")
        
        self.logger.info(f"Connected to OrchestrationAgent at {self.host}:{self.port}")


    def initialize_connection(self):
        """Initialize connection to Orchestrator Agent"""
        try:
            orchestrator_port = self.config.get('orchestrator_port', 5555)
            orchestrator_host = self.config.get('orchestrator_host', 'localhost')
            
            self.orchestrator_socket = self.context.socket(zmq.REQ)
            self.orchestrator_socket.connect(f"tcp://{orchestrator_host}:{orchestrator_port}")
            
            # Set timeouts
            self.orchestrator_socket.setsockopt(zmq.RCVTIMEO, 5000) # 5 second receive timeout
            self.orchestrator_socket.setsockopt(zmq.SNDTIMEO, 5000) # 5 second send timeout
            
            self.logger.info(f"Connected to Orchestrator at {orchestrator_host}:{orchestrator_port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Orchestrator: {e}")
            raise

    def _reset_request_socket(self):
        """Recreate the REQ socket after timeout or state failure."""

        try:
            if getattr(self, "socket", None) is not None:
                self.socket.close(linger=0)

            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)
            self.socket.setsockopt(zmq.SNDTIMEO, 5000)

            self.socket.connect(
                f"tcp://{self.host}:{self.port}"
            )

            self.logger.info(
                "Orchestration REQ socket reset successfully"
            )

        except Exception as error:
            self.logger.error(
                "Failed to reset Orchestration REQ socket: %s",
                error,
            )

    def send_command(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to OrchestrationAgent and get response"""
        try:
            print(f"[DEBUG] OrchestratorInterface sending: {command_dict.get('action', 'unknown')}")
            
            # Send command
            message = json.dumps(command_dict)
            self.socket.send_string(message)
            
            # Wait for response
            response_str = self.socket.recv_string()
            
            print(f"[DEBUG] OrchestratorInterface received: {response_str[:100]}...")
            
            response = json.loads(response_str)
            return response
            
        except zmq.Again:
            self.logger.error(
                "OrchestrationAgent command timeout"
            )
            self._reset_request_socket()
            return {
                "status": "TIMEOUT",
                "detail": "Command timeout",
            }

        except Exception as error:
            self.logger.error(
                "Orchestrator command error: %s",
                error,
            )
            self._reset_request_socket()
            return {
                "status": "ERROR",
                "detail": str(error),
            }

    def execute_optimization_plan(self, bs_id: str, optimized_config: OptimizedConfiguration) -> Dict[str, Any]:
        """Execute complete optimization plan through orchestrator"""
        execution_results = {
            'status': 'SUCCESS',
            'executed_actions': [],
            'failed_actions': [],
            'execution_timeline': []
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Apply Energy Saving Modes
            self.logger.info(f"Executing energy saving modes for {bs_id}")
            es_results = self._apply_energy_saving_modes(bs_id, optimized_config)
            execution_results['executed_actions'].extend(es_results['success'])
            execution_results['failed_actions'].extend(es_results['failed'])

            # Step 2: Adjust TX Power
            self.logger.info(f"Adjusting TX power for {bs_id}")
            power_results = self._apply_tx_power(bs_id, optimized_config)
            execution_results['executed_actions'].extend(power_results['success'])
            execution_results['failed_actions'].extend(power_results['failed'])

            # Step 3: Configure Bandwidth and Carriers
            self.logger.info(f"Configuring bandwidth and carriers for {bs_id}")
            bw_results = self._apply_bandwidth_carriers(bs_id, optimized_config)
            execution_results['executed_actions'].extend(bw_results['success'])
            execution_results['failed_actions'].extend(bw_results['failed'])

            # Determine overall status
            if execution_results['failed_actions']:
                if execution_results['executed_actions']:
                    execution_results['status'] = 'PARTIAL_SUCCESS'
                else:
                    execution_results['status'] = 'FAILED'
            
            execution_results['execution_time_seconds'] = time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"Optimization execution failed for {bs_id}: {e}")
            execution_results['status'] = 'ERROR'
            execution_results['error'] = str(e)
            
        return execution_results

    def _apply_energy_saving_modes(self, bs_id: str, config: OptimizedConfiguration) -> Dict[str, List]:
        """Apply energy saving mode configurations"""
        results = {'success': [], 'failed': []}
        
        es_mode_names = ['deep_sleep', 'light_sleep', 'mimo_reduction', 'dtx_mode', 'bwp_adaptation', 'rf_chain_sleep']
        
        for i, mode_enabled in enumerate(config.es_mode_config):
            if mode_enabled and i < len(es_mode_names):
                command = {
                    "action": "enterSleepMode",
                    "target": f"gnb[{bs_id}]",
                    "param": {"mode": es_mode_names[i]}
                }
                
                response = self.send_command(command)
                if response.get('status') == 'OK':
                    results['success'].append(f"ES Mode {i}: {es_mode_names[i]}")
                    time.sleep(0.5) # Small delay between ES mode changes
                else:
                    results['failed'].append(f"ES Mode {i}: {response.get('detail', 'Unknown error')}")
        
        return results

    def _apply_tx_power(self, bs_id: str, config: OptimizedConfiguration) -> Dict[str, List]:
        """Apply TX power configuration"""
        results = {'success': [], 'failed': []}
        
        command = {
            "action": "setTxPower",
            "target": f"gnb[{bs_id}]",
            "param": {"power": config.tx_power}
        }
        
        response = self.send_command(command)
        if response.get('status') == 'OK':
            results['success'].append(f"TX Power: {config.tx_power} dBm")
        else:
            results['failed'].append(f"TX Power: {response.get('detail', 'Unknown error')}")
        
        return results

    def _apply_bandwidth_carriers(self, bs_id: str, config: OptimizedConfiguration) -> Dict[str, List]:
        """Apply bandwidth and carrier configurations"""
        results = {'success': [], 'failed': []}
        
        # Set bandwidth
        bw_command = {
            "action": "setBandwidth",
            "target": f"gnb[{bs_id}]",
            "param": {"bandwidth": config.bandwidth}
        }
        
        response = self.send_command(bw_command)
        if response.get('status') == 'OK':
            results['success'].append(f"Bandwidth: {config.bandwidth} MHz")
        else:
            results['failed'].append(f"Bandwidth: {response.get('detail', 'Unknown error')}")
        
        # Set carrier count
        carrier_command = {
            "action": "setCarrierCount",
            "target": f"gnb[{bs_id}]",
            "param": {"count": config.carrier_count}
        }
        
        response = self.send_command(carrier_command)
        if response.get('status') == 'OK':
            results['success'].append(f"Carrier Count: {config.carrier_count}")
        else:
            results['failed'].append(f"Carrier Count: {response.get('detail', 'Unknown error')}")
        
        return results

    def get_current_metrics(self, bs_id: str) -> Dict[str, Any]:
        """Get current metrics from the base station"""
        command = {
            "action": "getMetrics",
            "target": f"gnb[{bs_id}]",
            "param": {}
        }
        
        response = self.send_command(command)
        if response.get('status') == 'OK':
            return response.get('metrics', {})
        else:
            self.logger.error(f"Failed to get metrics: {response.get('detail', 'Unknown error')}")
            return {}

    def cleanup(self):
        """Cleanup ZeroMQ resources"""
        if self.orchestrator_socket:
            self.orchestrator_socket.close()
        self.context.term()

class QoSPredictor:
    """Enhanced QoS prediction model for network performance estimation"""
    
    def __init__(self):
        self.logger = logging.getLogger('QoSPredictor')
        self.models_initialized = True

    def predict_throughput(self, tx_power: float, bandwidth: int,
                          load: float, sinr_estimate: float) -> float:
        """Predict user throughput based on configuration"""
        # Enhanced throughput model based on Shannon capacity
        base_throughput = bandwidth * np.log2(1 + sinr_estimate)
        
        # Load impact with more realistic modeling
        load_factor = max(0.1, 1 - load * 0.7) # More gradual degradation
        
        # Power factor with realistic scaling
        power_factor = min(1.0, max(0.3, tx_power / 23.0)) # 23dBm reference
        
        # MIMO and scheduling efficiency
        scheduling_efficiency = 0.75 # Typical 5G efficiency
        
        return base_throughput * load_factor * power_factor * scheduling_efficiency

    def predict_latency(self, tx_power: float, load: float,
                       es_modes: List[int]) -> float:
        """Predict end-to-end latency with enhanced modeling"""
        # Base processing latency
        base_latency = 3.0 # ms for 5G
        
        # Load impact with exponential increase
        load_penalty = load * load * 8.0 # Quadratic penalty
        
        # Power impact (retransmissions)
        power_penalty = max(0, (20 - tx_power) * 0.3)
        
        # Energy saving mode penalties
        es_mode_penalties = {
            0: 1.5, # Sleep mode
            1: 0.5, # Carrier aggregation
            2: 1.0, # MIMO reduction
            3: 2.0, # Micro-DTX
            4: 0.8, # BWP adaptation
            5: 1.2  # RF chain sleep
        }
        
        es_penalty = sum(es_modes[i] * es_mode_penalties[i] for i in range(len(es_modes)))
        
        return base_latency + load_penalty + power_penalty + es_penalty

    def predict_packet_loss(self, tx_power: float, sinr_estimate: float,
                           load: float) -> float:
        """Predict packet loss rate with realistic modeling"""
        # Base packet loss for 5G
        base_loss = 0.0001 # Very low for 5G
        
        # SINR impact with threshold behavior
        if sinr_estimate < 5:
            sinr_penalty = (5 - sinr_estimate) * 0.01
        else:
            sinr_penalty = max(0, (15 - sinr_estimate) * 0.001)
        
        # Load impact with congestion modeling
        if load > 0.8:
            load_penalty = (load - 0.8) * 0.05
        else:
            load_penalty = load * 0.002
        
        # Power impact
        power_penalty = max(0, (15 - tx_power) * 0.0005)
        
        return min(0.1, base_loss + sinr_penalty + load_penalty + power_penalty)

    def estimate_sinr(self, tx_power: float, distance: float = 100.0,
                     frequency: float = 3.5) -> float:
        """Estimate SINR with enhanced path loss model"""
        # 5G path loss model
        path_loss_db = 32.4 + 20 * np.log10(distance) + 20 * np.log10(frequency) # GHz
        
        # Shadow fading (typical urban)
        shadow_fading = np.random.normal(0, 8) # 8dB standard deviation
        
        received_power = tx_power - path_loss_db + shadow_fading
        
        # Noise and interference
        noise_power = -104 # dBm for 5G
        interference_power = -100 + np.random.normal(0, 3) # Variable interference
        
        total_noise = 10 * np.log10(10**(noise_power/10) + 10**(interference_power/10))
        
        sinr = received_power - total_noise
        
        return max(0, sinr)

class MultiObjectiveOptimizer:
    """Advanced multi-objective optimizer using NSGA-II genetic algorithm"""
    
    def __init__(
        self,
        qos_predictor: QoSPredictor,
        constraints: Dict[str, Any],
        bounds_config: Optional[Dict[str, Any]] = None,
    ):
        self.qos_predictor = qos_predictor
        self.constraints = constraints
        self.logger = logging.getLogger(
            "MultiObjectiveOptimizer"
        )

        bounds_config = bounds_config or {}

        tx_power_range = bounds_config.get(
            "tx_power_range",
            [5.0, 43.0],
        )

        if (
            not isinstance(tx_power_range, (list, tuple))
            or len(tx_power_range) != 2
        ):
            tx_power_range = [5.0, 43.0]

        minimum_tx_power = float(tx_power_range[0])
        maximum_tx_power = float(tx_power_range[1])

        if minimum_tx_power > maximum_tx_power:
            minimum_tx_power, maximum_tx_power = (
                maximum_tx_power,
                minimum_tx_power,
            )

        bandwidth_options = [
            int(value)
            for value in bounds_config.get(
                "bandwidth_options",
                [20],
            )
            if int(value) > 0
        ]

        if not bandwidth_options:
            bandwidth_options = [20]

        carrier_options = [
            int(value)
            for value in bounds_config.get(
                "carrier_count_range",
                [1],
            )
            if 1 <= int(value) <= 4
        ]

        if not carrier_options:
            carrier_options = [1]

        number_of_es_modes = max(
            1,
            int(
                bounds_config.get(
                    "energy_saving_modes",
                    6,
                )
            ),
        )

        self.bounds = {
            "tx_power": (
                minimum_tx_power,
                maximum_tx_power,
            ),
            "bandwidth": bandwidth_options,
            "carrier_count": carrier_options,
            "es_modes": [
                (0, 1)
            ] * number_of_es_modes,
        }

        if "max_tx_power" in constraints:
            self.bounds["tx_power"] = (
                self.bounds["tx_power"][0],
                min(
                    self.bounds["tx_power"][1],
                    float(
                        constraints["max_tx_power"]
                    ),
                ),
            )

        self.logger.info(
            "Physical optimization bounds: %s",
            self.bounds,
        )

        self.setup_genetic_algorithm()

    def setup_genetic_algorithm(self):
        """Configure NSGA-II with mutually exclusive ES modes."""

        if not hasattr(creator, "FitnessMin"):
            creator.create(
                "FitnessMin",
                base.Fitness,
                weights=(-1.0, -1.0, -1.0),
            )

        if not hasattr(creator, "Individual"):
            creator.create(
                "Individual",
                list,
                fitness=creator.FitnessMin,
            )

        self.toolbox = base.Toolbox()

        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self._create_individual,
        )

        self.toolbox.register(
            "mate",
            self.constrained_mate,
        )

        self.toolbox.register(
            "mutate",
            self.adaptive_mutate,
        )

        self.toolbox.register(
            "select",
            tools.selNSGA2,
        )

        self.toolbox.register(
            "evaluate",
            self.evaluate_individual,
        )

    def _create_individual(self):
        """Create one active ES mode, or fully active mode."""

        tx_power = np.random.uniform(
            self.bounds["tx_power"][0],
            self.bounds["tx_power"][1],
        )

        bandwidth_index = np.random.randint(
            0,
            len(self.bounds["bandwidth"]),
        )

        carrier_index = np.random.randint(
            0,
            len(self.bounds["carrier_count"]),
        )

        number_of_modes = len(
            self.bounds["es_modes"]
        )

        # -1 represents no ES mode: fully active operation.
        selected_mode = np.random.randint(
            -1,
            number_of_modes,
        )

        es_flags = [0] * number_of_modes

        if selected_mode >= 0:
            es_flags[selected_mode] = 1

        return creator.Individual(
            [
                tx_power,
                bandwidth_index,
                carrier_index,
                *es_flags,
            ]
        )

    def repair_es_modes(self, individual):
        """Ensure no individual enables incompatible ES modes."""

        active_indices = [
            index
            for index in range(3, len(individual))
            if int(individual[index]) != 0
        ]

        if len(active_indices) <= 1:
            return individual

        selected_index = int(
            np.random.choice(active_indices)
        )

        for index in range(3, len(individual)):
            individual[index] = (
                1 if index == selected_index else 0
            )

        return individual

    def constrained_mate(
        self,
        first_individual,
        second_individual,
    ):
        """Apply crossover and restore valid ES configurations."""

        tools.cxTwoPoint(
            first_individual,
            second_individual,
        )

        self.repair_es_modes(first_individual)
        self.repair_es_modes(second_individual)

        return first_individual, second_individual

    def adaptive_mutate(
        self,
        individual,
        generation=0,
    ):
        """Apply adaptive mutation while preserving ES validity."""

        mutation_rate = max(
            0.05,
            0.3 - generation * 0.002,
        )

        if np.random.random() < mutation_rate:
            step_size = max(
                0.5,
                3.0 - generation * 0.02,
            )

            individual[0] = np.clip(
                individual[0] +
                np.random.normal(0, step_size),
                self.bounds["tx_power"][0],
                self.bounds["tx_power"][1],
            )

        if np.random.random() < mutation_rate:
            individual[1] = np.random.randint(
                0,
                len(self.bounds["bandwidth"]),
            )

        if np.random.random() < mutation_rate:
            individual[2] = np.random.randint(
                0,
                len(self.bounds["carrier_count"]),
            )

        if np.random.random() < mutation_rate:
            number_of_modes = len(
                self.bounds["es_modes"]
            )

            selected_mode = np.random.randint(
                -1,
                number_of_modes,
            )

            for index in range(3, len(individual)):
                individual[index] = 0

            if selected_mode >= 0:
                individual[3 + selected_mode] = 1

        self.repair_es_modes(individual)

        return individual,

    def decode_individual(
        self,
        individual: List,
    ) -> Dict[str, Any]:
        """Decode an individual into a valid configuration."""

        es_modes = [
            1 if int(value) != 0 else 0
            for value in individual[3:]
        ]

        active_indices = [
            index
            for index, value in enumerate(es_modes)
            if value == 1
        ]

        if len(active_indices) > 1:
            selected_index = active_indices[0]
            es_modes = [0] * len(es_modes)
            es_modes[selected_index] = 1

        return {
            "tx_power": float(individual[0]),
            "bandwidth": self.bounds["bandwidth"][
                int(individual[1])
            ],
            "carrier_count": self.bounds[
                "carrier_count"
            ][int(individual[2])],
            "es_modes": es_modes,
        }
    @staticmethod
    def es_power_factor(
        es_modes: List[int],
        load: float = 0.0,
    ) -> float:
        """Match the physical TwinMetricsAgent ES energy model."""

        load = float(np.clip(load, 0.0, 1.0))

        enabled_indices = [
            index
            for index, enabled in enumerate(es_modes[:6])
            if enabled
        ]

        if not enabled_indices:
            return 1.0

        mode_index = enabled_indices[0]

        factors = {
            0: 0.35,  # cell sleep
            1: 0.75,  # carrier off
            2: 0.90,  # MIMO reduction
            4: 0.85,  # BWP adaptation
            5: 0.60,  # light/RF-chain sleep
        }

        if mode_index == 3:
            # Micro-DTX savings decrease as traffic load rises.
            return 1.0 - 0.30 * (1.0 - load)

        return factors.get(mode_index, 1.0)
    @staticmethod
    def es_mode_feasibility_violation(
        es_modes: List[int],
        load: float,
        active_ues: int,
    ) -> Optional[str]:
        """Return why an ES mode is outside its safe load envelope."""

        load = float(np.clip(load, 0.0, 1.0))
        active_ues = max(0, int(active_ues))

        enabled_indices = [
            index
            for index, enabled in enumerate(es_modes[:6])
            if enabled
        ]

        if not enabled_indices:
            return None

        mode_index = enabled_indices[0]

        if mode_index == 0 and active_ues > 0:
            return "deep_sleep_with_active_ues"

        # The simulated deployment has only one carrier.
        if mode_index == 1 and active_ues > 0:
            return "carrier_off_with_single_active_carrier"

        operating_envelopes = {
            5: (0.00, 0.05),  # light sleep
            4: (0.05, 0.10),  # BWP adaptation
            3: (0.10, 0.25),  # micro-DTX
            2: (0.25, 0.45),  # MIMO reduction
        }

        if mode_index in operating_envelopes:
            minimum_load, maximum_load = (
                operating_envelopes[mode_index]
            )

            if not (
                minimum_load <= load < maximum_load
            ):
                return (
                    f"es_mode_{mode_index}_outside_"
                    f"load_envelope"
                )

        return None            
    @staticmethod
    def estimate_uncalibrated_energy(
        config: Dict[str, Any],
        load: float = 0.0,
    ) -> float:
        """Estimate power using the physical Simu5G ES factors."""

        tx_power = float(
            config.get("tx_power", 43.0)
        )
        bandwidth = float(
            config.get("bandwidth", 20.0)
        )
        carrier_count = int(
            config.get("carrier_count", 1)
        )
        es_modes = list(
            config.get("es_modes", [0] * 6)
        )

        base_energy = 25.0
        tx_energy = (
            10 ** (tx_power / 10.0)
        ) * 0.001
        digital_processing = (
            bandwidth * 0.4
            + carrier_count * 8.0
        )

        pre_es_energy = (
            base_energy
            + tx_energy
            + digital_processing
        )

        power_factor = (
            MultiObjectiveOptimizer.es_power_factor(
                es_modes,
                load,
            )
        )

        return max(
            0.001,
            pre_es_energy * power_factor,
        )

    def evaluate_individual(self, individual: List, current_metrics: Dict[str, Any]) -> Tuple[float, float, float]:
        """Enhanced multi-objective evaluation"""
        config = self.decode_individual(individual)
        current_load = current_metrics.get('load', 0.5)
        
        # Estimate SINR with realistic modeling
        sinr = self.qos_predictor.estimate_sinr(config['tx_power'])
        
        # Predict QoS metrics
        throughput = self.qos_predictor.predict_throughput(
            config['tx_power'], config['bandwidth'], current_load, sinr
        )
        latency = self.qos_predictor.predict_latency(
            config['tx_power'], current_load, config['es_modes']
        )
        packet_loss = self.qos_predictor.predict_packet_loss(
            config['tx_power'], sinr, current_load
        )
        
        # Energy model aligned with TwinMetricsAgent.
        base_energy = 25.0
        tx_energy = (
            10 ** (config["tx_power"] / 10.0)
        ) * 0.001
        digital_processing = (
            config["bandwidth"] * 0.4
            + config["carrier_count"] * 8.0
        )

        pre_es_energy = (
            base_energy
            + tx_energy
            + digital_processing
        )

        total_energy = (
            pre_es_energy
            * self.es_power_factor(
                config["es_modes"],
                current_load,
            )
        )
        
        # Apply constraint penalties (including policy-derived constraints)
        penalty_energy = 0
        penalty_latency = 0
        penalty_loss = 0
        active_ues = max(
            0,
            int(float(current_metrics.get("active_ues", 0))),
        )

        mode_violation = (
            self.es_mode_feasibility_violation(
                config["es_modes"],
                current_load,
                active_ues,
            )
        )

        if mode_violation is not None:
            infeasible_penalty = 1_000_000.0
            penalty_energy += infeasible_penalty
            penalty_latency += infeasible_penalty
            penalty_loss += infeasible_penalty
        
        # Throughput constraint
        min_throughput = self.constraints.get('min_throughput', 50.0)
        if throughput < min_throughput:
            penalty = (min_throughput - throughput) * 2
            penalty_energy += penalty
            penalty_latency += penalty * 0.5
        
        # Latency constraint
        max_latency = self.constraints.get('max_latency', 20.0)
        if latency > max_latency:
            penalty = (latency - max_latency) * 3
            penalty_latency += penalty
            penalty_energy += penalty * 0.3
        
        # Packet loss constraint
        max_packet_loss = self.constraints.get('max_packet_loss', 0.01)
        if packet_loss > max_packet_loss:
            penalty = (packet_loss - max_packet_loss) * 1000
            penalty_loss += penalty
            penalty_energy += penalty * 0.1
        
        # NEW: Policy-based power constraint
        max_tx_power = self.constraints.get('max_tx_power', 43.0)
        if config['tx_power'] > max_tx_power:
            penalty = (config['tx_power'] - max_tx_power) * 50
            penalty_energy += penalty
        
        # Return three objectives: energy, latency, packet loss
        return (total_energy + penalty_energy,
                latency + penalty_latency,
                packet_loss + penalty_loss)

    def optimize_genetic_algorithm(self, current_metrics: Dict[str, Any],
                                 population_size: int = 100,
                                 generations: int = 150) -> List[OptimizedConfiguration]:
        """Enhanced genetic algorithm optimization using NSGA-II"""
        self.logger.info(f"Starting enhanced NSGA-II optimization: {population_size} pop, {generations} gen")
        
        # Update evaluation function
        self.toolbox.register("evaluate", self.evaluate_individual, current_metrics=current_metrics)
        
        # Initialize population
        population = self.toolbox.population(n=population_size)
        
        # Evaluate initial population
        fitnesses = [self.toolbox.evaluate(ind) for ind in population]
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        
        # Enhanced evolutionary algorithm with elitism (NSGA-II)
        algorithms.eaMuPlusLambda(
            population, self.toolbox,
            mu=population_size//2,
            lambda_=population_size,
            cxpb=0.7, mutpb=0.3, # Higher mutation for exploration
            ngen=generations,
            stats=stats,
            verbose=False
        )
        
        # Extract Pareto front
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        
        # Convert to OptimizedConfiguration objects
        configurations = []
        for individual in pareto_front:
            config = self.decode_individual(individual)
            
            # Recalculate detailed metrics
            current_load = current_metrics.get('load', 0.5)
            sinr = self.qos_predictor.estimate_sinr(config['tx_power'])
            
            throughput = self.qos_predictor.predict_throughput(
                config['tx_power'], config['bandwidth'], current_load, sinr
            )
            latency = self.qos_predictor.predict_latency(
                config['tx_power'], current_load, config['es_modes']
            )
            packet_loss = self.qos_predictor.predict_packet_loss(
                config['tx_power'], sinr, current_load
            )
            
            # Energy calculation aligned with TwinMetricsAgent.
            base_energy = 25.0
            tx_energy = (
                10 ** (config["tx_power"] / 10.0)
            ) * 0.001
            digital_processing = (
                config["bandwidth"] * 0.4
                + config["carrier_count"] * 8.0
            )

            pre_es_energy = (
                base_energy
                + tx_energy
                + digital_processing
            )

            total_energy = (
                pre_es_energy
                * self.es_power_factor(
                    config["es_modes"],
                    current_load,
                )
            )
            
            # Check constraint violations
            violations = {}
            active_ues = max(
                0,
                int(float(current_metrics.get("active_ues", 0))),
            )

            mode_violation = (
                self.es_mode_feasibility_violation(
                    config["es_modes"],
                    current_load,
                    active_ues,
                )
            )

            if mode_violation is not None:
                violations[mode_violation] = 1.0
                
            if throughput < self.constraints.get('min_throughput', 50.0):
                violations['throughput'] = self.constraints['min_throughput'] - throughput
            if latency > self.constraints.get('max_latency', 20.0):
                violations['latency'] = latency - self.constraints['max_latency']
            if packet_loss > self.constraints.get('max_packet_loss', 0.01):
                violations['packet_loss'] = packet_loss - self.constraints['max_packet_loss']
            if config['tx_power'] > self.constraints.get('max_tx_power', 43.0):
                violations['tx_power'] = config['tx_power'] - self.constraints['max_tx_power']
            
            opt_config = OptimizedConfiguration(
                tx_power=config['tx_power'],
                bandwidth=config['bandwidth'],
                carrier_count=config['carrier_count'],
                es_mode_config=config['es_modes'],
                predicted_energy=total_energy,
                predicted_qos_metrics={
                    'throughput': throughput,
                    'latency': latency,
                    'packet_loss': packet_loss,
                    'sinr': sinr
                },
                optimization_score=individual.fitness.values[0],
                constraint_violations=violations
            )
            configurations.append(opt_config)
        
        self.logger.info(f"NSGA-II optimization completed. Found {len(configurations)} Pareto solutions")
        return configurations

class OptimizationAgent:
    """
    Enhanced Multi-Objective Optimization Agent with 3GPP Policy Integration using FAISS
    """
    async def request_prediction_from_agent(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send prediction request to PredictionAgent and get response"""
        try:
            # Create a direct socket to PredictionAgent for requests
            prediction_request_socket = self.communication.context.socket(zmq.REQ)
            prediction_request_socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10 second timeout
            prediction_request_socket.connect("tcp://localhost:5555")  # PredictionAgent request port
            
            # Send request
            await prediction_request_socket.send_json(request_data)
            
            # Receive response
            response = await prediction_request_socket.recv_json()
            
            # Cleanup
            prediction_request_socket.close()
            
            self.logger.debug(f"Received prediction response for {request_data.get('bs_id', 'unknown')}")
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to get prediction from PredictionAgent: {e}")
            return {'error': str(e)}

    def __init__(
        self,
        config_path: str = "optimization_agent_config.yaml",
        control_mode_override: Optional[str] = None,
    ):
        self.logger = logging.getLogger('OptimizationAgent')
        self.config = self.load_config(config_path)
        self.agent_id = self.config['agent']['id']
        
        # Experiment/control mode configuration
        experiment_config = self.config.get("experiment", {})

        self.control_mode = (
            control_mode_override
            or experiment_config.get(
                "mode",
                "rule_based",
            )
        )

        if self.control_mode not in {
            "rule_based",
            "ai_based",
        }:
            raise ValueError(
                "experiment.mode must be either "
                "'rule_based' or 'ai_based'"
            )

        self.decision_interval_sim_seconds = float(
            experiment_config.get(
                "decision_interval_sim_seconds", 900
            )
        )

        self.minimum_load_change = float(
            experiment_config.get(
                "minimum_load_change", 0.05
            )
        )
        self.minimum_energy_saving_percent = max(
            0.0,
            float(
                experiment_config.get(
                    "minimum_energy_saving_percent",
                    0.5,
                )
            ),
        )
        # Per-BS decision tracking
        self.last_decision_sim_time = {}
        self.last_decision_load = {}

        # AI plans waiting for FeedbackAgent approval
        self.pending_validated_plans = {}
        # Track asynchronous LLM healing without blocking the
        # optimization and feedback message loops.
        self.healing_tasks = set()
        self.logger.info(
            "Control mode: %s | decision interval: %.0fs",
            self.control_mode,
            self.decision_interval_sim_seconds,
        )
        # Initialize components
        self.qos_predictor = QoSPredictor()
        self.optimizer = None
        
        # NEW: Initialize 3GPP Policy Manager with FAISS
        default_policies = [
            "3GPP TS 28.541: Network slice instances shall have maximum end-to-end latency < 10 ms for URLLC services",
            "3GPP TS 22.261: Enhanced mobile broadband minimum user data rate shall be > 100 Mbps in dense urban areas",
            "3GPP TS 38.300: Packet loss rate for eMBB services must not exceed 0.1%",
            "3GPP TS 28.530: Energy efficiency optimization should maintain QoS constraints while reducing power consumption",
            "3GPP TR 38.901: Maximum transmission power for gNB shall not exceed 43 dBm for macro cells",
            "Network slice for autonomous vehicles requires latency < 5 ms and reliability > 99.999%",
            "Industrial IoT applications need minimum throughput 1 Mbps with maximum latency 50 ms",
            "Massive IoT connections should have packet loss < 0.01% and support up to 1M devices per km²"
        ]
        
        policy_texts = self.config.get('policy_texts', default_policies)
        self.policy_manager = PolicyManager(policy_texts)
        
        # Initialize Orchestrator Interface
       # Around line 1376 - REPLACE THE ORCHESTRATOR INITIALIZATION:
        print(f"[DEBUG] ==================== CONFIG DEBUG ====================")
        print(f"[DEBUG] Config keys: {list(self.config.keys())}")

        orchestrator_config = self.config.get('orchestrator', {})
        print(f"[DEBUG] Orchestrator config found: {orchestrator_config}")
        print(f"[DEBUG] Orchestrator enabled: {orchestrator_config.get('enabled', 'NOT_FOUND')}")
        print(f"[DEBUG] Orchestrator host: {orchestrator_config.get('orchestrator_host', 'NOT_FOUND')}")
        print(f"[DEBUG] Orchestrator port: {orchestrator_config.get('orchestrator_port', 'NOT_FOUND')}")
        print(f"[DEBUG] ======================================================")

# Check if orchestrator is enabled
        if orchestrator_config.get('enabled', False):
            print(f"[DEBUG] [ACTION] ORCHESTRATOR ENABLED - Creating interface...")

            try:
                self.orchestrator_interface = OrchestratorInterface(orchestrator_config)
                print(f"[DEBUG] [OK] OrchestratorInterface created successfully!")
            except Exception as e:
                print(f"[DEBUG] [ERROR] Failed to create OrchestratorInterface: {e}")
                self.orchestrator_interface = None
        else:
            print(f"[DEBUG] [ERROR] ORCHESTRATOR DISABLED in config!")
            self.orchestrator_interface = None
        
        # Initialize LLM template generator
        llm_config = self.config.get("llm", {})
        self.llm_healing_calls = 0
        self.max_llm_healing_calls = max(
            0,
            int(
                llm_config.get(
                    "max_healing_calls_per_run",
                    3,
                )
            ),
        )
        # Count healing actions scheduled during this experiment.
        # Rejected configurations are never deployed, so only a bounded
        # sample needs an executable LLM healing demonstration.
        self.healing_actions_scheduled = 0
        if (
            self.control_mode == "ai_based"
            and llm_config.get("enabled", True)
        ):
            self.template_generator = LLMTemplateGenerator(
                llm_config
            )
        else:
            self.template_generator = None
            self.logger.info(
                "LLM disabled for control mode: %s",
                self.control_mode,
            )
          # REPLACED: Enhanced ZeroMQ setup with dedicated communication handler
        self.communication = None
        
        # Internal state (keep existing) - NOTE: Queue will be created in async context
        self.optimization_queue = None  # Will be created in initialize_communication
        self.active_optimizations = {}
        
        # Performance tracking
        self.metrics = {
            'optimizations_completed': 0,
            'total_optimization_time': 0,
            'energy_savings_achieved': [],
            'constraint_violations': 0,
            'feedback_received': 0,    
            'feedback_applied': 0,  
            'policies_applied': 0, # NEW: Track policy applications
            'start_time': time.time()
        }
        
        self.logger.info(f"OptimizationAgent {self.agent_id} initialized with 3GPP Policy Integration")
    

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration"""
        try:
            # Get absolute path relative to script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(script_dir, 'config')
            abs_config_path = os.path.join(config_dir, config_path)
            
            # Create config directory if it doesn't exist
            os.makedirs(config_dir, exist_ok=True)
            
            # If config doesn't exist, create it with defaults
            if not os.path.exists(abs_config_path):
                default_config = self.get_default_config()
                with open(abs_config_path, 'w') as f:
                    yaml.dump(default_config, f)
                self.logger.info(f"Created default config at {abs_config_path}")
                return default_config
                
            # Load existing config
            with open(abs_config_path, 'r') as f:
                return yaml.safe_load(f)
                
        except Exception as e:
            self.logger.warning(f"Error loading config: {e}, using defaults")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Enhanced default configuration with ZeroMQ communication"""
        return {
            'agent': {
                'id': 'optimization_agent_with_enhanced_zeromq_001',
                'name': 'Enhanced Multi-Objective Optimization Agent with ZeroMQ Communication'
            },
            'optimization': {
                'default_method': 'genetic_algorithm',
                'population_size': 100,
                'generations': 150,
                'constraints': {
                    'min_throughput': 50.0,
                    'max_latency': 20.0,
                    'max_packet_loss': 0.01
                }
            },
            'policy_texts': [
                "3GPP TS 28.541: Network slice instances shall have maximum end-to-end latency < 10 ms for URLLC services",
                "3GPP TS 22.261: Enhanced mobile broadband minimum user data rate shall be > 100 Mbps in dense urban areas",
                "3GPP TS 38.300: Packet loss rate for eMBB services must not exceed 0.1%",
                "3GPP TS 28.530: Energy efficiency optimization should maintain QoS constraints while reducing power consumption",
                "3GPP TR 38.901: Maximum transmission power for gNB shall not exceed 43 dBm for macro cells"
            ],
            "llm": {
                "type": "openai",
                "model": "gpt-5-mini-2025-08-07",
                "api_key": "",
                "max_output_tokens": 1500,
                "execution_insights_enabled": False,
                "healing_enabled": True,
                "enabled": True,
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "model_path": "",
            },
            'orchestrator': {
                'enabled': True,
                'orchestrator_host': 'localhost',
                'orchestrator_port': 5562,
                'auto_execute': True,
                'execution_timeout': 300,
                'retry_attempts': 3
            },
            'communication': {
                'request_port': 5559,
                'pub_port': 5560,
                'feedback_host': 'localhost',
                'feedback_agent_port': 5561,                'prediction_host': 'localhost', 
                'prediction_agent_port': 5558
            }
        }
    
    async def initialize_communication(self):
        """Initialize ZeroMQ communication using dedicated handler"""
        try:
            # Create optimization queue in async context
            self.optimization_queue = asyncio.Queue()
            self.logger.info("[OK] Created optimization queue in async context")
            
            # Create communication handler with configuration
            communication_config = self.config["communication"]

            comm_config = {
                "agent_id": self.agent_id,

                # Local sockets owned by OptimizationAgent
                "prediction_port": communication_config.get(
                    "prediction_port", 5558
                ),
                "response_port": communication_config.get(
                    "response_port", 5559
                ),
                "request_port": communication_config.get(
                    "request_port", 5565
                ),
                "pub_port": communication_config.get(
                    "pub_port", 5564
                ),

                "endpoints": {
                    # OptimizationAgent PUSH -> FeedbackAgent PULL
                    "feedback_agent": {
                        "host": communication_config.get(
                            "feedback_host", "localhost"
                        ),
                        "port": communication_config.get(
                            "feedback_agent_port", 5560
                        ),
                    },

                    # Optional direct requests to PredictionAgent
                    "prediction_agent": {
                        "host": communication_config.get(
                            "prediction_host", "localhost"
                        ),
                        "port": communication_config.get(
                            "prediction_agent_port", 5555
                        ),
                    },
                },
            }
            
            self.communication = OptimizationCommunicationHandler(comm_config)
            await self.communication.initialize()
            
            # Register message handlers
            handlers = {
                OptimizationMessageType.OPTIMIZATION_REQUEST: self._handle_optimization_request_msg,
                OptimizationMessageType.PREDICTION_UPDATE: self._handle_prediction_update_msg,
                OptimizationMessageType.HEARTBEAT: self._handle_heartbeat_msg,
                OptimizationMessageType.STATUS_REQUEST: self._handle_status_request_msg,
                OptimizationMessageType.FEEDBACK_RESPONSE: self._handle_feedback_response_msg,  
                OptimizationMessageType.VALIDATION_RESULT: self._handle_feedback_response_msg   
            }
            
            self.communication.register_handlers(handlers)
            self.logger.info("Enhanced communication channels initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced communication: {e}")
            raise


    # NEW: Message handler methods
    async def _handle_optimization_request_msg(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle optimization request message via ZeroMQ"""
        try:
            request = OptimizationRequest(**payload)
            self.active_optimizations[request.request_id] = request
            response = await self.process_optimization_request(request)
            
            # Send result to Feedback Agent for validation
            await self.communication.send_to_feedback_agent(
                OptimizationMessageType.VALIDATE_ENERGY_PLAN,
                asdict(response)
            )
            
            if request.request_id in self.active_optimizations:
                del self.active_optimizations[request.request_id]
                
            return asdict(response)
            
        except Exception as e:
            self.logger.error(f"Error handling optimization request: {e}")
            return {'error': str(e)}

    async def _handle_prediction_update_msg(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route a prediction through rule-based or AI control."""

        try:
            bs_id = str(payload.get("bs_id", "unknown"))
            request_id = payload.get(
                "request_id",
                f"pred_{bs_id}_{int(time.time())}",
            )

            predicted_energy = float(
                payload.get("predicted_energy", 0.0)
            )
            confidence_score = float(
                payload.get("confidence_score", 0.0) or 0.0
            )
            sim_time_s = float(
                payload.get("sim_time_s", 0.0) or 0.0
            )

            current_metrics = payload.get(
                "current_metrics", {}
            )
            current_config = payload.get(
                "current_config", {}
            )
            cell_metadata = payload.get(
                "cell_metadata", {}
            )

            load = float(current_metrics.get("load", 0.5))
            latency_ms = float(
                current_metrics.get("latency", 0.0)
            )
            packet_loss_pct = float(
                current_metrics.get("packet_loss", 0.0)
            )

            self.logger.info(
                "Prediction received: BS=%s energy=%.2fW "
                "confidence=%.3f load=%.3f sim_time=%.1fs",
                bs_id,
                predicted_energy,
                confidence_score,
                load,
                sim_time_s,
            )

            # Keep the latest complete prediction for every BS.
            if not hasattr(self, "prediction_cache"):
                self.prediction_cache = {}

            self.prediction_cache[bs_id] = {
                "predicted_energy": predicted_energy,
                "confidence_score": confidence_score,
                "sim_time_s": sim_time_s,
                "request_id": request_id,
                "current_metrics": current_metrics,
                "current_config": current_config,
                "cell_metadata": cell_metadata,
                "received_at": time.time(),
            }

            # Decide whether a new control decision is due.
            first_decision = (
                bs_id not in self.last_decision_sim_time
            )

            last_time = self.last_decision_sim_time.get(
                bs_id, float("-inf")
            )
            last_load = self.last_decision_load.get(
                bs_id, load
            )

            elapsed = sim_time_s - last_time
            load_change = abs(load - last_load)

            if not first_decision:
                if elapsed < self.decision_interval_sim_seconds:
                    return {
                        "status": "cached",
                        "action": "waiting_for_decision_interval",
                        "bs_id": bs_id,
                        "sim_time_s": sim_time_s,
                        "seconds_until_next_decision": (
                            self.decision_interval_sim_seconds
                            - elapsed
                        ),
                    }

                if load_change < self.minimum_load_change:
                    return {
                        "status": "cached",
                        "action": "insignificant_load_change",
                        "bs_id": bs_id,
                        "sim_time_s": sim_time_s,
                        "load_change": load_change,
                    }

            self.last_decision_sim_time[bs_id] = sim_time_s
            self.last_decision_load[bs_id] = load

            # ==================================================
            # RULE-BASED / DETERMINISTIC MODE
            # ==================================================
            if self.control_mode == "rule_based":
                decision = select_rule_based_config(
                    load=load,
                    latency_ms=latency_ms,
                    packet_loss_pct=packet_loss_pct,
                    active_ues=int(
                        current_metrics.get(
                            "active_ues",
                            0,
                        )
                    ),
                    current_tx_power=float(
                        current_config.get("tx_power", 43.0)
                    ),
                    current_bandwidth=int(
                        current_config.get("bandwidth", 20)
                    ),
                    current_carrier_count=int(
                        current_config.get("carrier_count", 1)
                    ),
                    bs_type=str(
                        cell_metadata.get("bs_type", "macro")
                    ),
                )

                command = {
                    "action": "applyConfiguration",
                    "target": bs_id,
                    "control_mode": "rule_based",
                    "request_id": request_id,
                    "sim_time_s": sim_time_s,
                    "param": {
                        "bs_id": bs_id,
                        "tx_power_dbm": decision.tx_power,
                        "bandwidth_mhz": decision.bandwidth,
                        "carrier_count": decision.carrier_count,
                        "es_mode": decision.es_mode,
                        "es_modes": decision.es_modes,
                    },
                    "decision": decision.to_dict(),
                }

                if self.orchestrator_interface is None:
                    orchestration_result = {
                        "status": "NOT_SENT",
                        "detail": (
                            "OrchestrationAgent unavailable"
                        ),
                    }
                else:
                    orchestration_result = (
                        self.orchestrator_interface.send_command(
                            command
                        )
                    )

                orchestration_status = (
                    orchestration_result.get(
                        "status", "UNKNOWN"
                    )
                )

                if orchestration_status == "OK":
                    result_status = "processed"
                else:
                    result_status = "not_applied"
                    self.logger.warning(
                        "Rule-based decision for BS %s "
                        "could not be applied: %s",
                        bs_id,
                        orchestration_result,
                    )

                self.logger.info(
                    "Rule-based decision for BS %s: %s, "
                    "ES=%s, TX=%.1fdBm",
                    bs_id,
                    decision.rule_name,
                    decision.es_mode,
                    decision.tx_power,
                )

                return {
                    "status": result_status,
                    "action": "rule_based_configuration",
                    "bs_id": bs_id,
                    "request_id": request_id,
                    "decision": decision.to_dict(),
                    "orchestration_result": orchestration_result,
                }

            # ==================================================
            # AI-BASED MODE
            # ==================================================
            optimization_request = OptimizationRequest(
                request_id=request_id,
                timestamp=time.time(),
                bs_id=bs_id,
                current_config=current_config,
                current_metrics=current_metrics,
                predicted_energy=predicted_energy,
                confidence_score=confidence_score,
                constraints=self.config[
                    "optimization"
                ]["constraints"].copy(),
                objective=(
                    OptimizationObjective.MINIMIZE_ENERGY
                ),
                policy_text=payload.get("policy_text"),
            )

            # Extra experiment metadata used when logging results.
            optimization_request.sim_time_s = sim_time_s
            optimization_request.cell_metadata = cell_metadata
            optimization_request.control_mode = "ai_based"

            self.active_optimizations[
                request_id
            ] = optimization_request

            await self.optimization_queue.put(
                optimization_request
            )

            self.logger.info(
                "Queued AI optimization for BS %s at %.1fs",
                bs_id,
                sim_time_s,
            )

            return {
                "status": "processed",
                "action": "queued_ai_optimization",
                "bs_id": bs_id,
                "request_id": request_id,
                "sim_time_s": sim_time_s,
            }

        except Exception as error:
            self.logger.error(
                "Error handling prediction update: %s",
                error,
                exc_info=True,
            )
            return {
                "status": "error",
                "error": str(error),
                "payload": payload,
            }
        
    async def _handle_heartbeat_msg(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle heartbeat from other agents"""
        return {
            'agent_id': self.agent_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'alive'
        }

    async def _handle_status_request_msg(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status request"""
        return self.get_performance_metrics()

    async def process_optimization_request(self, request: OptimizationRequest) -> OptimizationResponse:
        """Process optimization request with 3GPP policy integration and LLM template generation"""
        start_time = time.time()
        applied_policies = []
        
        try:
            # NEW: Apply 3GPP policy constraints if policy_text is provided
            base_constraints = {**self.config['optimization']['constraints'], **request.constraints}
            
            if hasattr(request, 'policy_text') and request.policy_text:
                self.logger.info(f"Processing 3GPP policy: {request.policy_text}")
                updated_constraints, applied_policies = self.policy_manager.update_constraints_with_policy(
                    request.policy_text, base_constraints
                )
                self.metrics['policies_applied'] += len(applied_policies)
            else:
                updated_constraints = base_constraints
            
            # Request energy prediction from PredictionAgent if not already available
            if request.predicted_energy <= 0:
                self.logger.info(f"Requesting energy prediction for {request.bs_id}")
                
                prediction_request = {
                    'request_id': request.request_id,
                    'timestamp': time.time(),
                    'bs_id': request.bs_id,
                    'metrics': request.current_metrics,
                    'config_params': request.current_config
                }
                
                prediction_response = await self.request_prediction_from_agent(prediction_request)
                
                if 'error' not in prediction_response:
                    request.predicted_energy = prediction_response.get('predicted_energy', request.predicted_energy)
                    request.confidence_score = prediction_response.get('confidence_score', 0.5)
                    self.logger.info(f"Received energy prediction: {request.predicted_energy:.2f}W")
                else:
                    self.logger.warning(f"Failed to get prediction, using default: {prediction_response['error']}")

            # Initialize optimizer with policy-updated constraints
            self.optimizer = MultiObjectiveOptimizer(
                self.qos_predictor,
                updated_constraints,
                self.config.get(
                    "optimization",
                    {},
                ).get(
                    "bounds",
                    {},
                ),
            )
            
            # Perform multi-objective optimization using NSGA-II
            pareto_solutions = self.optimizer.optimize_genetic_algorithm(
                request.current_metrics,
                population_size=self.config['optimization']['population_size'],
                generations=self.config['optimization']['generations']
            )
            # Candidate ranking uses a relative analytical model.
            # Calibrate it to the Keras ANN prediction so current and
            # optimized energy remain on the same watt scale.
            uncalibrated_current_energy = (
                self.optimizer.estimate_uncalibrated_energy(
                    request.current_config,
                    load=float(
                        request.current_metrics.get(
                            "load", 0.0
                        )
                    ),
                )
            )

            if (
                request.predicted_energy > 0.0
                and uncalibrated_current_energy > 0.0
            ):
                energy_calibration_factor = (
                    request.predicted_energy
                    / uncalibrated_current_energy
                )

                for solution in pareto_solutions:
                    solution.predicted_energy *= (
                        energy_calibration_factor
                    )
            else:
                energy_calibration_factor = 1.0

            self.logger.info(
                "Energy calibration for BS %s: "
                "ANN baseline=%.3fW, analytical baseline=%.3fW, "
                "factor=%.6f",
                request.bs_id,
                request.predicted_energy,
                uncalibrated_current_energy,
                energy_calibration_factor,
            )            
            # Select best solution (minimum energy with no constraint violations)
            valid_solutions = [sol for sol in pareto_solutions if not sol.constraint_violations]
            if valid_solutions:
                best_solution = min(valid_solutions, key=lambda x: x.predicted_energy)
            else:
                best_solution = min(pareto_solutions,
                                  key=lambda x: sum(x.constraint_violations.values()) + x.predicted_energy)
            
            # Calculate energy savings
            current_energy = request.predicted_energy
            optimized_energy = best_solution.predicted_energy
            energy_saving_percent = ((current_energy - optimized_energy) / current_energy) * 100
            
            # Analyze QoS impact
            qos_impact = {}
            for metric, value in best_solution.predicted_qos_metrics.items():
                if metric in request.current_metrics:
                    current_value = request.current_metrics[metric]
                    if current_value != 0:
                        impact_percent = ((value - current_value) / current_value) * 100
                        qos_impact[metric] = impact_percent
            
            # Generate LLM-powered execution template
            execution_template = self.template_generator.generate_execution_template(
                request, best_solution, energy_saving_percent
            )
            
            # Add policy context to execution template
            if applied_policies:
                execution_template["applied_3gpp_policies"] = applied_policies
                execution_template["policy_constraints"] = updated_constraints
            
            # Generate explanation with policy information
            explanation = self.generate_optimization_explanation(
                request, best_solution, energy_saving_percent, applied_policies
            )
            
            # Calculate confidence
            confidence = self.calculate_optimization_confidence(
                best_solution, energy_saving_percent
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # AI configurations must be approved by FeedbackAgent
            # before being sent to OrchestrationAgent.
            execution_results = {
                "status": "PENDING_VALIDATION"
            }
            
            response = OptimizationResponse(
                request_id=request.request_id,
                timestamp=time.time(),
                bs_id=request.bs_id,
                original_config=request.current_config,
                optimized_config=best_solution,
                energy_saving_percent=energy_saving_percent,
                qos_impact_analysis=qos_impact,
                pareto_solutions=pareto_solutions,
                optimization_method="Enhanced NSGA-II with 3GPP Policy Integration and LLM Templates",
                processing_time_ms=processing_time,
                explanation=explanation,
                confidence=confidence,
                execution_template=execution_template,
                applied_policies=applied_policies # NEW: Include applied policies
            )
            
            # Update metrics
            self.metrics['optimizations_completed'] += 1
            self.metrics['total_optimization_time'] += processing_time
            self.metrics['energy_savings_achieved'].append(energy_saving_percent)
            if best_solution.constraint_violations:
                self.metrics['constraint_violations'] += 1
            
            self.logger.info(f"Optimization completed for {request.bs_id}: "
                           f"{energy_saving_percent:.1f}% energy savings with {len(applied_policies)} 3GPP policies applied")
            
            # Add orchestrator execution results to template
            if execution_template:
                execution_template['orchestrator_execution'] = execution_results
            
            return response
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            # Return error response
            error_config = OptimizedConfiguration(
                tx_power=request.current_config.get('tx_power', 20.0),
                bandwidth=request.current_config.get('bandwidth', 10),
                carrier_count=request.current_config.get('carrier_count', 1),
                es_mode_config=request.current_config.get('es_modes', [0]*6),
                predicted_energy=request.predicted_energy,
                predicted_qos_metrics={},
                optimization_score=0.0,
                constraint_violations={'error': 1.0}
            )
            
            return OptimizationResponse(
                request_id=request.request_id,
                timestamp=time.time(),
                bs_id=request.bs_id,
                original_config=request.current_config,
                optimized_config=error_config,
                energy_saving_percent=0.0,
                qos_impact_analysis={},
                pareto_solutions=[],
                optimization_method="Failed",
                processing_time_ms=processing_time,
                explanation=f"Optimization failed: {str(e)}",
                confidence=0.0,
                execution_template={"error": str(e)},
                applied_policies=applied_policies
            )

    def generate_optimization_explanation(self, request: OptimizationRequest,
                                        solution: OptimizedConfiguration,
                                        energy_saving: float,
                                        applied_policies: List[str] = None) -> str:
        """Generate comprehensive optimization explanation with policy information"""
        explanations = []
        
        # Energy savings analysis
        if energy_saving > 0:
            explanations.append(f"Achieved {energy_saving:.1f}% energy reduction "
                              f"({solution.predicted_energy - request.predicted_energy:.1f}W savings)")
        else:
            explanations.append("No energy savings possible while maintaining QoS constraints")
        
        # Configuration changes analysis
        original_config = request.current_config
        if solution.tx_power != original_config.get('tx_power', 20):
            power_change = solution.tx_power - original_config.get('tx_power', 20)
            direction = "increased" if power_change > 0 else "reduced"
            explanations.append(f"TX power {direction} by {abs(power_change):.1f} dBm "
                              f"(from {original_config.get('tx_power', 20)} to {solution.tx_power})")
        
        if solution.bandwidth != original_config.get('bandwidth', 20):
            explanations.append(f"Bandwidth optimized from {original_config.get('bandwidth', 20)} "
                              f"to {solution.bandwidth} MHz")
        
        # Energy saving modes analysis
        original_es = original_config.get('es_modes', [0]*6)
        new_es_count = sum(solution.es_mode_config)
        original_es_count = sum(original_es)
        
        if new_es_count > original_es_count:
            activated_modes = [i for i in range(6)
                             if solution.es_mode_config[i] and not original_es[i]]
            explanations.append(f"Activated {len(activated_modes)} energy saving modes: {activated_modes}")
        elif new_es_count < original_es_count:
            deactivated_modes = [i for i in range(6)
                               if not solution.es_mode_config[i] and original_es[i]]
            explanations.append(f"Deactivated {len(deactivated_modes)} energy saving modes: {deactivated_modes}")
        
        # QoS impact analysis
        qos_status = []
        if solution.predicted_qos_metrics.get('throughput', 0) >= request.constraints.get('min_throughput', 50):
            qos_status.append("[OK] Throughput maintained")
        else:
            qos_status.append("⚠️ Throughput may be impacted")
        
        if solution.predicted_qos_metrics.get('latency', 100) <= request.constraints.get('max_latency', 20):
            qos_status.append("[OK] Latency within limits")
        else:
            qos_status.append("⚠️ Latency may increase")
        
        explanations.append(" | ".join(qos_status))
        
        # NEW: 3GPP Policy information
        if applied_policies:
            explanations.append(f"📋 Applied {len(applied_policies)} 3GPP policy constraints")
        
        # Constraint compliance
        if not solution.constraint_violations:
            explanations.append("[OK] All 5G QoS constraints satisfied")
        else:
            violations = list(solution.constraint_violations.keys())
            explanations.append(f"⚠️ Constraint violations detected: {violations}")
        
        return " • ".join(explanations)

    def calculate_optimization_confidence(self, solution: OptimizedConfiguration,
                                        energy_saving: float) -> float:
        """Enhanced confidence calculation"""
        confidence = 1.0
        
        # Constraint violation penalty
        if solution.constraint_violations:
            violation_penalty = sum(solution.constraint_violations.values()) * 0.2
            confidence -= min(0.6, violation_penalty)
        
        # Energy savings boost
        if energy_saving > 0:
            savings_boost = min(0.3, energy_saving / 100)
            confidence += savings_boost
        else:
            confidence *= 0.5
        
        # QoS metrics reliability
        throughput = solution.predicted_qos_metrics.get('throughput', 0)
        latency = solution.predicted_qos_metrics.get('latency', 100)
        if throughput > 0 and latency < 50: # Reasonable predictions
            confidence += 0.1
        else:
            confidence *= 0.8
        
        # Configuration reasonableness
        if 10 <= solution.tx_power <= 40 and solution.bandwidth > 0:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))

    # In process_optimization_queue method around line 1555

    async def process_optimization_queue(self):
        """Optimize AI requests and submit plans for validation."""

        while True:
            request = await self.optimization_queue.get()

            try:
                self.logger.info(
                    "Processing AI optimization for BS %s",
                    request.bs_id,
                )

                response = await self.process_optimization_request(
                    request
                )

                if response.optimization_method == "Failed":
                    self.logger.error(
                        "Optimization failed for BS %s: %s",
                        response.bs_id,
                        response.explanation,
                    )
                    continue
                if (
                    response.energy_saving_percent
                    < self.minimum_energy_saving_percent
                ):
                    self.metrics["no_benefit_plans_skipped"] = (
                        self.metrics.get(
                            "no_benefit_plans_skipped", 0
                        ) + 1
                    )

                    self.active_optimizations.pop(
                        request.request_id,
                        None,
                    )

                    self.logger.info(
                        "Skipping AI plan for BS %s: predicted "
                        "saving %.2f%% is below the %.2f%% "
                        "deployment threshold",
                        response.bs_id,
                        response.energy_saving_percent,
                        self.minimum_energy_saving_percent,
                    )
                    continue
                optimized = response.optimized_config

                # The simulator currently accepts one named ES mode,
                # while retaining all six flags for analysis.
                simulator_mode_names = [
                    "sleep",
                    "carrier_off",
                    "mimo_reduce",
                    "micro_dtx",
                    "bwp_adaptation",
                    "light_sleep",
                ]

                enabled_indices = [
                    index
                    for index, enabled in enumerate(
                        optimized.es_mode_config
                    )
                    if enabled
                ]

                if enabled_indices:
                    es_mode = simulator_mode_names[
                        enabled_indices[0]
                    ]
                else:
                    es_mode = "active"

                proposed_config = {
                    "tx_power": float(optimized.tx_power),
                    "tx_power_dbm": float(optimized.tx_power),
                    "bandwidth": int(optimized.bandwidth),
                    "bandwidth_mhz": int(optimized.bandwidth),
                    "carrier_count": int(
                        optimized.carrier_count
                    ),
                    "es_mode": es_mode,
                    "es_modes": list(
                        optimized.es_mode_config
                    ),
                }

                plan_id = (
                    f"PLAN_{response.bs_id}_"
                    f"{request.request_id}"
                )

                feedback_payload = {
                    "plan_id": plan_id,
                    "target_bs_id": response.bs_id,
                    "expected_energy_saving": (
                        response.energy_saving_percent
                    ),
                    "energy_saving_percentage": (
                        response.energy_saving_percent
                    ),
                    "confidence_score": response.confidence,
                    "proposed_config": proposed_config,
                    "optimized_config": asdict(optimized),
                    "processing_time_ms": (
                        response.processing_time_ms
                    ),
                    "applied_policies": (
                        response.applied_policies or []
                    ),
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "request_id": request.request_id,
                        "sim_time_s": getattr(
                            request, "sim_time_s", 0.0
                        ),
                        "cell_metadata": getattr(
                            request, "cell_metadata", {}
                        ),
                        "control_mode": "ai_based",
                        "predicted_energy_before_w": (
                            request.predicted_energy
                        ),
                        "predicted_energy_after_w": (
                            optimized.predicted_energy
                        ),
                    },
                }

                # Retain the plan until FeedbackAgent approves or
                # rejects it.
                self.pending_validated_plans[plan_id] = {
                    "bs_id": response.bs_id,
                    "request_id": request.request_id,
                    "proposed_config": proposed_config,
                    "feedback_payload": feedback_payload,
                    "created_at": time.time(),
                }

                await self.communication.send_to_feedback_agent(
                    OptimizationMessageType.OPTIMIZATION_RESPONSE,
                    feedback_payload,
                )

                self.logger.info(
                    "AI plan %s sent to FeedbackAgent; "
                    "waiting for validation",
                    plan_id,
                )

            except Exception as error:
                self.logger.error(
                    "Error processing optimization queue: %s",
                    error,
                    exc_info=True,
                )

            finally:
                self.optimization_queue.task_done()

    # REPLACED: async task handling in run() method
    async def run(self):
        """Main agent execution loop with enhanced communication"""
        self.logger.info(f"Starting OptimizationAgent {self.agent_id} with Enhanced ZeroMQ Communication")
        
        try:
            await self.initialize_communication()
            self.logger.info("OptimizationAgent ready for requests")
            # Step 2: Register message handlers ← ADD THIS BLOCK
            #self.communication.message_handlers = {
            #OptimizationMessageType.PREDICTION_UPDATE: self._handle_prediction_update_msg,
            #OptimizationMessageType.OPTIMIZATION_REQUEST: self._handle_optimization_request_msg,
            #OptimizationMessageType.HEARTBEAT: self._handle_heartbeat_msg,
            #OptimizationMessageType.STATUS_REQUEST: self._handle_status_request_msg,
            #OptimizationMessageType.FEEDBACK_RESPONSE: self._handle_feedback_response_msg,
            #ptimizationMessageType.VALIDATION_RESULT: self._handle_feedback_response_msg
            #}
            self.logger.info(f"Registered {len(self.communication.handlers)} message handlers")
            self.logger.info("OptimizationAgent ready for requests")

            # Start background tasks
            tasks = await asyncio.gather(
                self._communication_loop(),  # NEW: Handle ZeroMQ communication
                self._heartbeat_loop(),
                self.process_optimization_queue(),
                return_exceptions=True
            )
            
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            if self.healing_tasks:
                self.logger.info(
                    "Cancelling %d unfinished healing tasks",
                    len(self.healing_tasks),
                )

                for healing_task in self.healing_tasks:
                    healing_task.cancel()

                await asyncio.gather(
                    *self.healing_tasks,
                    return_exceptions=True,
                )

                self.healing_tasks.clear()            
            # Enhanced cleanup
            if self.communication:
                await self.communication.close()
            self.logger.info("OptimizationAgent stopped")

    # NEW: Communication processing loops
    async def _communication_loop(self):
        """Main communication processing loop"""
        while True:
            try:
                await self.communication.process_messages(timeout=1.0)
                await asyncio.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Error in communication loop: {e}")
                await asyncio.sleep(1.0)

    async def _heartbeat_loop(self):
        """Send periodic heartbeats via enhanced communication"""
        while True:
            try:
                await self.communication.send_heartbeat()
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            except Exception as e:
                self.logger.error(f"Heartbeat failed: {e}")
                await asyncio.sleep(5.0)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics including policy usage"""
        uptime = time.time() - self.metrics['start_time']
        avg_optimization_time = (
            self.metrics['total_optimization_time'] /
            max(1, self.metrics['optimizations_completed'])
        )
        avg_energy_savings = (
            np.mean(self.metrics['energy_savings_achieved'])
            if self.metrics['energy_savings_achieved'] else 0.0
        )
        
        return {
            'agent_id': self.agent_id,
            'uptime_seconds': uptime,
            'optimizations_completed': self.metrics['optimizations_completed'],
            'average_optimization_time_ms': avg_optimization_time,
            'average_energy_savings_percent': avg_energy_savings,
            'constraint_violations': self.metrics['constraint_violations'],
            'policies_applied': self.metrics['policies_applied'], # NEW: Policy usage tracking
            'success_rate': (
                1 - (self.metrics['constraint_violations'] /
                     max(1, self.metrics['optimizations_completed']))
            )
        }
    async def _handle_feedback_response_msg(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply approved AI plans or trigger healing for rejected plans."""

        try:
            plan_id = payload.get("plan_id", "unknown")
            is_valid = payload.get("is_valid")
            policy_blocks = bool(
                payload.get(
                    "policy_blocks_optimization", False
                )
            )
            violations = payload.get("violations", [])
            recommended_adjustments = payload.get(
                "recommended_adjustments", {}
            )
            feedback_blocks = (
                policy_blocks
                or any(
                    bool(
                        violation.get(
                            "blocks_optimization", False
                        )
                    )
                    for violation in violations
                )
            )

            self.metrics["feedback_received"] = (
                self.metrics.get("feedback_received", 0) + 1
            )

            if is_valid is None:
                self.logger.error(
                    "Feedback for %s does not contain is_valid",
                    plan_id,
                )
                return {
                    "status": "error",
                    "plan_id": plan_id,
                    "error": "missing_is_valid",
                }

            pending_plan = self.pending_validated_plans.get(
                plan_id
            )

            if pending_plan is None:
                self.logger.warning(
                    "Feedback received for unknown or expired "
                    "plan: %s",
                    plan_id,
                )
                return {
                    "status": "ignored",
                    "plan_id": plan_id,
                    "reason": "unknown_plan",
                }

            bs_id = pending_plan["bs_id"]
            proposed_config = pending_plan["proposed_config"]
            request_id = pending_plan["request_id"]

            # ==============================================
            # APPROVED: apply the AI configuration
            # ==============================================
            if is_valid:
                command = {
                    "action": "applyConfiguration",
                    "target": bs_id,
                    "control_mode": "ai_based",
                    "plan_id": plan_id,
                    "request_id": request_id,
                    "param": {
                        "bs_id": bs_id,
                        "tx_power_dbm": proposed_config[
                            "tx_power_dbm"
                        ],
                        "bandwidth_mhz": proposed_config[
                            "bandwidth_mhz"
                        ],
                        "carrier_count": proposed_config[
                            "carrier_count"
                        ],
                        "es_mode": proposed_config["es_mode"],
                        "es_modes": proposed_config["es_modes"],
                    },
                    "validation": {
                        "validation_score": payload.get(
                            "validation_score", 0.0
                        ),
                        "policy_compliance_score": payload.get(
                            "policy_compliance_score", 0.0
                        ),
                    },
                }

                if self.orchestrator_interface is None:
                    orchestration_result = {
                        "status": "NOT_SENT",
                        "detail": (
                            "OrchestrationAgent unavailable"
                        ),
                    }
                else:
                    orchestration_result = (
                        self.orchestrator_interface.send_command(
                            command
                        )
                    )

                orchestration_status = (
                    orchestration_result.get(
                        "status", "UNKNOWN"
                    )
                )

                if orchestration_status != "OK":
                    # Keep the plan pending so it can be retried.
                    self.logger.error(
                        "AI plan %s was approved but could not "
                        "be applied to BS %s: %s",
                        plan_id,
                        bs_id,
                        orchestration_result,
                    )

                    return {
                        "status": "approved_not_applied",
                        "plan_id": plan_id,
                        "bs_id": bs_id,
                        "orchestration_result": (
                            orchestration_result
                        ),
                        "healing_triggered": False,
                    }

                self.metrics["feedback_applied"] = (
                    self.metrics.get(
                        "feedback_applied", 0
                    ) + 1
                )

                self.pending_validated_plans.pop(
                    plan_id, None
                )
                self.active_optimizations.pop(
                    request_id, None
                )

                self.logger.info(
                    "Approved AI plan %s applied to BS %s",
                    plan_id,
                    bs_id,
                )

                return {
                    "status": "approved",
                    "plan_id": plan_id,
                    "bs_id": bs_id,
                    "orchestration_result": (
                        orchestration_result
                    ),
                    "healing_triggered": False,
                }

            # ==============================================
            # REJECTED: never apply the original plan
            # ==============================================
            self.pending_validated_plans.pop(plan_id, None)
            self.active_optimizations.pop(request_id, None)

            if feedback_blocks:
                self.metrics["policy_violations"] = (
                    self.metrics.get(
                        "policy_violations", 0
                    ) + 1
                )

                healing_triggered = False

                if (
                    self.healing_actions_scheduled
                    < self.max_llm_healing_calls
                ):
                    self.healing_actions_scheduled += 1

                    self.logger.warning(
                        "AI plan %s rejected by a blocking safety "
                        "or policy violation; scheduling healing "
                        "demonstration %d/%d",
                        plan_id,
                        self.healing_actions_scheduled,
                        self.max_llm_healing_calls,
                    )

                    healing_task = asyncio.create_task(
                        self._trigger_llm_healing(
                            plan_id,
                            violations,
                            recommended_adjustments,
                        )
                    )

                    self.healing_tasks.add(healing_task)
                    healing_task.add_done_callback(
                        self.healing_tasks.discard
                    )

                    healing_triggered = True
                else:
                    self.logger.warning(
                        "AI plan %s rejected safely; configuration "
                        "was not applied and the healing execution "
                        "budget is exhausted",
                        plan_id,
                    )

                return {
                    "status": "rejected",
                    "plan_id": plan_id,
                    "bs_id": bs_id,
                    "healing_triggered": healing_triggered,
                }

            self.logger.warning(
                "AI plan %s rejected; configuration was not applied",
                plan_id,
            )

            return {
                "status": "rejected",
                "plan_id": plan_id,
                "bs_id": bs_id,
                "healing_triggered": False,
            }

        except Exception as error:
            self.logger.error(
                "Error handling FeedbackAgent response: %s",
                error,
                exc_info=True,
            )
            return {
                "status": "error",
                "error": str(error),
                "payload": payload,
            }

    async def _trigger_llm_healing(self, plan_id: str, violations: List[Dict], recommended_adjustments: Dict):
        """Trigger LLM-based healing plan generation for policy violations"""
        try:
            self.logger.info(f"🤖 Generating LLM healing plan for {plan_id}")
        
        # [OK] EXTRACT CRITICAL VIOLATIONS
            critical_violations = [v for v in violations if v.get('severity') == 'critical']
        
        # [OK] CREATE HEALING PROMPT
            healing_prompt = self._create_healing_prompt(plan_id, critical_violations, recommended_adjustments)
        
        # [OK] INITIALIZE healing_response VARIABLE
            healing_response = None
        
        # [OK] GENERATE HEALING PLAN (this will use your existing LLM setup)
            # Generate a bounded number of LLM healing plans.
            llm_available = (
                self.template_generator is not None
                and self.template_generator.llm_client
                is not None
            )

            within_call_limit = (
                self.llm_healing_calls
                < self.max_llm_healing_calls
            )

            if llm_available and within_call_limit:
                self.llm_healing_calls += 1

                self.logger.info(
                    "Calling LLM for healing plan (%s/%s)",
                    self.llm_healing_calls,
                    self.max_llm_healing_calls,
                )

                try:
                    healing_response = await (
                        self._generate_healing_plan(
                            healing_prompt
                        )
                    )
                    self.logger.info(
                        "LLM healing plan generated for %s",
                        plan_id,
                    )
                except Exception as error:
                    self.logger.error(
                        "LLM healing failed for %s: %s",
                        plan_id,
                        error,
                    )
                    healing_response = (
                        "LLM generation failed; using "
                        "structured recommendations."
                    )

            elif llm_available:
                self.logger.info(
                    "LLM healing-call limit reached; "
                    "using structured recommendations"
                )
                healing_response = (
                    "LLM call limit reached; using structured "
                    "policy recommendations."
                )

            else:
                self.logger.warning(
                    "LLM unavailable; using structured "
                    "healing recommendations"
                )
                healing_response = (
                    "LLM client unavailable; using structured "
                    "policy recommendations."
                )
        
        # [OK] STORE HEALING PLAN (healing_response is now always defined)
            if not hasattr(self, 'healing_plans'):
                self.healing_plans = {}
        
            self.healing_plans[plan_id] = {
            'violations': critical_violations,
            'recommendations': recommended_adjustments,
            'llm_healing_plan': healing_response,
            'generated_at': time.time()
            }
        
        # [ACTION] NEW: SEND HEALING PLAN TO ORCHESTRATOR
            await self._send_healing_plan_to_orchestrator(plan_id, healing_response, critical_violations, recommended_adjustments)
        
        # [OK] ALWAYS LOG FINAL RESULT
            self.logger.info(f"📋 Structured recommendations: {recommended_adjustments}")
        
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to generate healing plan: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")


    def _create_healing_prompt(self, plan_id: str, violations: List[Dict], recommendations: Dict) -> str:
        """Create detailed prompt for LLM healing plan generation"""
        prompt = f"""
    5G Network Optimization HEALING PLAN Required

    PLAN ID: {plan_id}
    SITUATION: Critical policy violations detected that block optimization deployment.

    CRITICAL VIOLATIONS:
    """
    
        for i, violation in enumerate(violations, 1):
            prompt += f"{i}. {violation.get('message', 'Unknown violation')}\n"
            prompt += f"   Severity: {violation.get('severity', 'unknown')}\n"
            prompt += f"   Blocks Optimization: {violation.get('blocks_optimization', False)}\n\n"
    
        prompt += f"""
    SYSTEM RECOMMENDATIONS:
    {recommendations}

    REQUIRED: Generate a detailed healing plan that:
    1. Addresses each critical violation systematically
    2. Provides step-by-step corrective actions
    3. Suggests configuration changes to meet policy requirements
    4. Includes risk mitigation strategies
    5. Defines success criteria for each healing step

    Focus on practical implementation for a 5G network engineer.
    """
    
        return prompt

    async def _generate_healing_plan(
        self,
        prompt: str,
    ) -> str:
        """Generate a feedback-triggered healing plan."""

        config = self.template_generator.config

        if not config.get("healing_enabled", True):
            return (
                "LLM healing is disabled; using structured "
                "recommendations instead."
            )

        if self.template_generator.llm_client is None:
            return (
                "LLM client unavailable; using structured "
                "recommendations instead."
            )

        try:
            if config.get("type") == "openai":
                response = await asyncio.to_thread(
                    self.template_generator
                    .llm_client
                    .responses
                    .create,
                    model=(
                        self.template_generator
                        .openai_model
                    ),
                    instructions=(
                        "You are a 5G network optimization "
                        "expert specializing in policy "
                        "compliance and self-healing plans."
                    ),
                    input=prompt,
                    max_output_tokens=int(
                        config.get(
                            "max_output_tokens",
                            1500,
                        )
                    ),
                )
                healing_text = (
                    response.output_text or ""
                ).strip()

                if not healing_text:
                    self.logger.warning(
                        "OpenAI returned an empty healing response; "
                        "using structured recommendations"
                    )
                    return (
                        "OpenAI returned no textual healing plan; "
                        "using structured policy recommendations."
                    )

                return healing_text

            response = await asyncio.to_thread(
                self.template_generator.llm_client,
                prompt,
                max_length=1500,
                temperature=0.3,
            )
            return response[0]["generated_text"]

        except Exception as error:
            self.logger.error(
                "LLM healing generation failed: %s",
                error,
            )
            return (
                f"LLM healing generation failed: {error}. "
                "Using structured recommendations instead."
            )

    async def _send_healing_plan_to_orchestrator(self, plan_id: str, healing_plan: str, 
                                           violations: List[Dict], recommendations: Dict):
        """Send healing plan to OrchestrationAgent for execution"""
        try:
        # [OK] Extract BS ID from plan_id (assuming format: PLAN_B_1000_timestamp)
            bs_id = plan_id.split('_')[1] if '_' in plan_id else 'unknown'
        
        # [OK] CREATE ORCHESTRATOR HEALING COMMAND
            healing_command = {
            "action": "executeHealingPlan",
            "control_mode": "ai_based",
            "plan_id": plan_id,
            "target": bs_id,
            "healing_plan": healing_plan,
            "critical_violations": violations,
            "structured_recommendations": recommendations,
            "timestamp": time.time(),
            "priority": "HIGH"  # Healing plans have high priority
            }
            self.logger.info(f"[DEBUG] Sending command: {json.dumps(healing_command, indent=2)}")

            self.logger.info(f"📤 Sending healing plan to OrchestrationAgent for {plan_id}")
        
        # [OK] SEND VIA EXISTING ORCHESTRATOR INTERFACE
            if self.orchestrator_interface:
                response = self.orchestrator_interface.send_command(healing_command)
                self.logger.info(f"[DEBUG] Orchestrator response: {response}")
                if response.get('status') == 'OK':
                    self.logger.info(f"[OK] Healing plan sent successfully to OrchestrationAgent: {plan_id}")
                    self.logger.info(f"   Orchestrator Response: {response.get('message', 'Plan queued for execution')}")
                
                # [OK] UPDATE METRICS
                    self.metrics['healing_plans_sent'] = self.metrics.get('healing_plans_sent', 0) + 1
                
                else:
                    self.logger.error(f"[ERROR] OrchestrationAgent rejected healing plan: {response.get('detail', 'Unknown error')}")
                    self.metrics['healing_plan_failures'] = self.metrics.get('healing_plan_failures', 0) + 1
        
            else:
                self.logger.warning(f"⚠️  OrchestrationAgent not available - healing plan stored locally only")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to send healing plan to OrchestrationAgent: {e}")

    async def _apply_feedback_modifications(self, request_id: str, suggested_changes: Dict[str, Any]):
        """Apply modifications suggested by FeedbackAgent"""
        try:
            if request_id in self.active_optimizations:
                request = self.active_optimizations[request_id]
                
                # Apply suggested constraint modifications
                if 'constraints' in suggested_changes:
                    request.constraints.update(suggested_changes['constraints'])
                    self.logger.info(f"Updated constraints for {request_id}: {suggested_changes['constraints']}")
                
                # Re-run optimization with updated parameters
                if suggested_changes.get('reoptimize', False):
                    self.logger.info(f"Re-running optimization for {request_id} with feedback modifications")
                    await self.optimization_queue.put(request)
                    
        except Exception as e:
            self.logger.error(f"Failed to apply feedback modifications: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Multi-Objective Optimization Agent with 3GPP Policy Integration')
    parser.add_argument('--config', default='optimization_agent_config.yaml',
                       help='Configuration file path')
    parser.add_argument(
        "--mode",
        choices=["rule_based", "ai_based"],
        help=(
            "Override experiment.mode without editing YAML"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for a reproducible optimization run",
    )
    parser.add_argument('--policy', type=str,
                       help='3GPP policy text to apply (in quotes)')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    logging.getLogger(
        "OptimizationAgent"
    ).info(
        "Experiment seed: %d",
        args.seed,
    )

    agent = OptimizationAgent(
        args.config,
        control_mode_override=args.mode,
    )
    
    # Example usage with command-line policy
    if args.policy:
        print(f"Policy will be applied: {args.policy}")
    
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\nShutting down OptimizationAgent with 3GPP Policy Integration...")
