"""
Enhanced Multi-Objective Optimization Agent with 3GPP Policy Integration using FAISS
WITH ENHANCED ZEROMQ COMMUNICATION HANDLER
"""

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
import csv
import io
from pathlib import Path

os.makedirs(os.path.join(os.path.dirname(__file__), 'config'), exist_ok=True)
# Load environment variables from .env file
load_dotenv()

# Import the communication handler and message types
from optimizationCOMM import OptimizationCommunicationHandler, OptimizationMessageType, OptimizationMessage

# LLM Integration
import openai # or your preferred LLM client
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings('ignore')

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_agent.log'),
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

def export_execution_template(response: OptimizationResponse, output_dir: str = "output") -> Dict[str, str]:
    """
    Export execution template as JSON and CSV formats
    
    Args:
        response: OptimizationResponse containing the execution template
        output_dir: Directory to save the output files
        
    Returns:
        Dictionary with file paths of saved JSON and CSV files
    """
    import csv
    from io import StringIO
    
    if not response.execution_template:
        return {"error": "No execution template found in response"}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filenames with timestamp and request ID
    base_filename = f"execution_template_{response.request_id}_{int(response.timestamp)}"
    json_file = os.path.join(output_dir, f"{base_filename}.json")
    csv_file = os.path.join(output_dir, f"{base_filename}.csv")
    
    # 1. Export as JSON
    json_output = json.dumps(response.execution_template, indent=4, default=str)
    
    with open(json_file, 'w') as f:
        f.write(json_output)
    
    # 2. Flatten nested dictionary for CSV export
    def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict[str, str]:
        """Recursively flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                if all(isinstance(i, dict) for i in v):
                    # Handle list of dictionaries
                    for idx, item in enumerate(v):
                        items.extend(flatten_dict(item, f"{new_key}_{idx}", sep=sep).items())
                else:
                    # Handle list of primitives
                    items.append((new_key, '|'.join(str(i) for i in v)))
            else:
                items.append((new_key, str(v) if v is not None else ''))
        
        return dict(items)
    
    # Flatten the execution template
    flattened_data = flatten_dict(response.execution_template)
    
    # 3. Export as CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write headers
        writer.writerow(['Field', 'Value'])
        
        # Write data
        for key, value in flattened_data.items():
            writer.writerow([key, value])
    
    return {
        "json_file": json_file,
        "csv_file": csv_file,
        "json_content": json_output,
        "csv_rows": len(flattened_data)
    }

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
        self.initialize_llm()

    def initialize_llm(self):
        """Initialize LLM client based on configuration"""
        try:
            llm_type = self.config.get('type', 'openai')
            if llm_type == 'openai':
                # Get API key from environment variable
                api_key = os.getenv('OPENAI_API_KEY', self.config.get('api_key', ''))
                openai.api_key = api_key
                self.llm_client = openai
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
                "energy_savings_watts": solution.predicted_energy - request.predicted_energy,
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
        try:
            llm_response = self._generate_llm_insights(request, solution, energy_saving)
            base_template["llm_insights"] = llm_response
            execution_plan = self._extract_execution_plan(llm_response, request, solution)
            base_template["execution_plan"] = execution_plan
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            base_template["llm_insights"] = "LLM template generation unavailable"

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
            
            if self.config.get('type') == 'openai':
                response = self.llm_client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a 5G network optimization expert. Provide detailed, technical execution guidance for base station energy optimization."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                return response.choices[0].message.content
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
        self.context = zmq.Context()
        self.orchestrator_socket = None
        self.initialize_connection()

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

    def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to Orchestrator and get response"""
        try:
            command_str = json.dumps(command)
            self.orchestrator_socket.send_string(command_str)
            
            # Wait for response
            response_str = self.orchestrator_socket.recv_string()
            response = json.loads(response_str)
            
            self.logger.debug(f"Orchestrator command: {command['action']} -> {response.get('status', 'UNKNOWN')}")
            return response
            
        except zmq.Again:
            self.logger.error("Orchestrator command timeout")
            return {'status': 'TIMEOUT', 'detail': 'Command timeout'}
        except Exception as e:
            self.logger.error(f"Orchestrator command error: {e}")
            return {'status': 'ERROR', 'detail': str(e)}

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
    
    def __init__(self, qos_predictor: QoSPredictor, constraints: Dict[str, Any]):
        self.qos_predictor = qos_predictor
        self.constraints = constraints
        self.logger = logging.getLogger('MultiObjectiveOptimizer')
        
        # Enhanced optimization bounds based on real 5G parameters
        self.bounds = {
            'tx_power': (5.0, 43.0), # dBm - realistic 5G range
            'bandwidth': [5, 10, 15, 20, 25, 40, 50, 100], # MHz - 5G NR options
            'carrier_count': [1, 2, 3, 4], # Carrier aggregation
            'es_modes': [(0, 1)] * 6 # 6 energy saving modes
        }
        
        # Apply policy-based bounds if available
        if 'max_tx_power' in constraints:
            self.bounds['tx_power'] = (
                self.bounds['tx_power'][0], 
                min(self.bounds['tx_power'][1], constraints['max_tx_power'])
            )
        
        # Setup DEAP for genetic algorithm
        self.setup_genetic_algorithm()

    def setup_genetic_algorithm(self):
        """Setup DEAP genetic algorithm framework with enhanced parameters"""
        # Create fitness class for multi-objective optimization
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0)) # Energy, latency, packet loss
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # Enhanced gene generators
        self.toolbox.register("tx_power", np.random.uniform,
                            self.bounds['tx_power'][0], self.bounds['tx_power'][1])
        self.toolbox.register("bandwidth_idx", np.random.randint, 0, len(self.bounds['bandwidth']))
        self.toolbox.register("carrier_idx", np.random.randint, 0, len(self.bounds['carrier_count']))
        self.toolbox.register("es_mode", np.random.randint, 0, 2)
        
        # Individual and population generators
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                            (self.toolbox.tx_power, self.toolbox.bandwidth_idx,
                             self.toolbox.carrier_idx) + tuple([self.toolbox.es_mode] * 6), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Enhanced genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.adaptive_mutate)
        self.toolbox.register("select", tools.selNSGA2) # NSGA-II selection
        self.toolbox.register("evaluate", self.evaluate_individual)

    def adaptive_mutate(self, individual, generation=0):
        """Adaptive mutation based on generation"""
        mutation_rate = max(0.05, 0.3 - generation * 0.002) # Decreasing mutation rate
        
        # Mutate tx_power with adaptive step
        if np.random.random() < mutation_rate:
            step_size = max(0.5, 3.0 - generation * 0.02)
            individual[0] = np.clip(
                individual[0] + np.random.normal(0, step_size),
                self.bounds['tx_power'][0], self.bounds['tx_power'][1]
            )
        
        # Mutate bandwidth index
        if np.random.random() < mutation_rate:
            individual[1] = np.random.randint(0, len(self.bounds['bandwidth']))
        
        # Mutate carrier count index
        if np.random.random() < mutation_rate:
            individual[2] = np.random.randint(0, len(self.bounds['carrier_count']))
        
        # Mutate ES modes
        for i in range(3, 9):
            if np.random.random() < mutation_rate * 0.5: # Lower rate for ES modes
                individual[i] = 1 - individual[i]
        
        return individual,

    def decode_individual(self, individual: List) -> Dict[str, Any]:
        """Decode genetic individual to configuration parameters"""
        return {
            'tx_power': individual[0],
            'bandwidth': self.bounds['bandwidth'][int(individual[1])],
            'carrier_count': self.bounds['carrier_count'][int(individual[2])],
            'es_modes': [int(individual[i]) for i in range(3, 9)]
        }

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
        
        # Enhanced energy model
        base_energy = 25.0 # Base power for 5G gNB
        tx_energy = (10 ** (config['tx_power'] / 10)) * 0.001 # Convert dBm to Watts
        digital_processing = config['bandwidth'] * 0.4 + config['carrier_count'] * 8.0
        
        # Energy saving calculations
        es_savings = 0
        es_mode_savings = [3.0, 1.5, 2.0, 2.5, 1.0, 1.8] # Watts saved per mode
        for i, mode in enumerate(config['es_modes']):
            if mode and i < len(es_mode_savings):
                es_savings += es_mode_savings[i]
        
        total_energy = base_energy + tx_energy + digital_processing - es_savings
        
        # Apply constraint penalties (including policy-derived constraints)
        penalty_energy = 0
        penalty_latency = 0
        penalty_loss = 0
        
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
            cxpb=0.8, mutpb=0.4, # Higher mutation for exploration
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
            
            # Energy calculation
            base_energy = 25.0
            tx_energy = (10 ** (config['tx_power'] / 10)) * 0.001
            digital_processing = config['bandwidth'] * 0.4 + config['carrier_count'] * 8.0
            es_savings = sum([3.0, 1.5, 2.0, 2.5, 1.0, 1.8][i]
                           for i, mode in enumerate(config['es_modes']) if mode and i < 6)
            total_energy = base_energy + tx_energy + digital_processing - es_savings
            
            # Check constraint violations
            violations = {}
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

    def __init__(self, config_path: str = "optimization_agent_config.yaml"):
        self.logger = logging.getLogger('OptimizationAgent')
        self.config = self.load_config(config_path)
        self.agent_id = self.config['agent']['id']
        
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
        orchestrator_config = self.config.get('orchestrator', {})
        self.orchestrator_interface = OrchestratorInterface(orchestrator_config) if orchestrator_config.get('enabled', True) else None
        
        # Initialize LLM template generator
        llm_config = self.config.get('llm', {})
        self.template_generator = LLMTemplateGenerator(llm_config)
        
        # REPLACED: Enhanced ZeroMQ setup with dedicated communication handler
        self.communication = None
        
        # Internal state (keep existing)
        self.optimization_queue = asyncio.Queue()
        self.active_optimizations = {}
        
        # Performance tracking
        self.metrics = {
            'optimizations_completed': 0,
            'total_optimization_time': 0,
            'energy_savings_achieved': [],
            'constraint_violations': 0,
            'feedback_received': 0,    
            'feedback_applied': 0,  
            'policies_applied': 0, 
            'templates_exported': 0,
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
            'llm': {
                'type': 'openai', # Changed to use OpenAI with API key from .env
                'model_name': 'gpt-3.5-turbo',
                'api_key': '', # Will be loaded from environment
                'model_path': '' # For local models
            },
            'orchestrator': {
                'enabled': True,
                'orchestrator_host': 'localhost',
                'orchestrator_port': 5555,
                'auto_execute': True,
                'execution_timeout': 300,
                'retry_attempts': 3
            },
            'output': {
                'export_templates': True,
                'export_formats': ['json', 'csv'],
                'output_directory': 'optimization_outputs'
            },

            'communication': {
                'request_port': 5558,
                'pub_port': 5557,
                'feedback_host': 'localhost',
                'feedback_agent_port': 5559,
                'prediction_host': 'localhost', 
                'prediction_agent_port': 5556
            }
        }

    async def initialize_communication(self):
        """Initialize ZeroMQ communication using dedicated handler"""
        try:
            # Create communication handler with configuration
            comm_config = {
                'agent_id': self.agent_id,
                'request_port': self.config['communication']['request_port'],
                'pub_port': self.config['communication'].get('pub_port', 5557),
                'endpoints': {
                    'feedback_agent': {
                        'host': self.config['communication'].get('feedback_host', 'localhost'),
                        'validation_port': self.config['communication']['feedback_agent_port']
                    },
                    'prediction_agent': {
                        'host': self.config['communication'].get('prediction_host', 'localhost'),
                        'port': self.config['communication']['prediction_agent_port']
                    }
                }
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

    async def _handle_prediction_update_msg(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction update message via ZeroMQ"""
        try:
            bs_id = payload['bs_id']
            request_id = payload['request_id']
            
            if request_id in self.active_optimizations:
                optimization_request = self.active_optimizations[request_id]
                optimization_request.predicted_energy = payload['predicted_energy']
                optimization_request.confidence_score = payload['confidence_score']
                
                await self.optimization_queue.put(optimization_request)
                self.logger.debug(f"Received prediction update for {bs_id}")
                
            return {'status': 'processed'}
            
        except Exception as e:
            self.logger.error(f"Error handling prediction update: {e}")
            return {'error': str(e)}

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
            self.optimizer = MultiObjectiveOptimizer(self.qos_predictor, updated_constraints)
            
            # Perform multi-objective optimization using NSGA-II
            pareto_solutions = self.optimizer.optimize_genetic_algorithm(
                request.current_metrics,
                population_size=self.config['optimization']['population_size'],
                generations=self.config['optimization']['generations']
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
            
            # Execute optimization through orchestrator if enabled
            execution_results = {}
            if self.orchestrator_interface and self.config.get('orchestrator', {}).get('auto_execute', True):
                try:
                    self.logger.info(f"Executing optimization for {request.bs_id} via Orchestrator")
                    execution_results = self.orchestrator_interface.execute_optimization_plan(request.bs_id, best_solution)
                    self.logger.info(f"Orchestrator execution completed: {execution_results['status']}")
                    
                    # Update metrics based on execution results
                    if execution_results['status'] in ['SUCCESS', 'PARTIAL_SUCCESS']:
                        updated_metrics = self.orchestrator_interface.get_current_metrics(request.bs_id)
                        if updated_metrics:
                            best_solution.predicted_qos_metrics.update(updated_metrics)
                            
                except Exception as e:
                    self.logger.error(f"Orchestrator execution failed: {e}")
                    execution_results = {'status': 'ERROR', 'error': str(e)}
            
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
            qos_status.append("✓ Throughput maintained")
        else:
            qos_status.append("⚠️ Throughput may be impacted")
        
        if solution.predicted_qos_metrics.get('latency', 100) <= request.constraints.get('max_latency', 20):
            qos_status.append("✓ Latency within limits")
        else:
            qos_status.append("⚠️ Latency may increase")
        
        explanations.append(" | ".join(qos_status))
        
        # NEW: 3GPP Policy information
        if applied_policies:
            explanations.append(f"📋 Applied {len(applied_policies)} 3GPP policy constraints")
        
        # Constraint compliance
        if not solution.constraint_violations:
            explanations.append("✓ All 5G QoS constraints satisfied")
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
    
    async def export_template_outputs(self, response: OptimizationResponse, 
                                    export_formats: List[str] = ['json', 'csv']) -> Dict[str, Any]:
        """
        Export optimization response template in specified formats
        
        Args:
            response: OptimizationResponse to export
            export_formats: List of formats to export ('json', 'csv', or both)
            
        Returns:
            Dictionary containing export results and file paths
        """
        try:
            if not response.execution_template:
                self.logger.warning(f"No execution template found for request {response.request_id}")
                return {"status": "no_template", "message": "No execution template to export"}
            
            # Create output directory
            output_dir = os.path.join(os.path.dirname(__file__), 'optimization_outputs')
            os.makedirs(output_dir, exist_ok=True)
            
            export_results = export_execution_template(response, output_dir)
            
            # Update metrics
            self.metrics['templates_exported'] = self.metrics.get('templates_exported', 0) + 1
            
            self.logger.info(f"Exported execution template for {response.request_id}: "
                            f"JSON ({export_results.get('json_file', 'N/A')}), "
                            f"CSV ({export_results.get('csv_file', 'N/A')})")
            
            return {
                "status": "success",
                "request_id": response.request_id,
                "exported_files": export_results,
                "template_summary": {
                    "energy_saving_percent": response.energy_saving_percent,
                    "optimization_method": response.optimization_method,
                    "processing_time_ms": response.processing_time_ms,
                    "confidence": response.confidence
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export template for {response.request_id}: {e}")
            return {"status": "error", "error": str(e)}

    async def process_optimization_queue(self):
        """Process optimization requests from the queue"""
        while True:
            try:
                request = await self.optimization_queue.get()
                response = await self.process_optimization_request(request)
                
                # Send response via communication handler to FeedbackAgent
                await self.communication.send_to_feedback_agent(
                    OptimizationMessageType.OPTIMIZATION_RESPONSE,
                    asdict(response)
                )
                
                # Export execution template as JSON and CSV if enabled
                if self.config.get('export_templates', True):
                    if response.execution_template:
                        try:
                            export_result = await self.export_template_outputs(response)
                            self.logger.debug(f"Template export result: {export_result['status']}")
                            
                            # Add export info to response template
                            response.execution_template['export_info'] = {
                                'exported': export_result['status'] == 'success',
                                'export_timestamp': datetime.now().isoformat(),
                                'files': export_result.get('exported_files', {})
                            }
                            
                        except Exception as e:
                            self.logger.warning(f"Template export failed for {request.bs_id}: {e}")
                
                self.optimization_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing optimization from queue: {e}")
                await asyncio.sleep(1)




    # REPLACED: async task handling in run() method
    async def run(self):
        """Main agent execution loop with enhanced communication"""
        self.logger.info(f"Starting OptimizationAgent {self.agent_id} with Enhanced ZeroMQ Communication")
        
        try:
            await self.initialize_communication()
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
    async def _handle_feedback_response_msg(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle feedback response from FeedbackAgent"""
        try:
            request_id = payload.get('request_id')
            validation_result = payload.get('validation_result')
            feedback_data = payload.get('feedback_data', {})
            
            self.logger.info(f"Received feedback for request {request_id}: {validation_result}")
            
            # Process feedback based on validation result
            if validation_result == 'APPROVED':
                self.logger.info(f"Optimization plan approved for request {request_id}")
                # Execute approved plan or update status
                
            elif validation_result == 'REJECTED':
                self.logger.warning(f"Optimization plan rejected for request {request_id}: {feedback_data.get('reason', 'Unknown')}")
                # Handle rejection - maybe retry with different parameters
                
            elif validation_result == 'MODIFIED':
                self.logger.info(f"Optimization plan modified for request {request_id}")
                # Apply suggested modifications
                suggested_changes = feedback_data.get('suggested_changes', {})
                await self._apply_feedback_modifications(request_id, suggested_changes)
            
            # Update metrics
            self.metrics['feedback_received'] = self.metrics.get('feedback_received', 0) + 1
            
            return {'status': 'processed', 'request_id': request_id}
            
        except Exception as e:
            self.logger.error(f"Error handling feedback response: {e}")
            return {'error': str(e)}

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
    parser.add_argument('--policy', type=str,
                       help='3GPP policy text to apply (in quotes)')
    
    args = parser.parse_args()
    
    agent = OptimizationAgent(args.config)
    
    # Example usage with command-line policy
    if args.policy:
        print(f"Policy will be applied: {args.policy}")
    
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\nShutting down OptimizationAgent with 3GPP Policy Integration...")
