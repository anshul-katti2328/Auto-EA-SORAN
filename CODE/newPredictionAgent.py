"""
Prediction Agent for Auto EA-SORAN System - Keras Backend
Migrated from PyTorch to Keras/TensorFlow
"""

import asyncio
import zmq
import zmq.asyncio
import numpy as np
import pandas as pd
import json
import logging
import time
import uuid
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import yaml
from dataclasses import dataclass, asdict
import warnings

from inferenceENGINE import KerasEnergyPredictor  # ← FIXED IMPORT

warnings.filterwarnings('ignore')

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction_agent_keras.log'),
        logging.StreamHandler()
    ]
)


@dataclass
class PredictionRequest:
    """Data structure for prediction requests"""
    request_id: str
    timestamp: float
    bs_id: str
    metrics: Dict[str, Any]
    config_params: Dict[str, Any]


@dataclass
class PredictionResponse:
    """Data structure for prediction responses"""
    request_id: str
    timestamp: float
    bs_id: str
    predicted_energy: float
    confidence_score: float
    feature_importance: Dict[str, float]
    explanation: str
    processing_time_ms: float


class PredictionAgent:
    """
    Keras-based Prediction Agent for 5G Energy Optimization
    Receives unseen data from Simu5G via ZMQ and makes real-time predictions
    """
    
    def __init__(self, config_path: str = "prediction_agent_config.yaml"):
        self.logger = logging.getLogger('PredictionAgent')
        self.config = self.load_config(config_path)
        self.agent_id = self.config['agent']['id']
        
        # Communication config
        comm_cfg = self.config.get('communication', {})
        self.metrics_sub_endpoint = comm_cfg.get('metrics_sub_endpoint', 'tcp://127.0.0.1:5556')
        self.metrics_topic = ""  # Subscribe to ALL topics
        self.metrics_sub = None
        
        # Initialize Keras predictor
        self.predictor = None
        self.is_model_loaded = False
        
        # SHAP explainability with batching
        self.use_shap = self.config.get('features', {}).get('use_shap', False)
        self.shap_batch_size = self.config.get('features', {}).get('shap_batch_size', 5)
        self.shap_batch_buffer = []
        self.shap_plots_generated = 0
        
        # Store recent samples for SHAP analysis
        self.recent_samples = []
        self.max_recent_samples = 100
        
        # Check if model exists
        model_path = self.config['paths']['model_checkpoint']
        feature_path = self.config['paths']['feature_engineer']
        
        if not os.path.exists(model_path) or not os.path.exists(feature_path):
            self.logger.warning("Keras model not found. Please train using kerasTRAINER.py first")
            raise FileNotFoundError(f"Model files not found: {model_path}, {feature_path}")
        
        # ZeroMQ setup
        self.context = zmq.asyncio.Context()
        self.request_socket = None
        self.response_socket = None
        
        # Performance tracking
        self.metrics = {
            'requests_processed': 0,
            'total_processing_time': 0,
            'errors': 0,
            'predictions_made': 0,
            'shap_batches_processed': 0,
            'start_time': time.time()
        }
        
        self.logger.info(f"PredictionAgent {self.agent_id} initialized with Keras backend")
        if self.use_shap:
            self.logger.info(f"  - SHAP enabled: Generating plots every {self.shap_batch_size} requests")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Default configuration for Keras backend"""
        return {
            'agent': {
                'id': 'prediction_agent_keras_001',
                'name': 'Keras Energy Prediction Agent'
            },
            'communication': {
                'request_port': 5555,
                'response_port': 5557,
                'optimization_agent_port': 5558,
                'metrics_sub_endpoint': 'tcp://127.0.0.1:5556'
            },
            'paths': {
                'model_checkpoint': 'models/energy_prediction_keras.keras',
                'feature_engineer': 'models/feature_engineer_keras.pkl'
            },            'features': {
                'use_shap': False,
                'shap_batch_size': 5
            }
        }
    
    async def initialize_communication(self):
        """Initialize ZeroMQ communication sockets"""
        try:
            # Socket for receiving prediction requests (REP)
            self.request_socket = self.context.socket(zmq.REP)
            self.request_socket.bind(f"tcp://*:{self.config['communication']['request_port']}")
            
            # Socket for sending responses to optimization agent (PUSH)
            self.response_socket = self.context.socket(zmq.PUSH)
            self.response_socket.connect(
                f"tcp://localhost:{self.config['communication']['optimization_agent_port']}"
            )
            
            # Socket for receiving Simu5G metrics (SUB)
            self.metrics_sub = self.context.socket(zmq.SUB)
            self.metrics_sub.connect(self.metrics_sub_endpoint)
            self.metrics_sub.setsockopt_string(zmq.SUBSCRIBE, self.metrics_topic)
            
            self.logger.info(f"Communication initialized:")
            self.logger.info(f"  - Request port: {self.config['communication']['request_port']}")
            self.logger.info(f"  - Response port: {self.config['communication']['optimization_agent_port']}")
            self.logger.info(f"  - Metrics SUB: {self.metrics_sub_endpoint}")
            self.logger.info(f"  - Listening for Simu5G data on: {self.metrics_sub_endpoint}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise
    
    def load_model(self):
        """Load Keras model and feature engineer"""
        try:
            self.predictor = KerasEnergyPredictor(
                model_path=self.config['paths']['model_checkpoint'],
                feature_engineer_path=self.config['paths']['feature_engineer'],
                enable_shap=self.use_shap
            )
            self.is_model_loaded = True
            self.logger.info("✓ Keras model loaded successfully")
            
            if self.use_shap:
                self.logger.info("✓ SHAP explainability enabled")
            
        except Exception as e:
            self.logger.error(f"Failed to load Keras model: {e}")
            raise
    
    def prepare_dataframe(self, request: PredictionRequest) -> pd.DataFrame:
        """
        Convert request to DataFrame format expected by feature engineer
        Handles unseen Simu5G data format with column name mapping and defaults
        """
        # Merge metrics and config params
        data = {**request.metrics, **request.config_params}
          # ✅ CRITICAL FIX: Map Simu5G column names to model-expected names
        # This fixes the constant prediction issue by extracting actual varying data
        column_mapping = {
            # Network load/utilization metrics
            'prb_utilization': 'load',
            'prb_util': 'load',
            'prb_util_pct': 'load',  # ← ADD: TwinMetricsAgent sends this!
            'utilization': 'load',
            'resource_utilization': 'load',
            'load_percent': 'load',
            
            # Power metrics
            'tx_power_dbm': 'TXpower',
            'txpower_dbm': 'TXpower',
            'tx_power': 'TXpower',
            'power_dbm': 'TXpower',
            'transmit_power': 'TXpower',
            
            # Bandwidth metrics
            'bandwidth_mhz': 'Bandwidth',
            'bw_mhz': 'Bandwidth',
            'channel_bandwidth': 'Bandwidth',
            'bw': 'Bandwidth',
            
            # Frequency metrics
            'frequency_mhz': 'Frequency',
            'freq_mhz': 'Frequency',
            'carrier_frequency': 'Frequency',
            'center_frequency': 'Frequency',
            'freq': 'Frequency',
            
            # Antenna configuration
            'antenna_count': 'Antennas',
            'num_antennas': 'Antennas',
            'antenna_elements': 'Antennas',
            
            # Radio unit and mode
            'ru_type': 'RUType',
            'radio_unit_type': 'RUType',
            'duplex_mode': 'Mode',
            'tdd_mode': 'Mode',
            'fdd_mode': 'Mode',
            
            # Base station identifier
            'bs_id': 'BS',
            'base_station_id': 'BS',
            'gnb_id': 'BS',
            'cell_id': 'CellName',
              # Energy saving modes
            'es_mode_1': 'ESMode1',
            'es_mode_2': 'ESMode2',
            'es_mode_3': 'ESMode3',
            'es_mode_4': 'ESMode4',
            'es_mode_5': 'ESMode5',
            'es_mode_6': 'ESMode6',
              # Simu5G TwinMetricsAgent specific fields
            'dl_mbps': 'downlink_throughput',
            'ul_mbps': 'uplink_throughput',
            'active_ues': 'num_users',
            'avg_sinr_db': 'sinr',
            'avg_rsrp_dbm': 'rsrp',
        }
        
        # Apply column name mapping
        mapped_data = {}
        for orig_key, value in data.items():
            # Use mapped name if exists, otherwise keep original
            mapped_key = column_mapping.get(orig_key, orig_key)
            mapped_data[mapped_key] = value
            if orig_key in column_mapping:
                self.logger.debug(f"Mapped '{orig_key}' → '{mapped_key}' = {value}")
        
        data = mapped_data
        
        # ✅ SIMU5G SPECIFIC: Convert PRB percentage to decimal load
        if 'load' in data and data['load'] > 1.0:
            # PRB is sent as percentage (0-100), convert to decimal (0-1)
            data['load'] = data['load'] / 100.0
            self.logger.debug(f"Converted PRB percentage to decimal load: {data['load']:.3f}")
        
        # ✅ SIMU5G SPECIFIC: Parse es_mode string to ESMode flags
        if 'es_mode' in data:
            es_mode_str = str(data['es_mode']).lower()
            # Initialize all ES modes to 0
            for i in range(1, 7):
                data[f'ESMode{i}'] = 0
            
            # Map string to flags (extend this based on your ES mode definitions)
            if es_mode_str in ['sleep', 'deep_sleep']:
                data['ESMode1'] = 1  # Example: Mode1 = sleep
            elif es_mode_str in ['light_sleep', 'dtx']:
                data['ESMode2'] = 1  # Example: Mode2 = light sleep
            # Add more mappings as needed
            
            self.logger.debug(f"Parsed es_mode '{es_mode_str}' to ESMode flags")
            del data['es_mode']  # Remove the string version
        
        # Log what was actually received BEFORE applying defaults
        self.logger.info(f"📥 Received data from Simu5G: {list(data.keys())}")
        
        # Ensure required temporal columns
        if 'Time' not in data:
            data['Time'] = datetime.fromtimestamp(request.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # Ensure BS identifier
        if 'BS' not in data and 'bs_id' not in data:
            data['BS'] = request.bs_id
        elif 'bs_id' in data and 'BS' not in data:
            data['BS'] = data['bs_id']
        
        # Add default values for critical features if missing
        # These are typical 5G base station parameters
        defaults = {
            'load': 0.5,  # Default network load (50%)
            'ESMode1': 0,  # Energy saving modes (disabled by default)
            'ESMode2': 0,
            'ESMode3': 0,
            'ESMode4': 0,
            'ESMode5': 0,
            'ESMode6': 0,
            'Frequency': 3500,  # Default 5G n78 band frequency (MHz)
            'Bandwidth': 100,   # Default 5G bandwidth (MHz)
            'Antennas': 64,     # Default massive MIMO antenna count
            'TXpower': 43.0,    # Default TX power (dBm) for macro cell
            'RUType': 'Type1',  # Default Radio Unit type (matches training data)
            'Mode': 'Mode1',    # Default duplex mode (matches training data)
            'w': 1.0,           # Default weight
        }
        
        # Apply defaults only for missing keys
        defaults_used = []
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
                defaults_used.append(f"{key}={default_value}")
        
        if defaults_used:
            self.logger.warning(f"⚠️  Using defaults for missing columns: {', '.join(defaults_used)}")
        
        # ✅ Log actual feature values to diagnose predictions
        self.logger.info(f"📊 Final feature values for prediction:")
        self.logger.info(f"   load={data.get('load', 'N/A')}, TXpower={data.get('TXpower', 'N/A')}")
        self.logger.info(f"   Bandwidth={data.get('Bandwidth', 'N/A')}, Frequency={data.get('Frequency', 'N/A')}")
        self.logger.info(f"   Antennas={data.get('Antennas', 'N/A')}, RUType={data.get('RUType', 'N/A')}, Mode={data.get('Mode', 'N/A')}")
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        return df
    
    async def process_prediction_request(self, request: PredictionRequest) -> PredictionResponse:
        """Process a single prediction request using Keras model on unseen data"""
        start_time = time.time()
        
        try:
            # Convert to DataFrame
            df = self.prepare_dataframe(request)
            
            # Store for SHAP analysis
            self.recent_samples.append(df)
            if len(self.recent_samples) > self.max_recent_samples:
                self.recent_samples.pop(0)
              # Make prediction on unseen data
            predictions, confidence = self.predictor.predict(df, return_confidence=True)
            prediction = float(predictions[0])
            
            # Handle confidence - may be None if not implemented
            if confidence is not None:
                confidence_score = float(confidence[0])
                explanation = f"Predicted energy: {prediction:.2f}W (confidence: {confidence_score*100:.1f}%)"
            else:
                # No confidence available from model
                confidence_score = None
                explanation = f"Predicted energy: {prediction:.2f}W (confidence: unavailable)"
            
            # Feature importance (starts empty - honest approach!)
            feature_importance = {}
            
            # SHAP explanation (batched)
            if self.use_shap and self.predictor.shap_explainer is not None:
                # Add to batch buffer
                self.shap_batch_buffer.append(df)
                
                # Generate SHAP when buffer reaches batch size
                if len(self.shap_batch_buffer) >= self.shap_batch_size:
                    try:
                        batch_num = self.metrics['shap_batches_processed'] + 1
                        self.logger.info("=" * 80)
                        self.logger.info(f"📊 GENERATING SHAP BATCH #{batch_num} ({len(self.shap_batch_buffer)} samples)")
                        self.logger.info("=" * 80)
                        
                        # Combine buffered samples
                        df_batch = pd.concat(self.shap_batch_buffer, ignore_index=True)
                        
                        # Generate SHAP plots
                        timestamp = int(time.time())
                        shap_result = self.predictor.explain_prediction(
                            df_batch, 
                            plot=True, 
                            save_path=f"shap_batch_{batch_num}_{timestamp}.png",
                            n_samples=len(df_batch)
                        )
                        
                        # Extract global feature importance
                        if 'feature_importance' in shap_result:
                            top_features = shap_result['feature_importance'].head(5)
                            feature_importance = dict(zip(
                                top_features['feature'], 
                                top_features['importance']
                            ))
                            
                            self.logger.info(f"✓ Top 5 features for batch #{batch_num}:")
                            for feat, imp in feature_importance.items():
                                self.logger.info(f"    • {feat}: {imp:.3f}")
                        
                        # Update explanation for current request
                        if feature_importance:
                            explanation += f" | Key drivers: {', '.join(list(feature_importance.keys())[:3])}"
                        
                        self.logger.info(f"✓ SHAP plots saved with prefix: shap_batch_{batch_num}_{timestamp}")
                        self.logger.info("=" * 80)
                        
                        # Clear buffer and update counter
                        self.shap_batch_buffer = []
                        self.metrics['shap_batches_processed'] = batch_num
                        self.shap_plots_generated += 1
                        
                    except Exception as e:
                        self.logger.error(f"SHAP calculation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        self.shap_batch_buffer = []
                else:
                    # Log progress toward next batch
                    remaining = self.shap_batch_size - len(self.shap_batch_buffer)
                    self.logger.debug(f"SHAP buffer: {len(self.shap_batch_buffer)}/{self.shap_batch_size} "
                                     f"({remaining} more requests until next batch)")
            
            # Fallback feature importance if SHAP not generated yet
            # DO NOT hardcode fake values - return empty dict for transparency!
            if not feature_importance:
                feature_importance = {}  # ← HONEST! Shows SHAP not ready
                
                # Add informative explanation about SHAP status
                if self.use_shap:
                    if len(self.shap_batch_buffer) > 0:
                        explanation += f" | ⏳ SHAP pending ({len(self.shap_batch_buffer)}/{self.shap_batch_size} samples)"
                    else:
                        explanation += " | ⚠️ SHAP initializing..."
                else:
                    explanation += " | ℹ️ SHAP disabled"
            
            # Create response
            processing_time = (time.time() - start_time) * 1000
            response = PredictionResponse(
                request_id=request.request_id,
                timestamp=time.time(),
                bs_id=request.bs_id,
                predicted_energy=prediction,
                confidence_score=confidence_score,
                feature_importance=feature_importance,
                explanation=explanation,
                processing_time_ms=processing_time
            )
              # Update metrics
            self.metrics['requests_processed'] += 1
            self.metrics['predictions_made'] += 1
            self.metrics['total_processing_time'] += processing_time
            
            conf_str = f"{confidence_score:.3f}" if confidence_score is not None else "N/A"
            self.logger.info(
                f"✓ Prediction for {request.bs_id}: {prediction:.2f}W "
                f"(confidence: {conf_str}, time: {processing_time:.2f}ms)"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to process prediction: {e}", exc_info=True)
            self.metrics['errors'] += 1
            
            # Return error response
            return PredictionResponse(
                request_id=request.request_id,
                timestamp=time.time(),
                bs_id=request.bs_id,
                predicted_energy=-1.0,
                confidence_score=0.0,
                feature_importance={},
                explanation=f"Prediction failed: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def run(self):
        """Main agent execution loop - processes unseen Simu5G data"""
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING PREDICTION AGENT {self.agent_id} (Keras Backend)")
        self.logger.info("=" * 80)
        
        try:
            # Initialize
            await self.initialize_communication()
            self.load_model()
            
            self.logger.info("=" * 80)
            self.logger.info("✓ PREDICTION AGENT READY")
            self.logger.info("  Waiting for unseen data from Simu5G...")
            self.logger.info("=" * 80)
            
            # Initialize poller
            poller = zmq.asyncio.Poller()
            poller.register(self.metrics_sub, zmq.POLLIN)
            poller.register(self.request_socket, zmq.POLLIN)
            
            while True:
                try:
                    # Poll for messages (10ms timeout)
                    events = dict(await poller.poll(timeout=10))
                    
                    # 1) Handle Simu5G metrics (UNSEEN DATA from simulation)
                    if self.metrics_sub in events:
                        frames = await self.metrics_sub.recv_multipart()
                        
                        # Parse message
                        if len(frames) == 1:
                            topic = ""
                            payload = json.loads(frames[0].decode("utf-8"))
                        else:
                            topic = frames[0].decode("utf-8")
                            payload = json.loads(frames[1].decode("utf-8"))
                        
                        cells = payload.get("cells", [])
                        twin_id = payload.get("twin_id", "simu5g-twin")
                        sim_time = payload.get("sim_time_s")
                        
                        self.logger.info(f"📥 Received data from Simu5G: {len(cells)} cells at sim_time={sim_time}")
                        
                        # Process each cell (UNSEEN DATA)
                        for cell in cells:
                            req = PredictionRequest(
                                bs_id=str(
                                    cell.get("bs_id") or 
                                    cell.get("BS") or 
                                    cell.get("cell_id") or 
                                    "unknown"
                                ),
                                metrics=cell,  # ← UNSEEN DATA FROM SIMU5G
                                config_params={},
                                request_id=f"simu5g-{uuid.uuid4().hex[:8]}",
                                timestamp=time.time()
                            )
                            
                            # Make prediction on unseen data
                            resp = await self.process_prediction_request(req)
                            
                            # Send prediction to optimization agent
                            resp_flat = asdict(resp)
                            payload = {
                                "request_id": resp_flat["request_id"],
                                "bs_id": resp_flat["bs_id"],
                                "predicted_energy": resp_flat["predicted_energy"],
                                "confidence": resp_flat["confidence_score"],
                                "twin_id": twin_id,
                                "sim_time_s": sim_time,
                            }
                            
                            enveloped = {
                                "type": "prediction",
                                "timestamp": resp_flat["timestamp"],
                                "processing_time_ms": resp_flat["processing_time_ms"],
                                "payload": payload,
                            }
                            
                            await self.response_socket.send_json(enveloped)
                            self.logger.debug(f"📤 Sent prediction to optimization agent")
                    
                    # 2) Handle direct requests (REP socket - for testing)
                    if self.request_socket in events:
                        msg = await self.request_socket.recv_json()
                        
                        # Backfill required fields
                        if "request_id" not in msg:
                            msg["request_id"] = f"ext-{uuid.uuid4().hex[:8]}"
                        if "timestamp" not in msg:
                            msg["timestamp"] = time.time()
                        if "bs_id" not in msg:
                            m = msg.get("metrics", {}) or {}
                            msg["bs_id"] = str(
                                m.get("bs_id") or 
                                m.get("BS") or 
                                m.get("cell_id") or 
                                "unknown"
                            )
                        
                        # Process request
                        req = PredictionRequest(**msg)
                        resp = await self.process_prediction_request(req)
                        
                        # Send replies
                        await self.request_socket.send_json(asdict(resp))  # REP reply
                        await self.response_socket.send_json(asdict(resp))  # PUSH to optimizer
                    
                    # Light yield if nothing arrived
                    if not events:
                        await asyncio.sleep(0.01)
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}", exc_info=True)
                    await asyncio.sleep(0.05)
        
        finally:
            # Cleanup
            self.logger.info("\n" + "=" * 80)
            self.logger.info("SHUTTING DOWN PREDICTION AGENT")
            self.logger.info("=" * 80)
            
            if self.request_socket:
                self.request_socket.close()
            if self.response_socket:
                self.response_socket.close()
            if self.metrics_sub:
                self.metrics_sub.close()
            
            self.context.term()
            
            # Log final statistics
            uptime = time.time() - self.metrics['start_time']
            self.logger.info(f"\nFinal Statistics:")
            self.logger.info(f"  - Uptime: {uptime:.2f}s")
            self.logger.info(f"  - Requests processed: {self.metrics['requests_processed']}")
            self.logger.info(f"  - Predictions made: {self.metrics['predictions_made']}")
            self.logger.info(f"  - SHAP batches generated: {self.metrics['shap_batches_processed']}")
            self.logger.info(f"  - Errors: {self.metrics['errors']}")
            avg_time = self.metrics['total_processing_time']/max(1, self.metrics['requests_processed'])
            self.logger.info(f"  - Avg processing time: {avg_time:.2f}ms")
            self.logger.info("=" * 80)
            self.logger.info("✓ PredictionAgent stopped")


# ==================== CLI Interface ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Keras Energy Prediction Agent')
    parser.add_argument(
        '--config', 
        default='prediction_agent_config.yaml',
        help='Configuration file path'
    )
    
    args = parser.parse_args()
    
    agent = PredictionAgent(args.config)
    
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\n✓ Shutting down PredictionAgent (Keras)...")