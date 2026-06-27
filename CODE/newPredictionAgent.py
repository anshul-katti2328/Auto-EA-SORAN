"""
Prediction Agent for Auto EA-SORAN System - Keras Backend
Migrated from PyTorch to Keras/TensorFlow
"""
import sys
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
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
import yaml
from dataclasses import dataclass, asdict
import warnings
from pathlib import Path
if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

SIMULATION_EPOCH = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()

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
        shap_output_setting = self.config.get(
            "paths",
            {},
        ).get(
            "shap_output_dir",
            "../OUTPUTs/shap",
        )

        self.shap_output_dir = Path(
            shap_output_setting
        ).expanduser()

        if not self.shap_output_dir.is_absolute():
            self.shap_output_dir = (
                Path(__file__).resolve().parent
                / self.shap_output_dir
            ).resolve()
        
        # Store recent samples for SHAP analysis
        self.recent_samples = []
        self.max_recent_samples = 100
        
        # Resolve trained-model artifacts independently of the directory from which this agent is launched.
        def resolve_required_path(path_value: str) -> Path:
            candidate = Path(path_value).expanduser()

            if candidate.is_absolute():
                return candidate

            cwd_candidate = (
                Path.cwd() / candidate
            ).resolve()
            if cwd_candidate.is_file():
                return cwd_candidate

            return (
                Path(__file__).resolve().parent
                / candidate
            ).resolve()

        model_path = resolve_required_path(
            self.config["paths"]["model_checkpoint"]
        )
        feature_path = resolve_required_path(
            self.config["paths"]["feature_engineer"]
        )
        hardware_profile_path = resolve_required_path(
            self.config["paths"]["hardware_profiles"]
        )
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Keras model not found: {model_path}"
            )

        if not feature_path.is_file():
            raise FileNotFoundError(
                "Feature engineer not found: "
                f"{feature_path}"
            )
        if not hardware_profile_path.is_file():
            raise FileNotFoundError(
                "Hardware profiles not found: "
                f"{hardware_profile_path}"
            )

        self.hardware_profiles = pd.read_csv(
            hardware_profile_path
        )

        required_profile_columns = {
            "BS",
            "CellName",
            "RUType",
            "Mode",
            "Frequency",
            "Bandwidth",
            "Antennas",
            "TXpower",
        }

        missing_profile_columns = (
            required_profile_columns
            - set(self.hardware_profiles.columns)
        )

        if missing_profile_columns:
            raise ValueError(
                "Hardware profile file is missing columns: "
                f"{sorted(missing_profile_columns)}"
            )

        self.hardware_profiles = (
            self.hardware_profiles[
                self.hardware_profiles["CellName"].astype(str)
                == "Cell0"
            ]
            .drop_duplicates(subset=["BS"])
            .set_index("BS")
        )

        required_profile_ids = {
            f"B_{index}"
            for index in range(30)
        }

        missing_profile_ids = (
            required_profile_ids
            - set(self.hardware_profiles.index)
        )

        if missing_profile_ids:
            raise ValueError(
                "Missing hardware profiles for simulated gNBs: "
                f"{sorted(missing_profile_ids)}"
            )
        
        self.config["paths"]["model_checkpoint"] = str(
            model_path
        )
        self.config["paths"]["feature_engineer"] = str(
            feature_path
        )
        self.config["paths"]["hardware_profiles"] = str(
            hardware_profile_path
        )

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
            'batches_processed': 0,
            'total_batch_processing_time': 0.0,
            'stale_snapshots_dropped': 0,            
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
    def get_model_hardware_profile(
        self,
        bs_id: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            profile_number = int(str(bs_id)) - 1001
        except (TypeError, ValueError):
            return None

        if profile_number < 0 or profile_number >= 30:
            return None

        profile_id = f"B_{profile_number}"
        profile = self.hardware_profiles.loc[profile_id]

        return {
            "model_profile_id": profile_id,
            "RUType": str(profile["RUType"]),
            "Mode": str(profile["Mode"]),
            "Frequency": float(profile["Frequency"]),
            "Bandwidth": float(profile["Bandwidth"]),
            "Antennas": int(profile["Antennas"]),
            "TXpower": float(profile["TXpower"]),
        }

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

            'mode': 'Mode',
            'wall_time_utc': 'Time',
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
        hardware_profile = (
            self.get_model_hardware_profile(
                request.bs_id
            )
        )

        if hardware_profile is not None:
            profile_id = hardware_profile.pop(
                "model_profile_id"
            )
            data.update(hardware_profile)

            self.logger.info(
                "Using ITU hardware profile %s for gNB %s",
                profile_id,
                request.bs_id,
            )        
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
            if self.use_shap:
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
                            save_path=str(
                                self.shap_output_dir
                                / (
                                    f"shap_batch_{batch_num}_"
                                    f"{timestamp}.png"
                                )
                            ),
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
    async def process_prediction_batch(
        self,
        requests: List[PredictionRequest],
    ) -> List[PredictionResponse]:
        """Predict all cells from one Simu5G snapshot in one Keras batch."""

        if not requests:
            return []

        # Inline SHAP is intentionally incompatible with the real-time
        # batching path. Paper SHAP is generated offline.
        if self.use_shap:
            self.logger.warning(
                "Live SHAP disables batch acceleration; "
                "processing requests individually"
            )
            return [
                await self.process_prediction_request(request)
                for request in requests
            ]

        start_time = time.time()

        try:
            dataframes = [
                self.prepare_dataframe(request)
                for request in requests
            ]

            batch_dataframe = pd.concat(
                dataframes,
                ignore_index=True,
            )

            self.recent_samples.extend(dataframes)
            self.recent_samples = self.recent_samples[
                -self.max_recent_samples:
            ]

            predictions, confidence = self.predictor.predict(
                batch_dataframe,
                return_confidence=True,
            )

            predictions = np.asarray(
                predictions,
                dtype=float,
            ).reshape(-1)

            if len(predictions) != len(requests):
                raise ValueError(
                    "Batch prediction count does not match "
                    f"request count: {len(predictions)} != "
                    f"{len(requests)}"
                )

            if confidence is None:
                confidence_values = np.full(
                    len(requests),
                    np.nan,
                    dtype=float,
                )
            else:
                confidence_values = np.asarray(
                    confidence,
                    dtype=float,
                ).reshape(-1)

                if len(confidence_values) != len(requests):
                    raise ValueError(
                        "Batch confidence count does not match "
                        f"request count: {len(confidence_values)} != "
                        f"{len(requests)}"
                    )

            batch_time_ms = (
                time.time() - start_time
            ) * 1000.0

            responses = []

            for index, request in enumerate(requests):
                prediction = float(predictions[index])
                confidence_value = confidence_values[index]

                if np.isfinite(confidence_value):
                    confidence_score = float(confidence_value)
                    explanation = (
                        f"Predicted energy: {prediction:.2f}W "
                        f"(confidence: "
                        f"{confidence_score * 100.0:.1f}%)"
                    )
                else:
                    confidence_score = None
                    explanation = (
                        f"Predicted energy: {prediction:.2f}W "
                        "(confidence unavailable)"
                    )

                responses.append(
                    PredictionResponse(
                        request_id=request.request_id,
                        timestamp=time.time(),
                        bs_id=request.bs_id,
                        predicted_energy=prediction,
                        confidence_score=confidence_score,
                        feature_importance={},
                        explanation=explanation,
                        processing_time_ms=batch_time_ms,
                    )
                )

            request_count = len(requests)

            self.metrics["requests_processed"] += request_count
            self.metrics["predictions_made"] += request_count
            self.metrics["batches_processed"] += 1
            self.metrics[
                "total_batch_processing_time"
            ] += batch_time_ms

            # Each cell waits for the complete batch result, so record
            # batch latency for every response.
            self.metrics["total_processing_time"] += (
                batch_time_ms * request_count
            )

            self.logger.info(
                "Batch prediction complete: %d cells in %.2f ms "
                "(%.2f ms/cell effective compute)",
                request_count,
                batch_time_ms,
                batch_time_ms / request_count,
            )

            return responses

        except Exception as error:
            self.logger.error(
                "Batch prediction failed: %s",
                error,
                exc_info=True,
            )
            self.metrics["errors"] += len(requests)

            elapsed_ms = (
                time.time() - start_time
            ) * 1000.0

            return [
                PredictionResponse(
                    request_id=request.request_id,
                    timestamp=time.time(),
                    bs_id=request.bs_id,
                    predicted_energy=-1.0,
                    confidence_score=0.0,
                    feature_importance={},
                    explanation=(
                        f"Batch prediction failed: {error}"
                    ),
                    processing_time_ms=elapsed_ms,
                )
                for request in requests
            ]

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
                        # Drain queued telemetry and retain only the
                        # newest complete snapshot. Each snapshot still
                        # contains every base station.
                        stale_snapshots = 0

                        while True:
                            try:
                                newer_frames = await (
                                    self.metrics_sub.recv_multipart(
                                        flags=zmq.NOBLOCK
                                    )
                                )
                                frames = newer_frames
                                stale_snapshots += 1
                            except zmq.Again:
                                break

                        if stale_snapshots:
                            self.metrics[
                                "stale_snapshots_dropped"
                            ] += stale_snapshots

                            self.logger.warning(
                                "Discarded %d stale telemetry "
                                "snapshots; processing newest",
                                stale_snapshots,
                            )

                        # Parse the newest snapshot.
                        if len(frames) == 1:
                            topic = ""
                            payload = json.loads(
                                frames[0].decode("utf-8")
                            )
                        else:
                            topic = frames[0].decode("utf-8")
                            payload = json.loads(
                                frames[1].decode("utf-8")
                            )

                        cells = payload.get("cells", [])
                        twin_id = payload.get(
                            "twin_id", "simu5g-twin"
                        )
                        sim_time = payload.get("sim_time_s")
                        physical_sim_time = payload.get(
                            "physical_sim_time_s"
                        )
                        kpi_source = payload.get(
                            "kpi_source",
                            "unknown",
                        )
                        energy_source = payload.get(
                            "energy_source",
                            "unknown",
                        )

                        self.logger.info(
                            "Received latest Simu5G snapshot: "
                            "%d cells at sim_time=%s",
                            len(cells),
                            sim_time,
                        )

                        requests = []

                        for cell in cells:
                            requests.append(
                                PredictionRequest(
                                    bs_id=str(
                                        cell.get("bs_id")
                                        or cell.get("BS")
                                        or cell.get("cell_id")
                                        or "unknown"
                                    ),
                                    metrics=cell,
                                    config_params={},
                                    request_id=(
                                        "simu5g-"
                                        f"{uuid.uuid4().hex[:8]}"
                                    ),
                                    timestamp=(
                                        SIMULATION_EPOCH
                                        + float(sim_time or 0.0)
                                    ),
                                )
                            )

                        responses = (
                            await self.process_prediction_batch(
                                requests
                            )
                        )

                        if len(responses) != len(cells):
                            raise RuntimeError(
                                "Snapshot response count does not "
                                f"match cell count: "
                                f"{len(responses)} != {len(cells)}"
                            )

                        for cell, resp in zip(
                            cells,
                            responses,
                        ):
                            resp_flat = asdict(resp)

                            load = float(cell.get(
                                "load",
                                float(cell.get("prb_util_pct", 50.0)) / 100.0
                            ))

                            raw_es_modes = cell.get(
                                "es_modes",
                                [0, 0, 0, 0, 0, 0],
                            )

                            if not isinstance(
                                raw_es_modes,
                                (list, tuple),
                            ):
                                raw_es_modes = []

                            es_modes = []

                            for value in raw_es_modes[:6]:
                                try:
                                    es_modes.append(
                                        1 if int(value) != 0 else 0
                                    )
                                except (TypeError, ValueError):
                                    es_modes.append(0)

                            es_modes.extend(
                                [0] * (6 - len(es_modes))
                            )

                            payload = {
                                "request_id": resp_flat["request_id"],
                                "bs_id": resp_flat["bs_id"],
                                "predicted_energy": resp_flat["predicted_energy"],
                                "confidence_score": resp_flat["confidence_score"],
                                "twin_id": twin_id,
                                "sim_time_s": sim_time,
                                "physical_sim_time_s": physical_sim_time,
                                "current_metrics": {
                                    "load": load,
                                    "prb_utilization": float(
                                        cell.get(
                                            "prb_util_pct",
                                            load * 100.0,
                                        )
                                    ),
                                    "available_prbs": int(
                                        cell.get("available_prbs", 0)
                                    ),
                                    "used_prbs": float(
                                        cell.get("used_prbs", 0.0)
                                    ),
                                    "throughput": float(
                                        cell.get("dl_mbps", 0.0)
                                    ),
                                    "generated_dl_mbps": float(
                                        cell.get(
                                            "generated_dl_mbps",
                                            0.0,
                                        )
                                    ),
                                    "dl_mbps": float(
                                        cell.get("dl_mbps", 0.0)
                                    ),
                                    "ul_mbps": float(
                                        cell.get("ul_mbps", 0.0)
                                    ),
                                    "total_throughput_mbps": float(
                                        cell.get(
                                            "total_throughput_mbps",
                                            0.0,
                                        )
                                    ),
                                    "latency": float(
                                        cell.get(
                                            "avg_latency_ms",
                                            0.0,
                                        )
                                    ),
                                    "packet_loss": float(
                                        cell.get(
                                            "packet_loss_pct",
                                            0.0,
                                        )
                                    ),
                                    "sinr": float(
                                        cell.get(
                                            "avg_sinr_db",
                                            0.0,
                                        )
                                    ),
                                    "active_ues": int(
                                        cell.get("active_ues", 0)
                                    ),
                                    "measured_energy_w": float(
                                        cell.get(
                                            "energy_consumption_w",
                                            0.0,
                                        )
                                    ),
                                    "interval_energy_j": float(
                                        cell.get(
                                            "interval_energy_j",
                                            0.0,
                                        )
                                    ),
                                    "represented_interval_s": float(
                                        cell.get(
                                            "represented_interval_s",
                                            0.0,
                                        )
                                    ),
                                    "represented_energy_j": float(
                                        cell.get(
                                            "represented_energy_j",
                                            0.0,
                                        )
                                    ),
                                    "energy_efficiency_mbit_per_j": float(
                                        cell.get(
                                            "energy_efficiency_mbit_per_j",
                                            0.0,
                                        )
                                    ),
                                    "measurement_interval_s": float(
                                        cell.get(
                                            "measurement_interval_s",
                                            0.0,
                                        )
                                    ),
                                },

                                "current_config": {
                                    "tx_power": float(cell.get("tx_power_dbm", 43.0)),
                                    "bandwidth": int(cell.get("bandwidth_mhz", 20)),
                                    "carrier_count": int(cell.get("carrier_count", 1)),
                                    "es_mode": str(cell.get("es_mode", "active")),
                                    "es_modes": es_modes,
                                    "physical_tx_power": float(
                                        cell.get(
                                            "physical_tx_power_dbm",
                                            cell.get(
                                                "tx_power_dbm",
                                                43.0,
                                            ),
                                        )
                                    ),
                                    "radio_sleeping": bool(
                                        cell.get(
                                            "radio_sleeping",
                                            False,
                                        )
                                    ),
                                    "requested_bandwidth": float(
                                        cell.get(
                                            "requested_bandwidth_mhz",
                                            cell.get(
                                                "bandwidth_mhz",
                                                20.0,
                                            ),
                                        )
                                    ),
                                    "requested_carrier_count": int(
                                        cell.get(
                                            "requested_carrier_count",
                                            cell.get(
                                                "carrier_count",
                                                1,
                                            ),
                                        )
                                    ),
                                },

                                "cell_metadata": {
                                    "zone": cell.get("zone"),
                                    "zone_name": cell.get("zone_name"),
                                    "bs_type": cell.get("bs_type"),
                                    "ru_type": cell.get("ru_type"),
                                    "mode": cell.get("mode"),
                                    "frequency_mhz": cell.get("frequency_mhz"),
                                    "num_antennas": cell.get("num_antennas"),
                                    "traffic_level": cell.get("traffic_level"),
                                    "kpi_source": kpi_source,
                                    "energy_source": energy_source,
                                },
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
            self.logger.info(
                "  - Batches processed: "
                f"{self.metrics['batches_processed']}"
            )
            self.logger.info(
                "  - Stale snapshots discarded: "
                f"{self.metrics['stale_snapshots_dropped']}"
            )

            average_batch_time = (
                self.metrics["total_batch_processing_time"]
                / max(
                    1,
                    self.metrics["batches_processed"],
                )
            )

            self.logger.info(
                f"  - Average batch time: "
                f"{average_batch_time:.2f}ms"
            )            
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