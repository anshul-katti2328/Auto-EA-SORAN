#!/usr/bin/env python3
"""
Communication Handler for Optimization Agent
Manages ZeroMQ-based messaging with Feedback Agent and other agents in Auto EA-SORAN.
"""

import asyncio
import traceback
import zmq
import zmq.asyncio
import json
import time
import logging
import uuid
from typing import Dict, List, Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass,asdict
from datetime import datetime

class CommunicationError(Exception):
    """Exception raised for communication-related errors"""
    pass

class OptimizationMessageType(Enum):
    """Message types for optimization agent communication"""
    # Optimization requests/responses
    OPTIMIZATION_REQUEST = "optimization_request"
    OPTIMIZATION_RESPONSE = "optimization_response"
    
    # Validation with Feedback Agent
    VALIDATE_ENERGY_PLAN = "validate_energy_plan"
    VALIDATION_RESULT = "validation_result"
    
    # Prediction updates
    PREDICTION_UPDATE = "prediction_update"
    
    # System messages
    HEARTBEAT = "heartbeat"
    STATUS_REQUEST = "status_request"
    STATUS_RESPONSE = "status_response"
    SHUTDOWN = "shutdown"
    
    # Error messages
    ERROR = "error"
    ACK = "ack"
    FEEDBACK_RESPONSE = "feedback_response"

@dataclass
class OptimizationMessage:
    """Standard message structure for optimization agent communication"""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: OptimizationMessageType
    payload: Dict[str, Any]
    timestamp: datetime
    reply_to: Optional[str] = None
    correlation_id: Optional[str] = None

class OptimizationCommunicationHandler:
    """
    Handles all ZeroMQ communication for the Optimization Agent.
    Supports request-reply, push-pull, and publish-subscribe patterns.
    """

    def __init__(self, config: Dict):
        """Initialize optimization communication handler"""
        self.config = config
        self.logger = logging.getLogger('OptimizationCommunication')
        
        # ZeroMQ context and sockets
        self.context = zmq.asyncio.Context()
        self.sockets = {}
        
        # Message handlers
        self.handlers: Dict[OptimizationMessageType, Callable] = {}
        
        # Agent identification
        self.agent_id = config['agent_id']
        
        # Connection endpoints
        self.endpoints = config['endpoints']
        
        # Message tracking
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.message_history: List[OptimizationMessage] = []
        self.message_handlers: Dict[OptimizationMessageType, Callable] = {}  # ← ADD THIS

        # Performance metrics
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'avg_response_time': 0.0
        }
        
        self.logger.info(f"Optimization communication handler initialized for agent {self.agent_id}")

    async def initialize(self):
        """Initialize ZeroMQ sockets and connections"""
        try:
        # ✅ PULL socket for receiving predictions (from prediction agent)
            self.sockets['prediction_pull'] = self.context.socket(zmq.PULL)
        # Try different port if 5555 is busy
            prediction_port = self.config.get('prediction_port', 5558)
            try:
                self.sockets['prediction_pull'].bind(f"tcp://*:{prediction_port}")
                self.logger.info(f"✅ Listening for predictions on PULL socket: port {prediction_port}")
            except zmq.ZMQError as e:
                if "Address already in use" in str(e):
                    prediction_port = 5558  # Try alternative port
                    self.sockets['prediction_pull'].bind(f"tcp://*:{prediction_port}")
                    self.logger.info(f"✅ Listening for predictions on PULL socket: port {prediction_port} (fallback)")
                else:
                    raise
        
        # ✅ PULL socket for receiving responses from feedback agent
            self.sockets['feedback_pull'] = self.context.socket(zmq.PULL)
        # Try different port if 5557 is busy
            response_port = self.config.get('response_port', 5557)
            try:
                self.sockets['feedback_pull'].bind(f"tcp://*:{response_port}")
                self.logger.info(f"✅ Listening for feedback responses on PULL socket: port {response_port}")
            except zmq.ZMQError as e:
                if "Address already in use" in str(e):
                    response_port = 5559  # Try alternative port
                    self.sockets['feedback_pull'].bind(f"tcp://*:{response_port}")
                    self.logger.info(f"✅ Listening for feedback responses on PULL socket: port {response_port} (fallback)")
                else:
                    raise
        
        # ✅ PUSH socket for sending to feedback agent (instead of REQ)
            self.sockets['feedback_push'] = self.context.socket(zmq.PUSH)
            feedback_port = self.endpoints.get('feedback_agent', {}).get('port', 5560)
            self.sockets['feedback_push'].connect(f"tcp://localhost:{feedback_port}")
            self.logger.info(f"✅ Connected PUSH socket to feedback agent: port {feedback_port}")
        
        # REQ sockets for other agents (if needed) - store as dict
            self.sockets['request'] = {}
            for agent_name, endpoint in self.endpoints.items():
                if agent_name not in ['feedback_agent']:  # Skip feedback agent - we use PUSH for that
                    sock = self.context.socket(zmq.REQ)
                    sock.connect(f"tcp://{endpoint['host']}:{endpoint['port']}")
                    self.sockets['request'][agent_name] = sock
        
        # Publisher socket for broadcasting
            pub_port = self.config.get('pub_port', 5556)
            self.sockets['publisher'] = self.context.socket(zmq.PUB)
            try:
                self.sockets['publisher'].bind(f"tcp://*:{pub_port}")
                self.logger.info(f"✅ Publisher socket bound to port {pub_port}")
            except zmq.ZMQError as e:
                if "Address already in use" in str(e):
                    pub_port = 5570  # Try alternative port
                    self.sockets['publisher'].bind(f"tcp://*:{pub_port}")
                    self.logger.info(f"✅ Publisher socket bound to port {pub_port} (fallback)")
                else:
                    raise
        
        # Subscriber socket for receiving broadcasts
            self.sockets['subscriber'] = self.context.socket(zmq.SUB)
            self.sockets['subscriber'].subscribe("")  # Subscribe to all messages
        
        # Connect to other agents' publishers
            for agent_name, endpoint in self.endpoints.items():
                if 'pub_port' in endpoint:
                    self.sockets['subscriber'].connect(
                        f"tcp://{endpoint['host']}:{endpoint['pub_port']}"
                    )
        
            self.logger.info("ZeroMQ sockets initialized successfully")
        
        except Exception as e:
            raise CommunicationError(f"Failed to initialize communication: {e}")

    def register_handlers(self, handlers: Dict[OptimizationMessageType, Callable]):
        """Register message handlers for different message types"""
        self.handlers.update(handlers)
        self.logger.info(f"Registered {len(handlers)} optimization message handlers")

    async def send_to_feedback_agent(self, message_type: OptimizationMessageType, 
                                   payload: Dict[str, Any]) -> None:
        """Send message to Feedback Agent using PUSH socket"""
        if 'feedback_push' not in self.sockets:
            self.logger.error("❌ Feedback PUSH socket not initialized!")
            raise RuntimeError("Feedback PUSH socket not available")
        message = OptimizationMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id="feedback_agent",
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now()
        )
        
        try:
            # ✅ FIXED: Use correct field names from OptimizationAgent
            self.logger.info(f"📤 Sending {message_type.value} to FeedbackAgent")
            self.logger.info(f"   BS: {payload.get('target_bs_id', 'unknown')}")  # ✅ FIXED
            self.logger.info(f"   Energy Saving: {payload.get('energy_saving_percentage', 0):.2f}%")  # ✅ FIXED
            self.logger.info(f"   Confidence: {payload.get('confidence_score', 0):.3f}")  # ✅ FIXED
            
            # ✅ ADD: Debug payload structure
            self.logger.info(f"   Plan ID: {payload.get('plan_id', 'N/A')}")
            
            serialized = self._serialize_message(message)
            await self.sockets['feedback_push'].send_json(serialized)
        
            self.metrics['messages_sent'] += 1
        
            self.logger.info(f"✅ Successfully sent to FeedbackAgent (msg_id: {message.message_id[:8]})")
        
        except Exception as e:
            self.logger.error(f"❌ Error sending to FeedbackAgent: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _handle_prediction_update(self):
        """Handle incoming prediction and trigger optimization process"""
        try:
            if await self.sockets['prediction_pull'].poll(timeout=100, flags=zmq.POLLIN):
                message_data = await self.sockets['prediction_pull'].recv_json(flags=zmq.NOBLOCK)
            
                bs_id = message_data.get('bs_id', 'unknown')
                predicted_energy = message_data.get('predicted_energy', 0)
                confidence = message_data.get('confidence', 0)
            
                self.logger.info(f"📥 Received prediction update: {bs_id}")
            
            # Process prediction and trigger optimization
                if 'prediction_update' in self.handlers:
                    await self.handlers['prediction_update'](message_data)
            
            # Send optimization response to FeedbackAgent
                await self._send_optimization_to_feedback_agent(message_data)
                
        except zmq.Again:
            pass
        except Exception as e:
            self.logger.error(f"Error handling prediction update: {e}")

# Add this method to send optimization results to FeedbackAgent
    async def _send_optimization_to_feedback_agent(self, prediction_data: Dict[str, Any]):
        """Send optimization response to FeedbackAgent for validation"""
        try:
            bs_id = prediction_data.get('bs_id', 'unknown')
            predicted_energy = prediction_data.get('predicted_energy', 0)
            confidence = prediction_data.get('confidence', 0)
        
        # Create optimization response payload
            optimization_response = {
            'plan_id': str(uuid.uuid4()),
            'target_bs_id': bs_id,
            'proposed_config': {
                'tx_power': 25.0,  # Example optimization
                'sleep_mode_enabled': True,
                'carrier_aggregation': False,
                'mimo_layers': 2
            },
            'expected_energy_saving': 15.0,  # 15% energy saving
            'confidence_score': confidence,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'original_prediction': prediction_data,
                'optimization_algorithm': 'basic_energy_saver',
                'predicted_energy_consumption': predicted_energy
            }
            }
        
        # Send to FeedbackAgent
            await self.send_optimization_response(optimization_response)
        
            self.logger.info(f"✅ Sent optimization response to FeedbackAgent for BS {bs_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to send optimization to FeedbackAgent: {e}")

    async def process_messages(self, timeout: float = 1.0):
        """Process incoming messages from all sockets"""
        try:
        # === PREDICTION_PULL SOCKET ===
            if 'prediction_pull' in self.sockets:
                await self._handle_prediction_update()
        
        # === FEEDBACK_PULL SOCKET ===
            if 'feedback_pull' in self.sockets:
                try:
                    if await self.sockets['feedback_pull'].poll(timeout=100, flags=zmq.POLLIN):
                        message_data = await self.sockets['feedback_pull'].recv_json(flags=zmq.NOBLOCK)
                
                    # ✅ FIXED: Get plan_id instead of target_bs_id
                        payload = message_data.get('payload', {})
                        plan_id = payload.get('plan_id', 'unknown')
                        bs_id = payload.get('target_bs_id', 'N/A')
                    
                        self.logger.info(f"📥 Received feedback response: {plan_id} (BS: {bs_id})")
                
                    # ✅ FIXED: Use correct handler key and pass only payload
                        if OptimizationMessageType.FEEDBACK_RESPONSE in self.handlers:
                            await self.handlers[OptimizationMessageType.FEEDBACK_RESPONSE](payload)
                        elif 'feedback_response' in self.handlers:  # Fallback for string keys
                            await self.handlers['feedback_response'](payload)
                        else:
                            self.logger.warning("❌ No feedback_response handler registered!")
                        
                except zmq.Again:
                    pass
                except Exception as e:
                    self.logger.error(f"Error processing feedback response: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # === SUBSCRIBER SOCKET (broadcasts) ===
            if 'subscriber' in self.sockets:
                try:
                    if await self.sockets['subscriber'].poll(timeout=100, flags=zmq.POLLIN):
                        await self._handle_broadcast_message()
                except zmq.Again:
                    pass
                except Exception as e:
                    self.logger.error(f"Error processing broadcast: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in process_messages: {e}")
            self.metrics['errors'] += 1

    async def send_validation_request(self, request_data: Dict[str, Any]) -> bool:
        """Send validation request to feedback agent using PUSH socket"""
        try:
            bs_id = request_data.get('target_bs_id', 'unknown')
        
            message = {
            'message_type': 'validation_request',
            'message_id': str(uuid.uuid4()),
            'sender_id': self.agent_id,
            'timestamp': datetime.now().isoformat(),
            'payload': request_data
            }
        
            if 'feedback_push' in self.sockets:
                await self.sockets['feedback_push'].send_json(message)
                self.metrics['messages_sent'] += 1
                self.logger.info(f"✅ Validation request sent for {bs_id}")
                return True
            else:
                self.logger.error("No feedback_push socket available")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to send validation request: {e}")
            self.metrics['errors'] += 1
            return False

    async def send_optimization_response(self, response_data: Dict[str, Any]) -> bool:
        """Send optimization response to feedback agent using PUSH socket"""
        try:
            bs_id = response_data.get('target_bs_id', 'unknown')
            energy_saving = response_data.get('energy_saving_percentage', 0)
            confidence = response_data.get('confidence_score', 0)
        
            self.logger.info(f"📤 Sending optimization_response to FeedbackAgent")
            self.logger.info(f"   BS: {bs_id}")
            self.logger.info(f"   Energy Saving: {energy_saving:.2f}%")
            self.logger.info(f"   Confidence: {confidence:.3f}")
        
            message = {
            'message_type': 'optimization_response',
            'message_id': str(uuid.uuid4()),
            'sender_id': self.agent_id,
            'timestamp': datetime.now().isoformat(),
            'payload': {
                'plan_id': response_data.get('plan_id', str(uuid.uuid4())),  # ✅ Add this
                'target_bs_id': bs_id,
                'energy_saving_percentage': energy_saving,
                'confidence_score': confidence,
                **response_data  # Include all other fields
            }
        }
        
        # ✅ Use PUSH socket instead of REQ
            if 'feedback_push' in self.sockets:
                await self.sockets['feedback_push'].send_json(message)
                self.metrics['messages_sent'] += 1
                self.logger.info("✅ Optimization response sent successfully")
                return True
            else:
                self.logger.error("No feedback_push socket available")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to send optimization response: {e}")
            self.metrics['errors'] += 1
            return False


    async def _socket_has_message(self, socket, timeout: float) -> bool:
        """Check if socket has incoming message with timeout"""
        try:
            events = await socket.poll(timeout * 1000, zmq.POLLIN)
            return bool(events & zmq.POLLIN)
        except zmq.Again:
            return False

    async def _handle_request_message(self):
        """Handle incoming optimization requests"""
        try:
            raw_message = await self.sockets['request_reply'].recv_json(zmq.NOBLOCK)
            message = self._deserialize_message(raw_message)
            
            self.logger.debug(f"Received request: {message.message_type} from {message.sender_id}")
            self.metrics['messages_received'] += 1
            self.message_history.append(message)
            
            # Process message
            response_payload = await self._process_message(message)
            
            # Send response
            response = OptimizationMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=OptimizationMessageType.ACK,
                payload=response_payload,
                timestamp=datetime.now(),
                reply_to=message.message_id,
                correlation_id=message.correlation_id
            )
            
            await self.sockets['request_reply'].send_json(self._serialize_message(response))
            self.metrics['messages_sent'] += 1
            
        except zmq.Again:
            pass  # No message available
        except Exception as e:
            self.logger.error(f"Error handling request: {e}")

    async def _handle_prediction_update(self):
        """Handle incoming prediction updates"""
        try:
            raw_message = await self.sockets['prediction_pull'].recv_json(zmq.NOBLOCK)
            message = self._deserialize_message(raw_message)
            
            self.logger.debug(f"Received prediction update from {message.sender_id}")
            self.metrics['messages_received'] += 1
            self.message_history.append(message)
            
            # Process prediction update
            await self._process_message(message)
            
        except zmq.Again:
            pass  # No message available
        except Exception as e:
            self.logger.error(f"Error handling prediction update: {e}")

    async def _handle_broadcast_message(self):
        """Handle incoming broadcast messages"""
        try:
            raw_message = await self.sockets['subscriber'].recv_json(zmq.NOBLOCK)
            message = self._deserialize_message(raw_message)
            
            # Skip our own broadcasts
            if message.sender_id == self.agent_id:
                return
            
            self.logger.debug(f"Received broadcast: {message.message_type} from {message.sender_id}")
            self.metrics['messages_received'] += 1
            self.message_history.append(message)
            
            # Process message (no response needed for broadcasts)
            await self._process_message(message)
            
        except zmq.Again:
            pass  # No message available
        except Exception as e:
            self.logger.error(f"Error handling broadcast: {e}")

    async def _process_message(self, message: OptimizationMessage) -> Dict[str, Any]:
        """Process incoming message using registered handlers"""
        try:
            if message.message_type in self.handlers:
                handler = self.handlers[message.message_type]
                start_time = time.time()
                
                result = await handler(message.payload)
                
                response_time = time.time() - start_time
                self._update_avg_response_time(response_time)
                
                return result if result is not None else {'status': 'processed'}
            else:
                self.logger.warning(f"No handler for message type: {message.message_type}")
                return {'error': f'No handler for message type: {message.message_type}'}
                
        except Exception as e:
            self.logger.error(f"Error processing message {message.message_id}: {e}")
            return {'error': str(e)}

    async def broadcast(self, message_type: OptimizationMessageType, payload: Dict[str, Any]):
        """Broadcast message to all agents"""
        message = OptimizationMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id="*",  # Broadcast
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now()
        )
        
        try:
            await self.sockets['publisher'].send_json(self._serialize_message(message))
            self.metrics['messages_sent'] += 1
            self.logger.debug(f"Broadcast message: {message_type}")
        except Exception as e:
            self.logger.error(f"Error broadcasting message: {e}")
            raise

    async def send_heartbeat(self):
        """Send heartbeat to all other agents"""
        heartbeat_payload = {
            'agent_id': self.agent_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'alive',
            'metrics': self.metrics
        }
        
        await self.broadcast(OptimizationMessageType.HEARTBEAT, heartbeat_payload)

    def _serialize_message(self, message: OptimizationMessage) -> Dict[str, Any]:
        """Serialize message object to dictionary"""
        return {
            'message_id': message.message_id,
            'sender_id': message.sender_id,
            'recipient_id': message.recipient_id,
            'message_type': message.message_type.value,
            'payload': message.payload,
            'timestamp': message.timestamp.isoformat(),
            'reply_to': message.reply_to,
            'correlation_id': message.correlation_id
        }

    def _deserialize_message(self, data: Dict[str, Any]) -> OptimizationMessage:
        """Deserialize dictionary to message object - handles both structured and simple formats"""
        # Check if this is a structured message with all fields
        if 'message_id' in data and 'sender_id' in data:
            return OptimizationMessage(
                message_id=data['message_id'],
                sender_id=data['sender_id'],
                recipient_id=data['recipient_id'],
                message_type=OptimizationMessageType(data['message_type']),
                payload=data['payload'],
                timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else datetime.fromtimestamp(data['timestamp']),
                reply_to=data.get('reply_to'),
                correlation_id=data.get('correlation_id')
            )
        else:
            # Simple format from PredictionAgent - wrap it
            return OptimizationMessage(
                message_id=str(uuid.uuid4()),
                sender_id=data.get('twin_id', 'prediction_agent'),
                recipient_id=self.agent_id,
                message_type=OptimizationMessageType.PREDICTION_UPDATE,
                payload=data.get('payload', data),  # Handle both wrapped and unwrapped
                timestamp=datetime.fromtimestamp(data.get('timestamp', time.time())),
                reply_to=None,
                correlation_id=None
            )

    def _update_avg_response_time(self, response_time: float):
        """Update average response time metric"""
        current_avg = self.metrics['avg_response_time']
        total_responses = self.metrics['messages_received']
        
        if total_responses > 0:
            new_avg = ((current_avg * (total_responses - 1)) + response_time) / total_responses
            self.metrics['avg_response_time'] = new_avg

    async def close(self):
        """Close all sockets and cleanup"""
        self.logger.info("Closing optimization communication handler...")
    
        try:
        # Close individual sockets
            for socket_name, socket in self.sockets.items():
                if socket_name == 'request':
                # Handle request dictionary separately
                    if isinstance(socket, dict):
                        for req_socket in socket.values():
                            if hasattr(req_socket, 'close'):
                                req_socket.close()
                    else:
                        if hasattr(socket, 'close'):
                            socket.close()
                else:
                    # Handle regular sockets
                    if hasattr(socket, 'close'):
                        socket.close()
        
        # Clear sockets dictionary
            self.sockets.clear()
        
        # Terminate context
            if hasattr(self.context, 'term'):
                self.context.term()
        
            self.logger.info("Optimization communication handler closed")
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get communication metrics"""
        return {
            **self.metrics,
            'pending_requests': len(self.pending_requests),
            'message_history_size': len(self.message_history),
            'connected_agents': list(self.endpoints.keys())
        }