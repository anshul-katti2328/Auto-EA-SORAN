#!/usr/bin/env python3
"""
Communication Handler for Feedback Agent
Manages ZeroMQ-based messaging with other agents in the Auto EA-SORAN system.
"""
import sys
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )
import zmq
import zmq.asyncio
import json
import time
import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
import uuid
from optimizationCOMM import OptimizationMessageType as MessageType, OptimizationMessage as Message
import logging

def get_logger(name: str):
    """Setup logger with console handler"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Only add if no handlers exist
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

class CommunicationError(Exception):
    pass

class CommunicationHandler:
    """
    Handles all ZeroMQ communication for the Feedback Agent.
    Supports both request-reply and publish-subscribe patterns.
    """
    
    def __init__(self, config: Dict):
        """Initialize communication handler"""
        self.config = config
        self.config.setdefault('reply_port', 5560)
        self.config.setdefault('pub_port', 5561)
        self.logger = get_logger(__name__)
        
        # ZeroMQ context and sockets
        self.context = zmq.asyncio.Context()
        self.sockets = {}
        
        # Message handlers
        self.handlers: Dict[MessageType, Callable] = {}
        
        # Agent identification
        self.agent_id = config['agent_id']
        
        # Connection endpoints
        self.endpoints = config['endpoints']
        
        # Message tracking
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.message_history: List[Message] = []
        
        # Performance metrics
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'avg_response_time': 0.0
        }
        
        self.logger.info(f"Communication handler initialized for agent {self.agent_id}")
    
    async def initialize(self):
        """Initialize FeedbackAgent ZeroMQ communication."""
        try:
            validation_port = int(
                self.config.get("reply_port", 5560)
            )
            response_host = self.config.get(
                "optimization_response_host",
                "localhost",
            )
            response_port = int(
                self.config.get(
                    "optimization_response_port", 5559
                )
            )
            pub_port = int(
                self.config.get("pub_port", 5561)
            )

            # OptimizationAgent PUSH -> FeedbackAgent PULL
            self.sockets["validation_pull"] = (
                self.context.socket(zmq.PULL)
            )
            self.sockets["validation_pull"].setsockopt(
                zmq.LINGER, 0
            )
            self.sockets["validation_pull"].bind(
                f"tcp://*:{validation_port}"
            )

            self.logger.info(
                "Listening for validation requests on port %s",
                validation_port,
            )

            # FeedbackAgent PUSH -> OptimizationAgent PULL
            self.sockets["optimization_push"] = (
                self.context.socket(zmq.PUSH)
            )
            self.sockets["optimization_push"].setsockopt(
                zmq.LINGER, 0
            )
            self.sockets["optimization_push"].connect(
                f"tcp://{response_host}:{response_port}"
            )

            self.logger.info(
                "Connected to OptimizationAgent at %s:%s",
                response_host,
                response_port,
            )

            # Optional direct request connections.
            self.sockets["request"] = {}

            for agent_name, endpoint in self.endpoints.items():
                if agent_name == "optimization_agent":
                    continue

                if "host" not in endpoint or "port" not in endpoint:
                    continue

                socket = self.context.socket(zmq.REQ)
                socket.setsockopt(zmq.LINGER, 0)
                socket.connect(
                    f"tcp://{endpoint['host']}:{endpoint['port']}"
                )
                self.sockets["request"][agent_name] = socket

            # FeedbackAgent status publisher
            self.sockets["publisher"] = self.context.socket(
                zmq.PUB
            )
            self.sockets["publisher"].setsockopt(zmq.LINGER, 0)
            self.sockets["publisher"].bind(
                f"tcp://*:{pub_port}"
            )

            # Subscribe to configured agent publishers.
            self.sockets["subscriber"] = self.context.socket(
                zmq.SUB
            )
            self.sockets["subscriber"].setsockopt(zmq.LINGER, 0)
            self.sockets["subscriber"].setsockopt_string(
                zmq.SUBSCRIBE, ""
            )

            for endpoint in self.endpoints.values():
                if "host" in endpoint and "pub_port" in endpoint:
                    self.sockets["subscriber"].connect(
                        f"tcp://{endpoint['host']}:"
                        f"{endpoint['pub_port']}"
                    )

            self.logger.info(
                "FeedbackAgent ZeroMQ sockets initialized"
            )

        except Exception as error:
            raise CommunicationError(
                f"Failed to initialize communication: {error}"
            ) from error
    
    def register_handlers(self, handlers: Dict[MessageType, Callable]):
        """Register message handlers for different message types"""
        self.handlers.update(handlers)
        self.logger.info(f"Registered {len(handlers)} message handlers")
    

    # Lines 99-200 - Replace entire process_messages method

    async def process_messages(self, timeout: float = 1.0):
        """Process incoming messages from all sockets"""
        try:
        # === VALIDATION_PULL SOCKET (Main communication channel) ===
            if 'validation_pull' in self.sockets:
                try:
                    if await self.sockets['validation_pull'].poll(timeout=int(timeout * 1000), flags=zmq.POLLIN):
                        message_data = await self.sockets['validation_pull'].recv_json(flags=zmq.NOBLOCK)
                
                    # [OK] Use correct payload structure
                        bs_id = message_data.get('payload', {}).get('target_bs_id', 'unknown')
                        self.logger.info(f"📥 Received validation request: {bs_id}")
                        self.logger.info(f"   Message type: {message_data.get('message_type', 'unknown')}")

                        message_type_str = message_data.get('message_type', 'unknown')
                        payload = message_data.get('payload', {})
            
                        try:
                            # [OK] Handle message type conversion properly
                            if message_type_str in ['optimization_response', 'OPTIMIZATION_RESPONSE']:
                                message_type = MessageType.OPTIMIZATION_RESPONSE
                            else:
                                try:
                                    message_type = MessageType(message_type_str)
                                except ValueError:
                                    self.logger.warning(f"Unknown message type: {message_type_str}")
                                    return
                
                        # [OK] Use self.handlers (correct attribute name)
                            if message_type in self.handlers:
                                response = await self.handlers[message_type](payload)
                                self.logger.info(f"Handler response: {response}")
                            
                            # [OK] Send response back using PUSH socket instead of REQ
                                if response and 'optimization_push' in self.sockets:
                                    response_message = {
                                    'message_type': 'validation_result',
                                    'sender_id': self.agent_id,
                                    'payload': response,
                                    'timestamp': datetime.now().isoformat(),
                                    'reply_to': message_data.get('message_id')
                                    }
                                
                                    await self.sockets['optimization_push'].send_json(self._make_json_safe(response_message))
                                    self.logger.info(f"[OK] Sent validation result back to optimization agent for {bs_id}")
                            
                                self.logger.info(f"[OK] Validation completed for {bs_id}")
                            else:
                                self.logger.warning(f"No handler for message type: {message_type}")
                        
                        except Exception as handler_error:
                            self.logger.error(f"Handler error: {handler_error}")
                            import traceback
                            self.logger.error(f"Traceback: {traceback.format_exc()}")
                
                except zmq.Again:
                    pass  # No message available
                except Exception as e:
                    self.logger.error(f"Error processing validation request: {e}")
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
    
    async def send_validation_response(self, response_data: Dict[str, Any]):
        """Send validation response back to optimization agent"""
        try:
            if 'optimization_push' in self.sockets:
                await self.sockets['optimization_push'].send_json(self._make_json_safe(response_data))
                self.logger.info("[OK] Validation response sent to optimization agent")
                self.metrics['messages_sent'] += 1
            else:
                self.logger.error("No optimization_push socket available")
        except Exception as e:
            self.logger.error(f"Failed to send validation response: {e}")

    async def _handle_validation_pull_message(self):
        """Handle incoming validation request from PULL socket"""
        try:
        # Receive message (PULL socket - no reply needed)
            raw_message = await self.sockets['validation_pull'].recv_json(zmq.NOBLOCK)
            message = self._deserialize_message(raw_message)
        
            self.logger.info(f"📥 Received validation request: {message.message_type} from {message.sender_id}")
            self.metrics['messages_received'] += 1
        
        # Store message in history
            self.message_history.append(message)
        
        # Process message (PULL doesn't send reply - agent handles response via broadcast or different channel)
            result = await self._process_message(message)
        
        # FeedbackAgent can broadcast results or store them
            if result.get('is_valid') is not None:
            # Broadcast validation result
                await self.broadcast(MessageType.VALIDATION_RESULT, result)
            
        except zmq.Again:
            pass  # No message available
        except Exception as e:
            self.logger.error(f"Error handling validation request: {e}")

    async def _socket_has_message(self, socket, timeout: float) -> bool:
        """Check if socket has incoming message with timeout"""
        try:
            events = await socket.poll(timeout * 1000, zmq.POLLIN)
            return bool(events & zmq.POLLIN)
        except zmq.Again:
            return False
    
    async def _handle_request_message(self):
        """Handle incoming request message"""
        try:
            # Receive message
            raw_message = await self.sockets['reply'].recv_json(zmq.NOBLOCK)
            message = self._deserialize_message(raw_message)
            
            self.logger.debug(f"Received request: {message.message_type} from {message.sender_id}")
            self.metrics['messages_received'] += 1
            
            # Store message in history
            self.message_history.append(message)
            
            # Process message
            response_payload = await self._process_message(message)
            
            # Send response
            response = Message(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.ACK,
                payload=response_payload,
                timestamp=datetime.now(),
                reply_to=message.message_id,
                correlation_id=message.correlation_id
            )
            
            await self.sockets['reply'].send_json(self._serialize_message(response))
            self.metrics['messages_sent'] += 1
            
        except zmq.Again:
            pass  # No message available
        except Exception as e:
            self.logger.error(f"Error handling request: {e}")
            # Send error response
            error_response = {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            try:
                await self.sockets['reply'].send_json(error_response)
            except:
                pass
    
    async def _handle_broadcast_message(self):
        """Handle incoming broadcast message"""
        try:
            raw_message = await self.sockets['subscriber'].recv_json(zmq.NOBLOCK)
            message = self._deserialize_message(raw_message)
            
            # Skip our own broadcasts
            if message.sender_id == self.agent_id:
                return
            
            self.logger.debug(f"Received broadcast: {message.message_type} from {message.sender_id}")
            self.metrics['messages_received'] += 1
            
            # Store message in history
            self.message_history.append(message)
            
            # Process message (no response needed for broadcasts)
            await self._process_message(message)
            
        except zmq.Again:
            pass  # No message available
        except Exception as e:
            self.logger.error(f"Error handling broadcast: {e}")
    
    async def _process_message(self, message: Message) -> Dict[str, Any]:
        """Process incoming message using registered handlers"""
        try:
        # Handle both enum and string message types
            message_key = message.message_type
            if isinstance(message.message_type, MessageType):
                message_key = message.message_type
            else:
            # Handle string-based message types
                message_key = str(message.message_type)
            
            if message_key in self.handlers:
                handler = self.handlers[message_key]
                start_time = time.time()
            # Call handler
                result = await handler(message.payload)
            # Update metrics
                response_time = time.time() - start_time
                self._update_avg_response_time(response_time)
                return result if result is not None else {'status': 'processed'}
            else:
                self.logger.warning(f"No handler for message type: {message_key}")
                return {'error': f'No handler for message type: {message_key}'}
        except Exception as e:
            self.logger.error(f"Error processing message {message.message_id}: {e}")
            return {'error': str(e)}
    
    async def send_request(self, agent_name: str, message_type: MessageType, 
                          payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """Send request to another agent and wait for response"""
        
        if agent_name not in self.sockets['request']:
            raise CommunicationError(f"No connection to agent: {agent_name}")
        
        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=agent_name,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now(),
            correlation_id=str(uuid.uuid4())
        )
        
        try:
            socket = self.sockets['request'][agent_name]
            
            # Send request
            await socket.send_json(self._serialize_message(message))
            self.metrics['messages_sent'] += 1
            
            self.logger.debug(f"Sent request {message_type} to {agent_name}")
            
            # Wait for response with timeout
            if await socket.poll(timeout * 1000, zmq.POLLIN):
                response_data = await socket.recv_json()
                response = self._deserialize_message(response_data)
                
                self.logger.debug(f"Received response from {agent_name}")
                return response.payload
            else:
                raise CommunicationError(f"Timeout waiting for response from {agent_name}")
                
        except Exception as e:
            self.logger.error(f"Error sending request to {agent_name}: {e}")
            raise CommunicationError(f"Failed to send request: {e}")
    
    async def broadcast(self, message_type: MessageType, payload: Dict[str, Any]):
        """Broadcast message to all agents"""
        message = Message(
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
            raise CommunicationError(f"Failed to broadcast: {e}")
    
    async def send_heartbeat(self):
        """Send heartbeat to all other agents"""
        heartbeat_payload = {
            'agent_id': self.agent_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'alive',
            'metrics': self.metrics
        }
        
        await self.broadcast(MessageType.HEARTBEAT, heartbeat_payload)
    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        """Recursively convert supported values into JSON-safe objects."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, (datetime, date)):
            return value.isoformat()

        if isinstance(value, Enum):
            return CommunicationHandler._make_json_safe(value.value)

        if is_dataclass(value):
            return CommunicationHandler._make_json_safe(asdict(value))

        if isinstance(value, dict):
            return {
                str(key): CommunicationHandler._make_json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                CommunicationHandler._make_json_safe(item)
                for item in value
            ]

        # Handles NumPy scalar values without importing NumPy here.
        item_method = getattr(value, "item", None)
        if callable(item_method):
            return CommunicationHandler._make_json_safe(item_method())

        raise TypeError(
            f"Unsupported JSON value type: {type(value).__name__}"
        )

    def _serialize_message(self, message: Message) -> Dict[str, Any]:
        """Serialize a message and all nested payload values safely."""
        serialized = {
            'message_id': message.message_id,
            'sender_id': message.sender_id,
            'recipient_id': message.recipient_id,
            'message_type': message.message_type.value,
            'payload': message.payload,
            'timestamp': message.timestamp.isoformat(),
            'reply_to': message.reply_to,
            'correlation_id': message.correlation_id
        }
        return self._make_json_safe(serialized)
    
    def _deserialize_message(self, data: Dict[str, Any]) -> Message:
        """Deserialize dictionary to message object"""
        return Message(
            message_id=data['message_id'],
            sender_id=data['sender_id'],
            recipient_id=data['recipient_id'],
            message_type=MessageType(data['message_type']),
            payload=data['payload'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            reply_to=data.get('reply_to'),
            correlation_id=data.get('correlation_id')
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
        self.logger.info("Closing communication handler...")
        
        for socket_name, socket in self.sockets.items():
            if isinstance(socket, dict):
                for agent_socket in socket.values():
                    agent_socket.close()
            else:
                socket.close()
        
        self.context.term()
        self.logger.info("Communication handler closed")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get communication metrics"""
        return {
            **self.metrics,
            'pending_requests': len(self.pending_requests),
            'message_history_size': len(self.message_history),
            'connected_agents': list(self.endpoints.keys())
        }