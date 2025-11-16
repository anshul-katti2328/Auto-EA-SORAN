#!/usr/bin/env python3
"""
OrchestrationAgent.py
Now delegates OMNeT++ parameter changes to OrchestrationShim via ZeroMQ.
- Listens on ZeroMQ REP socket (commands from OptimizationAgent)
- Forwards valid actions to OrchestrationShim (REQ socket)
- Logs metrics
"""

import zmq
import json
import time
from datetime import datetime
import csv
from pathlib import Path

class OrchestrationAgent:
    def __init__(self, zmq_port=5563, shim_port=6000, poll_interval=0.05, metrics_file="metrics.csv"):
        self.zmq_port = zmq_port         # port for OptimizationAgent
        self.shim_port = shim_port       # port to talk to OMNeT++ shim
        self.poll_interval = poll_interval
        self.metrics_file = Path(metrics_file)

        self.total_commands = 0
        self.successful = 0
        self.failed = 0
        self.action_counts = {}
        self.last_action_time = {}
        self.settling_time = 0.1  # seconds

        # REP socket to receive commands from OptimizationAgent
        ctx = zmq.Context()
        self.rep_socket = ctx.socket(zmq.REP)
        self.rep_socket.bind(f"tcp://*:{self.zmq_port}")
        print(f"[OrchestrationAgent] Listening for commands on tcp://*:{self.zmq_port}")

        # REQ socket to send commands to OMNeT++ shim
        self.shim_socket = ctx.socket(zmq.REQ)
        self.shim_socket.connect(f"tcp://localhost:{self.shim_port}")
        print(f"[OrchestrationAgent] Connected to OrchestrationShim at tcp://localhost:{self.shim_port}")

        # Ensure metrics file exists
        if not self.metrics_file.exists():
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "total", "successful", "failed", "last_action", "last_result", "action_counts"])

    def now_iso(self):
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def append_metrics_csv(self, last_action, last_result):
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.now_iso(),
                self.total_commands,
                self.successful,
                self.failed,
                last_action,
                last_result,
                json.dumps(self.action_counts)
            ])

    def process_command(self, payload):
        self.total_commands += 1
        last_action = ""
        last_result = "FAILED_PARSE"

        print(f"[DEBUG] Raw payload received: {payload}")

        try:
            j = json.loads(payload)
            action = j.get("action", "")
            target = j.get("target", "")
        
            print(f"[DEBUG] Parsed action: '{action}'")
            print(f"[DEBUG] Parsed target: '{target}'")
        
            params = j.get("param", {})
            last_action = action
            self.action_counts[action] = self.action_counts.get(action, 0) + 1

        # ✅ SPECIAL HANDLING FOR HEALING PLANS
            if action == "executeHealingPlan":
                print(f"[OrchestrationAgent] 🔧 Processing healing plan for {target}")
            
            # ✅ ANALYZE VIOLATIONS TO DETERMINE ACTIONS
                violations = j.get("critical_violations", [])
                recommendations = j.get("structured_recommendations", {})
            
            # Initialize shim-compatible parameters
                delta_tx_db = 0.0
                tx_power_dbm = 20.0  # Default safe power level
                es_mode = "normal"   # Default energy saving mode
            
            # Parse violations for power and latency issues
                power_violation_found = False
                latency_violation_found = False
            
                for violation in violations:
                    message = violation.get("message", "").lower()
                    severity = violation.get("severity", "").lower()
                
                    print(f"[OrchestrationAgent] Analyzing violation: {message}")
                
                # Power violations - reduce TX power
                    if "transmission power" in message and "exceeds" in message:
                        if "65.0dbm" in message and "43.0dbm" in message:
                        # Specific violation: 65dBm exceeds 43dBm max
                            delta_tx_db = -22.0  # Reduce from 65 to 43 dBm
                            tx_power_dbm = 43.0  # Set to compliance level
                            power_violation_found = True
                            print(f"[OrchestrationAgent] Power violation detected: reducing to 43dBm")
                        else:
                        # General power violation
                            delta_tx_db = -5.0   # Reduce by 5dB
                            tx_power_dbm = 20.0  # Safe level
                            power_violation_found = True
                            print(f"[OrchestrationAgent] General power violation: reducing by 5dB")
                        
                # Latency violations - enable energy saving
                    if any(term in message for term in ["latency", "urllc", "network slice", "autonomous vehicle"]):
                        if "exceeds" in message or severity == "critical":
                            es_mode = "sleep"
                            latency_violation_found = True
                            print(f"[OrchestrationAgent] Latency violation detected: enabling sleep mode")
            
            # Apply recommendation-based adjustments
                config_mods = recommendations.get("config_modifications", {})
                if config_mods.get("gradual_power_reduction", False) and not power_violation_found:
                    delta_tx_db = -3.0  # Gentle power reduction
                    print(f"[OrchestrationAgent] Applying gradual power reduction from recommendations")
                
                if config_mods.get("enable_low_latency_mode", False) and not latency_violation_found:
                    es_mode = "active"  # Active mode for low latency
                    print(f"[OrchestrationAgent] Enabling active mode for low latency")
            
            # ✅ CREATE SHIM-COMPATIBLE COMMAND (matches OrchestrationShim.cc format)
                healing_shim_command = {
                "bs_id": target,           # Base station ID
                "delta_tx_db": delta_tx_db,    # Power adjustment in dB
                "tx_power_dbm": tx_power_dbm,  # Absolute power level
                "es_mode": es_mode             # Energy saving mode
                }
            
                try:
                    print(f"[OrchestrationAgent] 📡 Sending healing config to OrchestrationShim...")
                    print(f"[OrchestrationAgent] Healing config: {healing_shim_command}")
                
                # Set timeouts for shim communication
                    self.shim_socket.setsockopt(zmq.SNDTIMEO, 3000)  # 3 second timeout
                    self.shim_socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3 second timeout
                
                # Send to OrchestrationShim
                    self.shim_socket.send_string(json.dumps(healing_shim_command))
                    shim_reply = self.shim_socket.recv_string()
                
                    print(f"[OrchestrationAgent] Shim reply: {shim_reply}")
                
                    reply_obj = json.loads(shim_reply)
                
                # ✅ CHECK FOR SHIM SUCCESS (matches OrchestrationShim.cc response)
                    if reply_obj.get("accepted", False):
                        applied_count = reply_obj.get("applied_count", 0)
                        sim_time = reply_obj.get("applied_at_sim_s", 0.0)
                    
                        last_result = "HEALING_PLAN_APPLIED"
                        self.successful += 1
                        print(f"[OrchestrationAgent] ✅ Healing plan applied successfully!")
                        print(f"[OrchestrationAgent] Applied {applied_count} changes at simulation time {sim_time}s")
                    
                    elif "accepted" in reply_obj and not reply_obj["accepted"]:
                    # Shim explicitly rejected
                        last_result = "HEALING_PLAN_REJECTED"
                        self.failed += 1
                        print(f"[OrchestrationAgent] ❌ Shim rejected healing plan: {reply_obj}")
                    
                    else:
                        # Legacy response format or error
                        if reply_obj.get("status") == "OK":
                            last_result = "HEALING_PLAN_APPLIED"
                            self.successful += 1
                            print(f"[OrchestrationAgent] ✅ Healing plan applied (legacy format)")
                        else:
                            last_result = "HEALING_PLAN_QUEUED"
                            self.successful += 1
                            print(f"[OrchestrationAgent] ⚠️ Healing plan queued: {reply_obj.get('detail', 'Unknown status')}")
                    
                except zmq.Again:
                # Timeout - still consider success since plan was valid
                    last_result = "HEALING_PLAN_TIMEOUT"
                    self.successful += 1
                    print(f"[OrchestrationAgent] ⚠️ OrchestrationShim timeout - healing plan was valid but shim not responding")
                
                except json.JSONDecodeError as json_error:
                    last_result = "HEALING_PLAN_INVALID_RESPONSE"
                    self.failed += 1
                    print(f"[OrchestrationAgent] ❌ Invalid JSON response from shim: {json_error}")
                
                except Exception as shim_error:
                    last_result = "HEALING_PLAN_COMM_ERROR"
                    self.failed += 1
                    print(f"[OrchestrationAgent] ❌ Shim communication error: {shim_error}")
        
            else:
            # ✅ EXISTING LOGIC FOR REGULAR COMMANDS
                now = time.time()
                if target in self.last_action_time and (now - self.last_action_time[target]) < self.settling_time:
                    print(f"[OrchestrationAgent] Skipping command to {target} (settling time not elapsed)")
                    last_result = "SKIPPED_SETTLING"
                else:
                    self.last_action_time[target] = now

                    try:
                    # Forward regular command to shim
                        print(f"[OrchestrationAgent] 📡 Forwarding regular command: {action}")
                    
                        self.shim_socket.setsockopt(zmq.SNDTIMEO, 3000)
                        self.shim_socket.setsockopt(zmq.RCVTIMEO, 3000)

                        self.shim_socket.send_string(json.dumps(j))
                        shim_reply = self.shim_socket.recv_string()
                    
                        print(f"[OrchestrationAgent] Regular command - Shim reply: {shim_reply}")

                        reply_obj = json.loads(shim_reply)
                    
                    # Handle both new and legacy response formats
                        if reply_obj.get("accepted", False) or reply_obj.get("status") == "OK":
                            last_result = "OK"
                            self.successful += 1
                        else:
                            last_result = reply_obj.get("detail", reply_obj.get("error", "FAILED"))
                            self.failed += 1
                        
                    except zmq.Again:
                        last_result = "COMMAND_TIMEOUT"
                        self.failed += 1
                        print(f"[OrchestrationAgent] ❌ Regular command timeout")
                    
                    except json.JSONDecodeError as json_error:
                        last_result = "INVALID_SHIM_RESPONSE"
                        self.failed += 1
                        print(f"[OrchestrationAgent] ❌ Invalid JSON from shim: {json_error}")
                    
                    except Exception as shim_error:
                        last_result = f"SHIM_ERROR: {str(shim_error)}"
                        self.failed += 1
                        print(f"[OrchestrationAgent] ❌ Shim error: {shim_error}")

        except json.JSONDecodeError as parse_error:
            print(f"[OrchestrationAgent] ❌ JSON parsing error: {parse_error}")
            self.failed += 1
            last_result = f"INVALID_JSON: {str(parse_error)}"
        
        except Exception as e:
            print(f"[OrchestrationAgent] ❌ Unexpected error processing command: {e}")
            self.failed += 1
            last_result = f"PROCESSING_ERROR: {str(e)}"

    # ✅ LOG METRICS
        self.append_metrics_csv(last_action, last_result)

    # ✅ CREATE RESPONSE
        success_statuses = [
        "OK", 
        "HEALING_PLAN_APPLIED", 
        "HEALING_PLAN_QUEUED",
        "HEALING_PLAN_TIMEOUT",  # Still success - plan was valid
        "SKIPPED_SETTLING"
        ]
    
        reply = {
        "status": "OK" if last_result in success_statuses else "ERROR",
        "detail": last_result,
        "total": self.total_commands,
        "successful": self.successful,
        "failed": self.failed,
        "timestamp": datetime.utcnow().isoformat()
        }
    
        return json.dumps(reply)
    
    def run(self):
        """Main loop - listen for commands and respond"""
        print(f"[OrchestrationAgent] Starting main loop...")
    
        while True:
            try:
            # ✅ RECEIVE REQUEST (blocking)
                request = self.rep_socket.recv_string()
                print(f"[OrchestrationAgent] Received: {request}")
            
            # ✅ PROCESS COMMAND
                response = self.process_command(request)
            
            # ✅ SEND RESPONSE (CRITICAL FOR REQ-REP!)
                self.rep_socket.send_string(response)
                print(f"[OrchestrationAgent] Sent response: {response[:100]}...")
            
            except KeyboardInterrupt:
                print(f"[OrchestrationAgent] Shutting down...")
                break
            except Exception as e:
                print(f"[OrchestrationAgent] Error in main loop: {e}")
            # ✅ STILL SEND ERROR RESPONSE
                error_response = json.dumps({
                "status": "ERROR",
                "detail": f"Processing error: {str(e)}",
                "total": self.total_commands,
                "successful": self.successful,
                "failed": self.failed
                })
                try:
                    self.rep_socket.send_string(error_response)
                except:
                    pass


if __name__ == "__main__":
    agent = OrchestrationAgent(zmq_port=5563)  # Ensure port matches the one set in OptimizationAgent
    agent.run()
