#!/usr/bin/env python3
"""
OrchestrationAgent.py
Now delegates OMNeT++ parameter changes to OrchestrationShim via ZeroMQ.
- Listens on ZeroMQ REP socket (commands from OptimizationAgent)
- Forwards valid actions to OrchestrationShim (REQ socket)
- Logs metrics
"""
import os
import subprocess
import zmq
import json
import time
from datetime import datetime, timezone
import csv
from pathlib import Path

class OrchestrationAgent:
    @staticmethod
    def _resolve_shim_host(shim_host, wsl_distro):
        if shim_host:
            return shim_host

        configured_host = os.environ.get(
            "ORCHESTRATION_SHIM_HOST"
        )
        if configured_host:
            return configured_host

        if os.name == "nt":
            try:
                result = subprocess.run(
                    [
                        "wsl.exe",
                        "-d",
                        wsl_distro,
                        "hostname",
                        "-I",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                addresses = result.stdout.strip().split()
                if addresses:
                    return addresses[0]

            except (OSError, subprocess.SubprocessError):
                pass

        return "localhost"

    def __init__(
        self,
        zmq_port=5563,
        shim_port=6000,
        shim_host=None,
        wsl_distro="opp_env",
        poll_interval=0.05,
        metrics_file="metrics.csv",
    ):
        self.zmq_port = zmq_port
        self.shim_port = shim_port
        self.shim_host = self._resolve_shim_host(
            shim_host,
            wsl_distro,
        )
        self.poll_interval = poll_interval
        self.metrics_file = Path(metrics_file)

        self.total_commands = 0
        self.successful = 0
        self.failed = 0
        self.action_counts = {}
        self.last_action_time = {}
        self.settling_time = 0.1  # seconds

        # REP socket to receive commands from OptimizationAgent
        self.context = zmq.Context()

        self.rep_socket = self.context.socket(zmq.REP)
        self.rep_socket.setsockopt(zmq.LINGER, 0)
        self.rep_socket.bind(
            f"tcp://*:{self.zmq_port}"
        )
        print(f"[OrchestrationAgent] Listening for commands on tcp://*:{self.zmq_port}")

        # REQ socket to send commands to OMNeT++ shim
        self.shim_socket = None
        self._connect_shim_socket()
        print(
            "[OrchestrationAgent] Connected to "
            f"OrchestrationShim at "
            f"tcp://{self.shim_host}:{self.shim_port}"
        )

        # Ensure metrics file exists
        if not self.metrics_file.exists():
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "total", "successful", "failed", "last_action", "last_result", "action_counts"])

    def now_iso(self):
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    def _connect_shim_socket(self):
        """Create a fresh REQ connection to OrchestrationShim."""

        if self.shim_socket is not None:
            self.shim_socket.close(linger=0)

        self.shim_socket = self.context.socket(zmq.REQ)
        self.shim_socket.setsockopt(zmq.LINGER, 0)
        self.shim_socket.setsockopt(zmq.SNDTIMEO, 3000)
        self.shim_socket.setsockopt(zmq.RCVTIMEO, 3000)
        self.shim_socket.connect(
            f"tcp://{self.shim_host}:{self.shim_port}"
        )

    def _send_to_shim(self, command):
        """Send one command and recover the REQ socket on failure."""

        try:
            self.shim_socket.send_string(
                json.dumps(command)
            )
            reply_text = self.shim_socket.recv_string()
            return json.loads(reply_text)

        except zmq.Again:
            # A timed-out REQ socket cannot safely send another
            # request until its request/reply state is reset.
            self._connect_shim_socket()

            return {
                "accepted": False,
                "error": "orchestration_shim_timeout",
            }

        except json.JSONDecodeError as error:
            self._connect_shim_socket()

            return {
                "accepted": False,
                "error": (
                    f"invalid_shim_response: {error}"
                ),
            }

        except Exception as error:
            self._connect_shim_socket()

            return {
                "accepted": False,
                "error": str(error),
            }

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
        """Normalize and forward an orchestration command."""

        self.total_commands += 1
        action = "unknown"
        result_name = "ERROR"

        try:
            if isinstance(payload, str):
                request = json.loads(payload)
            elif isinstance(payload, dict):
                request = payload
            else:
                raise ValueError(
                    "Payload must be JSON text or a dictionary"
                )

            action = str(request.get("action", ""))
            target = str(request.get("target", ""))
            control_mode = str(
                request.get("control_mode", "unknown")
            )

            self.action_counts[action] = (
                self.action_counts.get(action, 0) + 1
            )

            # --------------------------------------------------
            # Normal optimization configuration
            # --------------------------------------------------
            if action == "applyConfiguration":
                params = dict(request.get("param", {}))

                bs_id = str(
                    params.get("bs_id", target)
                )

                if not bs_id or bs_id == "unknown":
                    raise ValueError(
                        "applyConfiguration requires a BS ID"
                    )

                shim_command = {
                    "bs_id": bs_id,
                }

                if "tx_power_dbm" in params:
                    shim_command["tx_power_dbm"] = float(
                        params["tx_power_dbm"]
                    )

                if "es_mode" in params:
                    shim_command["es_mode"] = str(
                        params["es_mode"]
                    )

                # Retain these values for future shim support.
                if "bandwidth_mhz" in params:
                    shim_command["bandwidth_mhz"] = int(
                        params["bandwidth_mhz"]
                    )

                if "carrier_count" in params:
                    shim_command["carrier_count"] = int(
                        params["carrier_count"]
                    )

                if "es_modes" in params:
                    shim_command["es_modes"] = list(
                        params["es_modes"]
                    )

            # --------------------------------------------------
            # AI healing after a rejected optimization
            # --------------------------------------------------
            elif action == "executeHealingPlan":
                bs_id = target

                if not bs_id or bs_id == "unknown":
                    raise ValueError(
                        "executeHealingPlan requires a BS ID"
                    )

                recommendations = request.get(
                    "structured_recommendations", {}
                )
                config_modifications = recommendations.get(
                    "config_modifications", {}
                )

                # Begin with a conservative, QoS-safe state.
                tx_power_dbm = float(
                    config_modifications.get(
                        "tx_power_dbm",
                        config_modifications.get(
                            "target_tx_power_dbm",
                            43.0,
                        ),
                    )
                )
                es_mode = str(
                    config_modifications.get(
                        "es_mode", "active"
                    )
                )

                # Critical latency/reliability violations must
                # disable sleep modes.
                for violation in request.get(
                    "critical_violations", []
                ):
                    message = str(
                        violation.get("message", "")
                    ).lower()

                    if any(
                        term in message
                        for term in (
                            "latency",
                            "reliability",
                            "packet loss",
                            "throughput",
                            "urllc",
                        )
                    ):
                        es_mode = "active"

                    if any(
                        term in message
                        for term in (
                            "transmission power",
                            "tx power",
                            "43 dbm",
                        )
                    ):
                        tx_power_dbm = min(
                            tx_power_dbm, 43.0
                        )

                shim_command = {
                    "bs_id": bs_id,
                    "tx_power_dbm": tx_power_dbm,
                    "es_mode": es_mode,
                }

            # --------------------------------------------------
            # Limited legacy-command compatibility
            # --------------------------------------------------
            elif action in {
                "setTxPower",
                "enterSleepMode",
            }:
                params = dict(request.get("param", {}))

                bs_id = target
                if bs_id.startswith("gnb["):
                    bs_id = bs_id[4:-1]

                shim_command = {
                    "bs_id": bs_id,
                }

                if action == "setTxPower":
                    shim_command["tx_power_dbm"] = float(
                        params["power"]
                    )
                else:
                    shim_command["es_mode"] = str(
                        params["mode"]
                    )

            else:
                raise ValueError(
                    f"Unsupported orchestration action: {action}"
                )

            shim_reply = self._send_to_shim(
                shim_command
            )

            accepted = bool(
                shim_reply.get("accepted", False)
            )
            applied_count = int(
                shim_reply.get("applied_count", 0)
            )

            if accepted and applied_count > 0:
                result_name = "OK"
                self.successful += 1
            else:
                result_name = "SHIM_REJECTED"
                self.failed += 1

            response = {
                "status": (
                    "OK"
                    if result_name == "OK"
                    else "ERROR"
                ),
                "detail": result_name,
                "action": action,
                "control_mode": control_mode,
                "target": target,
                "shim_command": shim_command,
                "shim_reply": shim_reply,
                "total": self.total_commands,
                "successful": self.successful,
                "failed": self.failed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except json.JSONDecodeError as error:
            self.failed += 1
            result_name = "INVALID_JSON"

            response = {
                "status": "ERROR",
                "detail": result_name,
                "error": str(error),
                "total": self.total_commands,
                "successful": self.successful,
                "failed": self.failed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as error:
            self.failed += 1
            result_name = "PROCESSING_ERROR"

            response = {
                "status": "ERROR",
                "detail": result_name,
                "action": action,
                "error": str(error),
                "total": self.total_commands,
                "successful": self.successful,
                "failed": self.failed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        self.append_metrics_csv(
            action,
            result_name,
        )

        return json.dumps(response)
    
    def run(self):
        """Listen for commands and support clean shutdown."""
        print(
            "[OrchestrationAgent] Starting main loop..."
        )

        poller = zmq.Poller()
        poller.register(
            self.rep_socket,
            zmq.POLLIN,
        )

        try:
            while True:
                events = dict(
                    poller.poll(timeout=250)
                )

                if self.rep_socket not in events:
                    continue

                request = self.rep_socket.recv_string()
                print(
                    "[OrchestrationAgent] Received: "
                    f"{request}"
                )

                response = self.process_command(
                    request
                )

                self.rep_socket.send_string(response)
                print(
                    "[OrchestrationAgent] Sent response: "
                    f"{response[:100]}..."
                )

        except KeyboardInterrupt:
            print(
                "[OrchestrationAgent] Shutting down..."
            )

        finally:
            poller.unregister(self.rep_socket)
            self.rep_socket.close(linger=0)

            if self.shim_socket is not None:
                self.shim_socket.close(linger=0)

            self.context.term()

if __name__ == "__main__":
    agent = OrchestrationAgent(zmq_port=5563)  # Ensure port matches the one set in OptimizationAgent
    agent.run()
