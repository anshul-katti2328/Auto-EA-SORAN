"""
Collect reproducible per-cell Simu5G metrics for deterministic
and AI experiment runs.
"""

import argparse
import csv
import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import zmq


CSV_FIELDS = [
    "run_id",
    "scenario",
    "replicate",
    "seed",
    "received_at_utc",
    "schema_version",
    "twin_id",
    "sim_time_s",
    "physical_sim_time_s",
    "logical_time_scale",
    "hour_of_day",
    "unassociated_ues",
    "kpi_source",
    "energy_source",
    "bs_id",
    "simu5g_cell_index",
    "zone",
    "zone_name",
    "bs_type",
    "ru_type",
    "mode",
    "frequency_mhz",
    "bandwidth_mhz",
    "subcarrier_spacing_khz",
    "carrier_count",
    "requested_bandwidth_mhz",
    "requested_carrier_count",
    "num_antennas",
    "available_prbs",
    "used_prbs",
    "prb_util_pct",
    "load",
    "generated_dl_mbps",
    "dl_mbps",
    "ul_mbps",
    "total_throughput_mbps",
    "measurement_interval_s",
    "dl_bits",
    "ul_bits",
    "active_ues",
    "avg_sinr_db",
    "avg_latency_ms",
    "packet_loss_pct",
    "spectral_efficiency_bps_hz",
    "tx_power_dbm",
    "physical_tx_power_dbm",
    "radio_sleeping",
    "es_mode",
    "es_modes",
    "energy_consumption_w",
    "interval_energy_j",
    "represented_interval_s",
    "represented_energy_j",
    "energy_efficiency_mbit_per_j",
    "traffic_level",
    "traffic_multiplier",
]


class MetricsCollector:
    """Subscribe to TwinMetricsAgent and persist raw evidence."""

    def __init__(
        self,
        endpoint: str,
        scenario: str,
        run_id: str,
        replicate: int,
        seed: int,
        output_dir: Path,
        idle_timeout_s: float,
        expected_cells: int,
    ):
        self.endpoint = endpoint
        self.scenario = scenario
        self.run_id = run_id
        self.replicate = replicate
        self.seed = seed
        self.output_dir = output_dir
        self.idle_timeout_s = idle_timeout_s
        self.expected_cells = expected_cells

        self.running = True
        self.received_snapshots = 0
        self.written_rows = 0
        self.last_message_time: Optional[float] = None

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.connect(endpoint)

        safe_run_id = "".join(
            character
            if character.isalnum() or character in "-_"
            else "_"
            for character in run_id
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.csv_path = (
            self.output_dir
            / f"{scenario}_{safe_run_id}_cells.csv"
        )
        self.raw_path = (
            self.output_dir
            / f"{scenario}_{safe_run_id}_raw.jsonl"
        )

    def stop(self, *_args):
        self.running = False

    def _make_row(
        self,
        payload: Dict[str, Any],
        cell: Dict[str, Any],
        received_at: str,
    ) -> Dict[str, Any]:
        row = {
            "run_id": self.run_id,
            "scenario": self.scenario,
            "replicate": self.replicate,
            "seed": self.seed,
            "received_at_utc": received_at,
            "schema_version": payload.get(
                "schema_version"
            ),
            "twin_id": payload.get("twin_id"),
            "sim_time_s": payload.get("sim_time_s"),
            "physical_sim_time_s": payload.get(
                "physical_sim_time_s"
            ),
            "logical_time_scale": payload.get(
                "logical_time_scale"
            ),
            "hour_of_day": cell.get(
                "hour_of_day",
                payload.get("hour_of_day"),
            ),
            "unassociated_ues": payload.get(
                "unassociated_ues"
            ),
            "kpi_source": cell.get(
                "kpi_source",
                payload.get("kpi_source"),
            ),
            "energy_source": cell.get(
                "energy_source",
                payload.get("energy_source"),
            ),
        }

        row.update(cell)

        es_modes = row.get("es_modes", [])
        row["es_modes"] = json.dumps(
            es_modes,
            separators=(",", ":"),
        )

        return row

    def run(self):
        logging.info(
            "Collecting %s metrics from %s",
            self.scenario,
            self.endpoint,
        )
        logging.info("CSV output: %s", self.csv_path)
        logging.info("Raw output: %s", self.raw_path)

        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        cell_count_warning_emitted = False

        with (
            self.csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as csv_file,
            self.raw_path.open(
                "w",
                encoding="utf-8",
            ) as raw_file,
        ):
            writer = csv.DictWriter(
                csv_file,
                fieldnames=CSV_FIELDS,
                extrasaction="ignore",
            )
            writer.writeheader()

            while self.running:
                events = dict(poller.poll(timeout=500))

                if self.socket not in events:
                    if (
                        self.last_message_time is not None
                        and time.monotonic()
                        - self.last_message_time
                        >= self.idle_timeout_s
                    ):
                        logging.info(
                            "No metrics for %.1f seconds; "
                            "assuming the simulation ended",
                            self.idle_timeout_s,
                        )
                        break

                    continue

                raw_message = self.socket.recv()
                self.last_message_time = time.monotonic()

                try:
                    payload = json.loads(
                        raw_message.decode("utf-8")
                    )
                except (UnicodeDecodeError, json.JSONDecodeError):
                    logging.exception(
                        "Discarding malformed metrics message"
                    )
                    continue

                if not isinstance(payload, dict):
                    logging.warning(
                        "Discarding non-object metrics payload"
                    )
                    continue

                cells = payload.get("cells", [])
                if not isinstance(cells, list):
                    logging.warning(
                        "Discarding payload with invalid cells"
                    )
                    continue

                if (
                    len(cells) != self.expected_cells
                    and not cell_count_warning_emitted
                ):
                    logging.warning(
                        "Expected %d cells but received %d",
                        self.expected_cells,
                        len(cells),
                    )
                    cell_count_warning_emitted = True

                received_at = datetime.now(
                    timezone.utc
                ).isoformat()

                raw_record = {
                    "collector": {
                        "run_id": self.run_id,
                        "scenario": self.scenario,
                        "replicate": self.replicate,
                        "seed": self.seed,
                        "received_at_utc": received_at,
                    },
                    "payload": payload,
                }
                raw_file.write(
                    json.dumps(
                        raw_record,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

                for cell in cells:
                    if not isinstance(cell, dict):
                        continue

                    writer.writerow(
                        self._make_row(
                            payload,
                            cell,
                            received_at,
                        )
                    )
                    self.written_rows += 1

                csv_file.flush()
                raw_file.flush()
                self.received_snapshots += 1

        logging.info(
            "Collection complete: %d snapshots, %d cell rows",
            self.received_snapshots,
            self.written_rows,
        )

    def close(self):
        self.socket.close(linger=0)
        self.context.term()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Collect TwinMetricsAgent data for one experiment run"
        )
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=["deterministic", "ai"],
    )
    parser.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5556",
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--replicate",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--expected-cells",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--idle-timeout-s",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--output-dir",
        default="../OUTPUTs/experiments/raw",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    run_id = args.run_id or datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (
            Path(__file__).resolve().parent
            / output_dir
        ).resolve()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | %(message)s"
        ),
    )

    collector = MetricsCollector(
        endpoint=args.endpoint,
        scenario=args.scenario,
        run_id=run_id,
        replicate=args.replicate,
        seed=args.seed,
        output_dir=output_dir,
        idle_timeout_s=max(
            1.0,
            args.idle_timeout_s,
        ),
        expected_cells=max(
            1,
            args.expected_cells,
        ),
    )

    signal.signal(signal.SIGINT, collector.stop)
    signal.signal(signal.SIGTERM, collector.stop)

    try:
        collector.run()
    finally:
        collector.close()

if __name__ == "__main__":
    main()