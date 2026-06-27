"""
Deterministic energy-conservation controller for Auto EA-SORAN.

This controller uses fixed operator rules. It does not use the LLM,
FeedbackAgent, machine learning, or NSGA-II optimization.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class RuleBasedDecision:
    tx_power: float
    bandwidth: int
    carrier_count: int
    es_mode: str
    es_modes: List[int]
    rule_name: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def select_rule_based_config(
    load: float,
    latency_ms: float,
    packet_loss_pct: float,
    active_ues: int = 0,
    current_tx_power: float = 43.0,
    current_bandwidth: int = 100,
    current_carrier_count: int = 1,
    bs_type: str = "macro",
    max_latency_ms: float = 20.0,
    max_packet_loss_pct: float = 1.0,
) -> RuleBasedDecision:
    """
    Select a deterministic energy-saving configuration.

    Args:
        load:
            Normalized PRB utilization in the range 0.0 to 1.0.
        latency_ms:
            Current average latency in milliseconds.
        packet_loss_pct:
            Current packet-loss percentage.
        current_tx_power:
            Current transmission power in dBm.
        current_bandwidth:
            Current configured bandwidth in MHz.
        current_carrier_count:
            Current active carrier count.
        bs_type:
            Either "macro" or "small_cell".
        max_latency_ms:
            Operator-defined maximum acceptable latency.
        max_packet_loss_pct:
            Operator-defined maximum acceptable packet loss.
        active_ues:
            Number of UEs currently served by the cell.

    Returns:
        RuleBasedDecision containing the selected configuration.
    """

    load = max(0.0, min(1.0, float(load)))
    latency_ms = max(0.0, float(latency_ms))
    packet_loss_pct = max(0.0, float(packet_loss_pct))
    active_ues = max(0, int(active_ues))

    # Use a lower power ceiling for small cells.
    # Rated power ceiling. This allows recovery after a low-power state.
    power_ceiling = 33.0 if bs_type == "small_cell" else 43.0

    # QoS protection always overrides energy-saving rules.
    if (
        latency_ms > max_latency_ms
        or packet_loss_pct > max_packet_loss_pct
        or load >= 0.85
    ):
        return RuleBasedDecision(
            tx_power=power_ceiling,
            bandwidth=current_bandwidth,
            carrier_count=current_carrier_count,
            es_mode="active",
            es_modes=[0, 0, 0, 0, 0, 0],
            rule_name="qos_safety_override",
            reason=(
                "High load or QoS threshold violation; "
                "disable energy-saving modes."
            ),
        )

    if load < 0.15 and active_ues == 0:
        return RuleBasedDecision(
            tx_power=max(10.0, power_ceiling - 15.0),
            bandwidth=current_bandwidth,
            carrier_count=current_carrier_count,
            es_mode="sleep",
            es_modes=[1, 0, 0, 0, 0, 0],
            rule_name="empty_cell_deep_sleep",
            reason=(
                "PRB utilization is below 15 percent and "
                "the cell serves no UEs."
            ),
        )

    if load < 0.15:
        return RuleBasedDecision(
            tx_power=max(15.0, power_ceiling - 8.0),
            bandwidth=current_bandwidth,
            carrier_count=current_carrier_count,
            es_mode="light_sleep",
            es_modes=[0, 0, 0, 0, 0, 1],
            rule_name="active_ue_low_load_protection",
            reason=(
                "PRB utilization is below 15 percent, but "
                "served UEs prevent deep sleep."
            ),
        )

    if load < 0.30:
        return RuleBasedDecision(
            tx_power=max(15.0, power_ceiling - 8.0),
            bandwidth=current_bandwidth,
            carrier_count=current_carrier_count,
            es_mode="light_sleep",
            es_modes=[0, 0, 0, 0, 0, 1],
            rule_name="low_load_light_sleep",
            reason="PRB utilization is between 15 and 30 percent.",
        )

    if load < 0.45:
        return RuleBasedDecision(
            tx_power=max(20.0, power_ceiling - 5.0),
            bandwidth=current_bandwidth,
            carrier_count=current_carrier_count,
            es_mode="mimo_reduce",
            es_modes=[0, 0, 1, 0, 0, 0],
            rule_name="medium_low_load_mimo_reduction",
            reason="PRB utilization is between 30 and 45 percent.",
        )

    if load < 0.65:
        return RuleBasedDecision(
            tx_power=max(25.0, power_ceiling - 3.0),
            bandwidth=current_bandwidth,
            carrier_count=current_carrier_count,
            es_mode="micro_dtx",
            es_modes=[0, 0, 0, 1, 0, 0],
            rule_name="medium_load_micro_dtx",
            reason="PRB utilization is between 45 and 65 percent.",
        )

    return RuleBasedDecision(
        tx_power=power_ceiling,
        bandwidth=current_bandwidth,
        carrier_count=current_carrier_count,
        es_mode="active",
        es_modes=[0, 0, 0, 0, 0, 0],
        rule_name="high_load_active",
        reason=(
            "PRB utilization is above 65 percent; "
            "preserve capacity."
        ),
    )