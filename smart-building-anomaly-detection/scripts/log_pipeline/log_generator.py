import random
import pandas as pd
from datetime import timedelta

#Core log generator
def create_log(timestamp, building_id, event_type, event_code, severity, anomaly_link):
    return {
        "timestamp": timestamp,
        "building_id": building_id,
        "device_id": f"AHU_{random.randint(1,3)}",
        "subsystem": "HVAC",
        "event_type": event_type,
        "event_code": event_code,
        "severity": severity,
        "value": random.uniform(20, 30),
        "message": f"{event_code} occurred",
        # 0/1 link used as the weak label for sequence-level anomaly detection
        "anomaly_link": int(anomaly_link),
    }

#Normal logs
def generate_normal_logs(
    timestamp,
    building_id,
    *,
    ambiguous_normal_prob: float = 0.0,
):
    logs = []

    if random.random() < 0.3:
        logs.append(create_log(timestamp, building_id, "CONTROL", "SETPOINT_CHANGE", "INFO", 0))

    if random.random() < 0.2:
        logs.append(create_log(timestamp, building_id, "SYSTEM", "NETWORK_DELAY", "WARN", 0))

    # Hard-mode: sometimes emit an event that looks anomaly-like but is still labeled as normal (anomaly_link=0).
    if ambiguous_normal_prob > 0 and random.random() < ambiguous_normal_prob:
        ambiguous_choices = [
            ("CONTROL", "MODE_CHANGE", "INFO"),
            ("SENSOR", "DRIFT_DETECTED", "WARN"),
            ("FAULT", "OVERLOAD", "CRITICAL"),
            ("SYSTEM", "DEVICE_OFFLINE", "CRITICAL"),
            ("SENSOR", "NO_SIGNAL", "CRITICAL"),
        ]
        event_type, event_code, severity = random.choice(ambiguous_choices)
        logs.append(create_log(timestamp, building_id, event_type, event_code, severity, 0))

    return logs

#Anomaly sequences
def spike_sequence(timestamp, building_id):
    return [
        create_log(timestamp - timedelta(minutes=2), building_id, "CONTROL", "MODE_CHANGE", "INFO", 1),
        create_log(timestamp - timedelta(minutes=1), building_id, "CONTROL", "SETPOINT_CHANGE", "WARN", 1),
        create_log(timestamp, building_id, "FAULT", "OVERLOAD", "CRITICAL", 1),
    ]

def drift_sequence(timestamp, building_id):
    return [
        create_log(timestamp - timedelta(minutes=5), building_id, "SENSOR", "CALIBRATION_WARNING", "WARN", 1),
        create_log(timestamp, building_id, "SENSOR", "DRIFT_DETECTED", "WARN", 1),
    ]

def flatline_sequence(timestamp, building_id):
    return [
        create_log(timestamp - timedelta(minutes=2), building_id, "SYSTEM", "DEVICE_OFFLINE", "CRITICAL", 1),
        create_log(timestamp, building_id, "SENSOR", "NO_SIGNAL", "CRITICAL", 1),
    ]

def _maybe_truncate_sequence(seq: list, *, partial_sequence_prob: float, partial_sequence_max_events: int) -> list:
    """
    Hard-mode: sometimes drop the tail of an anomaly burst so the label isn't perfectly
    recoverable from a full deterministic pattern.
    """
    if partial_sequence_prob <= 0:
        return seq
    if len(seq) <= 1:
        return seq
    if random.random() < partial_sequence_prob:
        keep_max = max(1, min(partial_sequence_max_events, len(seq)))
        keep = random.randint(1, keep_max)
        return seq[:keep]
    return seq


def _normalize_burst_length(burst_length, seq_len: int) -> int:
    """Resolve burst_length config into a concrete keep length."""
    if isinstance(burst_length, tuple) and len(burst_length) == 2:
        lo = max(1, int(min(burst_length[0], burst_length[1])))
        hi = max(lo, int(max(burst_length[0], burst_length[1])))
        return min(seq_len, random.randint(lo, hi))
    return min(seq_len, max(1, int(burst_length)))


def _trim_groups_to_rate(all_logs: list, anomaly_group_indices: list, max_anomaly_rate: float) -> list:
    """
    Remove whole anomaly groups until anomaly rate <= max_anomaly_rate.
    Preserves semantic anomaly patterns instead of flipping labels.
    """
    if not all_logs:
        return all_logs

    ones = sum(int(log.get("anomaly_link", 0)) for log in all_logs)
    total = len(all_logs)
    if total == 0:
        return all_logs
    current = ones / total
    if current <= max_anomaly_rate or not anomaly_group_indices:
        return all_logs

    remove_mask = [False] * len(all_logs)
    candidate_groups = anomaly_group_indices[:]
    random.shuffle(candidate_groups)

    for group in candidate_groups:
        if current <= max_anomaly_rate:
            break
        group_anom = sum(int(all_logs[idx].get("anomaly_link", 0)) for idx in group if not remove_mask[idx])
        group_total = sum(1 for idx in group if not remove_mask[idx])
        if group_total == 0:
            continue
        for idx in group:
            remove_mask[idx] = True
        ones -= group_anom
        total -= group_total
        current = (ones / total) if total > 0 else 0.0

    return [log for i, log in enumerate(all_logs) if not remove_mask[i]]


def _build_anomaly_sequence(timestamp, building_id, anomaly_type: str) -> list:
    if anomaly_type == "spike":
        return spike_sequence(timestamp, building_id)
    if anomaly_type == "drift":
        return drift_sequence(timestamp, building_id)
    if anomaly_type == "flatline":
        return flatline_sequence(timestamp, building_id)
    if anomaly_type == "drop":
        return drift_sequence(timestamp, building_id)
    return []


#Main generator
def generate_logs(
    sensor_df: pd.DataFrame,
    *,
    anomaly_probability: float = 0.35,
    burst_frequency: float = 0.6,
    burst_length=2,
    min_anomaly_rate: float = 0.08,
    target_anomaly_rate: float = 0.10,
    max_anomaly_rate: float = 0.20,
    partial_sequence_prob: float = 0.05,
    partial_sequence_max_events: int = 2,
    ambiguous_normal_prob: float = 0.02,
    spurious_anomaly_link_flip_prob: float = 0.0,
) -> list:
    """
    Generate structured logs with controllable anomaly prevalence.

    Notes:
    - anomaly_probability and burst_frequency jointly control how often flagged sensor rows
      produce anomaly bursts.
    - burst_length controls the number of events kept from each anomaly template.
    - min_anomaly_rate can backfill a small number of structured anomaly groups when a
      building has too few anomaly events.
    - target_anomaly_rate is a soft upper guide used to attenuate anomaly injection.
    - max_anomaly_rate is a hard cap enforced by removing full anomaly groups.
    """
    all_logs = []
    anomaly_group_indices = []
    anomaly_candidates = []

    for _, row in sensor_df.iterrows():
        t = pd.to_datetime(row["timestamp"])
        building_id = row["building_id"]

        # Always add normal logs
        all_logs.extend(
            generate_normal_logs(
                t,
                building_id,
                ambiguous_normal_prob=ambiguous_normal_prob,
            )
        )

        # Add anomaly logs
        if row["anomaly_flag"] == 1:
            anomaly_candidates.append((t, building_id, str(row.get("anomaly_type", "drift"))))
            # Adaptive attenuation keeps prevalence near target across buildings.
            current_total = max(1, len(all_logs))
            current_anom = sum(int(log.get("anomaly_link", 0)) for log in all_logs)
            current_rate = current_anom / current_total
            attenuation = min(1.0, target_anomaly_rate / max(current_rate, 1e-6))
            effective_p = max(0.0, min(1.0, anomaly_probability * burst_frequency * attenuation))

            if random.random() < effective_p:
                seq = _build_anomaly_sequence(t, building_id, str(row.get("anomaly_type", "drift")))

                if seq:
                    keep = _normalize_burst_length(burst_length, len(seq))
                    seq = seq[:keep]
                    seq = _maybe_truncate_sequence(
                        seq,
                        partial_sequence_prob=partial_sequence_prob,
                        partial_sequence_max_events=partial_sequence_max_events,
                    )

                    start_idx = len(all_logs)
                    all_logs.extend(seq)
                    end_idx = len(all_logs)
                    anomaly_group_indices.append(list(range(start_idx, end_idx)))

    # Optional minimal label noise (default off).
    if spurious_anomaly_link_flip_prob > 0:
        for log in all_logs:
            if random.random() < spurious_anomaly_link_flip_prob:
                log["anomaly_link"] = int(1 - int(log.get("anomaly_link", 0)))

    # Soft floor: add a few extra structured groups if anomaly rate is too low.
    if anomaly_candidates and min_anomaly_rate > 0:
        attempts = 0
        max_attempts = min(500, len(anomaly_candidates) * 4)
        while attempts < max_attempts:
            attempts += 1
            total = len(all_logs)
            anom = sum(int(log.get("anomaly_link", 0)) for log in all_logs)
            rate = (anom / total) if total > 0 else 0.0
            if rate >= min_anomaly_rate:
                break

            t, building_id, anomaly_type = random.choice(anomaly_candidates)
            seq = _build_anomaly_sequence(t, building_id, anomaly_type)
            if not seq:
                continue
            keep = _normalize_burst_length(burst_length, len(seq))
            seq = seq[:keep]
            start_idx = len(all_logs)
            all_logs.extend(seq)
            end_idx = len(all_logs)
            anomaly_group_indices.append(list(range(start_idx, end_idx)))

    # Hard cap: keep anomaly rates from dominating.
    all_logs = _trim_groups_to_rate(all_logs, anomaly_group_indices, max_anomaly_rate=max_anomaly_rate)
    return all_logs