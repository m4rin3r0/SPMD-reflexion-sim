from typing import List


def _validate_attach_points(config:dict) -> List[float]:
    attach_points_raw = config.get("attach_points")
    if attach_points_raw is None:
        raise ValueError("'attach_points' is required and must be a list of positions.")
    if not isinstance(attach_points_raw, list):
        raise ValueError("'attach_points' must be a list of positions.")
    attach_points = [float(point) for point in attach_points_raw]
    expected_count = int(config["nodes"])
    if len(attach_points) != expected_count:
        raise ValueError(f"'attach_points' must contain exactly {expected_count} values, got {len(attach_points)}.")
    length = float(config["length"])
    for idx, point in enumerate(attach_points, start=1):
        if point < 0.0 or point > length:
            raise ValueError(f"attach_points[{idx}]={point} is outside the valid range [0, {length}].")
    for idx in range(1, len(attach_points)):
        if attach_points[idx] < attach_points[idx - 1]:
            raise ValueError("'attach_points' must be sorted in non-decreasing order.")
    return attach_points


def _validate_tx_node(config:dict) -> int:
    tx_node = int(config.get("tx_node", 1))
    if tx_node < 1 or tx_node > config["nodes"]:
        raise ValueError(f"tx_node must be within 1..{config['nodes']}")
    return tx_node