"""Small deterministic worker used by runtime-boundary tests."""

import time


def update_counter(payload):
    delay = float(payload.get("sleep_seconds", 0.0))
    if delay > 0:
        time.sleep(delay)
    value = int(payload.get("value", 0))
    result = {"value": value + 1}
    if payload.get("propose_state") is True:
        result["_state"] = {"counter": value + 1}
    return result
