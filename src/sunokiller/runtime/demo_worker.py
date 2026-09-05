"""Small deterministic worker used by runtime-boundary tests."""

import time


def update_counter(payload):
    delay = float(payload.get("sleep_seconds", 0.0))
    if delay > 0:
        time.sleep(delay)
    value = int(payload.get("value", 0))
    return {
        "value": value + 1,
        "_state": {"counter": value + 1},
    }
