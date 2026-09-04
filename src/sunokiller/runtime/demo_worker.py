"""Small deterministic worker used by runtime-boundary tests."""


def update_counter(payload):
    value = int(payload.get("value", 0))
    return {
        "value": value + 1,
        "_state": {"counter": value + 1},
    }
