
import random
import time
import uuid


def uuid7_string() -> 'str':
    timestamp_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand_a = random.getrandbits(12)
    rand_b = random.getrandbits(62)

    value = 0
    value |= timestamp_ms << 80
    value |= 0x7 << 76
    value |= rand_a << 64
    value |= 0b10 << 62
    value |= rand_b
    return str(uuid.UUID(int=value))
