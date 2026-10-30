"""Wire constants and content identifiers shared by portable client and server."""

import hashlib
import typing

DEFAULT_STORAGE_SERVER = "127.0.0.1:5577"
STORAGE_API_PREFIX = "/v1/storage"
HEALTHCHECK_PATH = "/healthz"
ENCRYPTED_BUNDLE_MAGIC = b"PCX1"
NONCE_LENGTH = 12
TOKEN_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def call_id_from_payload(payload: "bytes") -> "str":
    return base58_encode(hashlib.sha256(payload).digest()[:8])


def base58_encode(payload: "bytes") -> "str":
    number = int.from_bytes(payload, "big")
    if number == 0:
        return TOKEN_BASE58_ALPHABET[0]
    encoded: "typing.List[str]" = []
    while number:
        number, remainder = divmod(number, 58)
        encoded.append(TOKEN_BASE58_ALPHABET[remainder])
    encoded.reverse()
    prefix = TOKEN_BASE58_ALPHABET[0] * (len(payload) - len(payload.lstrip(b"\x00")))
    return prefix + "".join(encoded)
