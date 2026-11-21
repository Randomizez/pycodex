"""Check a core wheel in a clean environment, outside the source checkout."""

from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys

import pycodex
from pycodex.portable.client import _decrypt_bundle, _encrypt_bundle
from pycodex.utils.crypto import aes_gcm_decrypt, aes_gcm_encrypt
from pycodex.utils.model_metadata import model_metadata


def main():
    root = Path(__file__).resolve().parents[1]
    assert Path(pycodex.__file__).resolve().parent != root / "pycodex"
    for name in (
        "Cryptodome", "Crypto", "PIL", "pyagy", "workspace_server", "pycodex_ws",
        "fastapi", "pydantic", "pydantic_core",
    ):
        assert find_spec(name) is None, name

    assert model_metadata("step-5-preview")["slug"] == "step-5-preview"
    context = pycodex.ContextManager(pycodex.ContextConfig(model="step-5-preview"))
    assert context.resolve_base_instructions().startswith("You are Codex")

    key, nonce = b"\0" * 32, b"\0" * 12
    ciphertext = bytes.fromhex("cea7403d4d606b6e074ec5d3baf39d18")
    tag = bytes.fromhex("d0d1c8a799996bf0265b98b5d48ab919")
    assert aes_gcm_encrypt(key, nonce, b"\0" * 16) == (ciphertext, tag)
    assert aes_gcm_decrypt(key, nonce, ciphertext, tag) == b"\0" * 16
    try:
        aes_gcm_decrypt(key, nonce, ciphertext, b"\0" * 16)
    except ValueError:
        pass
    else:
        raise AssertionError("AES-GCM accepted an invalid authentication tag")

    plaintext = bytes(range(256)) * 4096
    bundle = _encrypt_bundle(plaintext, "public wheel fixture")
    assert _decrypt_bundle(bundle, "public wheel fixture") == plaintext
    for arguments in (
        ["pycodex", "--help"],
        ["pycodex", "doctor", "--help"],
        ["pycodex.portable.server", "--help"],
        ["responses_server", "--help"],
    ):
        subprocess.run(
            [sys.executable, "-m"] + arguments,
            check=True, stdout=subprocess.DEVNULL,
        )
    print(
        "Core wheel: model resources, pure Python AES-GCM, "
        "portable round-trip and CLI smoke passed"
    )


if __name__ == "__main__":
    main()
