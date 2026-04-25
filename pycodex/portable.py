
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import quote, urlparse

import requests
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import typing

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 path
    import tomli as tomllib

DEFAULT_STORAGE_SERVER = "127.0.0.1:5577"
STORAGE_SERVER_ENV = "PYCODEX_STORAGE_SERVER"
STORAGE_ROOT_ENV = "PYCODEX_STORAGE_ROOT"
STORAGE_CACHE_DIRNAME = ".pycodex-storage"
DEFAULT_ENTRY_CONFIG = "config.toml"
STORAGE_API_PREFIX = "/v1/storage"
HEALTHCHECK_PATH = "/healthz"
ENCRYPTED_BUNDLE_MAGIC = b"PCX1"
NONCE_LENGTH = 12
TOKEN_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
ALLOWED_TOP_LEVEL_FILES = (
    DEFAULT_ENTRY_CONFIG,
    ".env",
    "AGENTS.md",
    "AGENTS.override.md",
)
ALLOWED_TOP_LEVEL_DIRS = ("skills",)


class RemoteStorageError(RuntimeError):
    pass


ProgressHandler = Callable[[str], None]


def upload_codex_home(
    put_text: 'typing.Union[str, None]' = None,
    event_handler: 'typing.Union[ProgressHandler, None]' = None,
) -> 'str':
    source_dir, server = _parse_put_spec(put_text)
    resolved_source_dir = resolve_put_source_dir(source_dir)
    server_address, base_url = resolve_storage_server(server)
    emit = event_handler or (lambda _message: None)
    emit(f"[put] source: {resolved_source_dir}")
    emit(f"[put] checking server: {server_address}")
    _check_storage_server(server_address, base_url)
    bundle_bytes = _build_bundle_bytes(resolved_source_dir, emit)
    secret = _base58_encode(os.urandom(16))
    encrypted_bundle = _encrypt_bundle(bundle_bytes, secret)
    sha256 = hashlib.sha256(encrypted_bundle).hexdigest()
    call_id = _call_id_from_payload(encrypted_bundle)
    call_spec = f"{secret}-{call_id}@{server_address}"
    emit(f"[put] bundle: {len(bundle_bytes)} bytes plaintext, sha256={sha256}")
    emit(f"[put] uploading ciphertext to {server_address}")
    try:
        response = requests.post(
            f"{base_url}/put",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(encrypted_bundle)),
                "X-Pycodex-Sha256": sha256,
            },
            data=encrypted_bundle,
            timeout=(5.0, 120.0),
        )
    except requests.RequestException as exc:
        raise RemoteStorageError(f"storage upload failed: {exc}") from exc
    if response.status_code >= 400:
        raise RemoteStorageError(
            f"storage upload failed with status {response.status_code}"
        )
    emit(f"[put] uploaded: {call_spec}")
    return call_spec


def bootstrap_called_home(
    call_text: 'str',
    storage_root: 'typing.Union[typing.Union[str, Path], None]' = None,
) -> 'Path':
    secret, call_id, server_address, base_url = _parse_call_spec(call_text)
    root = resolve_storage_root(storage_root)
    cache_key = hashlib.sha256(call_text.strip().encode("utf-8")).hexdigest()[:16]
    cache_dir = root / cache_key
    metadata_path = cache_dir / "metadata.json"
    home_dir = cache_dir / "home"
    metadata = _load_cached_metadata(metadata_path)
    cached_config_path = home_dir / DEFAULT_ENTRY_CONFIG
    if (
        cached_config_path.is_file()
        and metadata.get("call_id") == call_id
        and metadata.get("server_address") == server_address
    ):
        return cached_config_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    encrypted_bundle = _download_encrypted_bundle(base_url, call_id)
    bundle_bytes = _decrypt_bundle(encrypted_bundle, secret)
    with tempfile.TemporaryDirectory(prefix="pycodex-home-") as tmpdir:
        extract_root = Path(tmpdir)
        _extract_bundle_bytes(bundle_bytes, extract_root)
        extracted_home = _resolve_extracted_home(extract_root)
        if home_dir.exists():
            shutil.rmtree(home_dir)
        shutil.copytree(str(extracted_home), str(home_dir))
    metadata_path.write_text(
        json.dumps(
            {
                "call_id": call_id,
                "server_address": server_address,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return home_dir / DEFAULT_ENTRY_CONFIG


def resolve_put_source_dir(source_dir: 'typing.Union[typing.Union[str, Path], None]') -> 'Path':
    if source_dir is None or str(source_dir).strip() == "":
        candidate = Path.home() / ".codex"
    else:
        candidate = Path(source_dir).expanduser()
    resolved = candidate.resolve()
    config_path = resolved / DEFAULT_ENTRY_CONFIG
    if not resolved.is_dir():
        raise RemoteStorageError(f"Codex home is not a directory: {resolved}")
    if not config_path.is_file():
        raise RemoteStorageError(
            f"Codex home is missing required file: {config_path}"
        )
    return resolved


def resolve_storage_root(storage_root: 'typing.Union[typing.Union[str, Path], None]' = None) -> 'Path':
    if storage_root is not None:
        return Path(storage_root).expanduser().resolve()
    env_value = os.environ.get(STORAGE_ROOT_ENV, "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return _discover_project_root() / STORAGE_CACHE_DIRNAME


def resolve_storage_server(server: 'typing.Union[str, None]' = None) -> 'typing.Tuple[str, str]':
    raw_value = (server or os.environ.get(STORAGE_SERVER_ENV) or "").strip()
    if not raw_value:
        raw_value = DEFAULT_STORAGE_SERVER
    if "://" in raw_value:
        parsed = urlparse(raw_value)
        if not parsed.scheme or not parsed.netloc:
            raise RemoteStorageError(f"invalid storage server: {raw_value}")
        return parsed.netloc, f"{parsed.scheme}://{parsed.netloc}{STORAGE_API_PREFIX}"
    if "/" in raw_value:
        raise RemoteStorageError(f"invalid storage server: {raw_value}")
    return raw_value, f"http://{raw_value}{STORAGE_API_PREFIX}"


def _build_bundle_bytes(root: 'Path', emit: 'ProgressHandler') -> 'bytes':
    files = _collect_upload_files(root)
    emit("[put] mode: whitelist")
    emit(f"[put] packing {len(files)} files")
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative_path in files:
            emit(f"[put] file: {relative_path}")
            archive.write(root / relative_path, relative_path)
    return buffer.getvalue()


def _collect_upload_files(root: 'Path') -> 'typing.List[str]':
    included: 'typing.Set[str]' = set()
    for relative_name in ALLOWED_TOP_LEVEL_FILES:
        candidate = root / relative_name
        if candidate.is_file():
            included.add(relative_name)
    for relative_dir in ALLOWED_TOP_LEVEL_DIRS:
        candidate = root / relative_dir
        if candidate.is_dir():
            for path in sorted(candidate.rglob("*")):
                if path.is_file():
                    included.add(path.relative_to(root).as_posix())
    included.update(_collect_config_referenced_files(root))
    return sorted(included)


def _collect_config_referenced_files(root: 'Path') -> 'typing.Set[str]':
    config_path = root / DEFAULT_ENTRY_CONFIG
    if not config_path.is_file():
        return set()
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    referenced: 'typing.Set[str]' = set()
    candidates = [data]
    profiles = data.get("profiles")
    if isinstance(profiles, dict):
        candidates.extend(
            profile_data for profile_data in profiles.values() if isinstance(profile_data, dict)
        )
    for candidate in candidates:
        model_instructions_file = candidate.get("model_instructions_file")
        if not isinstance(model_instructions_file, str):
            continue
        normalized = _normalize_optional_relative_file(root, model_instructions_file)
        if normalized is not None:
            referenced.add(normalized)
    return referenced


def _normalize_optional_relative_file(root: 'Path', value: 'str') -> 'typing.Union[str, None]':
    candidate = Path(value)
    if candidate.is_absolute():
        return None
    normalized = _normalize_member_path(candidate.as_posix(), field_name="config path")
    root_resolved = root.resolve()
    resolved = (root / normalized).resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise RemoteStorageError(f"config path points outside Codex home: {value}")
    if not resolved.is_file():
        return None
    return resolved.relative_to(root_resolved).as_posix()


def _encrypt_bundle(bundle_bytes: 'bytes', secret: 'str') -> 'bytes':
    nonce = os.urandom(NONCE_LENGTH)
    ciphertext = AESGCM(_encryption_key(secret)).encrypt(nonce, bundle_bytes, None)
    return ENCRYPTED_BUNDLE_MAGIC + nonce + ciphertext


def _decrypt_bundle(payload: 'bytes', secret: 'str') -> 'bytes':
    if not payload.startswith(ENCRYPTED_BUNDLE_MAGIC):
        raise RemoteStorageError("stored bundle is not a recognized encrypted payload")
    nonce = payload[len(ENCRYPTED_BUNDLE_MAGIC) : len(ENCRYPTED_BUNDLE_MAGIC) + NONCE_LENGTH]
    ciphertext = payload[len(ENCRYPTED_BUNDLE_MAGIC) + NONCE_LENGTH :]
    try:
        return AESGCM(_encryption_key(secret)).decrypt(nonce, ciphertext, None)
    except InvalidTag as exc:
        raise RemoteStorageError("call secret is invalid or bundle is corrupted") from exc


def _encryption_key(secret: 'str') -> 'bytes':
    return hashlib.sha256(secret.encode("utf-8")).digest()


def _call_id_from_payload(payload: 'bytes') -> 'str':
    return _base58_encode(hashlib.sha256(payload).digest()[:8])


def _base58_encode(payload: 'bytes') -> 'str':
    number = int.from_bytes(payload, "big")
    if number == 0:
        return TOKEN_BASE58_ALPHABET[0]
    encoded: 'typing.List[str]' = []
    while number:
        number, remainder = divmod(number, 58)
        encoded.append(TOKEN_BASE58_ALPHABET[remainder])
    encoded.reverse()
    prefix = TOKEN_BASE58_ALPHABET[0] * (len(payload) - len(payload.lstrip(b"\x00")))
    return prefix + "".join(encoded)


def _parse_put_spec(put_text: 'typing.Union[str, None]') -> 'typing.Tuple[typing.Union[str, None], typing.Union[str, None]]':
    raw_value = (put_text or "").strip()
    if not raw_value:
        return None, None
    if raw_value.startswith("@"):
        server = raw_value[1:].strip()
        if not server:
            raise RemoteStorageError("put spec is missing server after @")
        return None, server
    if "@" in raw_value:
        source_text, server = raw_value.rsplit("@", 1)
        if not source_text.strip():
            raise RemoteStorageError("put spec is missing source before @")
        if not server.strip():
            raise RemoteStorageError("put spec is missing server after @")
        return source_text.strip(), server.strip()
    return raw_value, None


def _parse_call_spec(call_text: 'str') -> 'typing.Tuple[str, str, str, str]':
    raw_value = call_text.strip()
    if not raw_value or "@" not in raw_value:
        raise RemoteStorageError("call spec must look like <secret>-<call_id>@<host:port>")
    secret_and_call_id, server_text = raw_value.rsplit("@", 1)
    if "-" not in secret_and_call_id:
        raise RemoteStorageError("call spec must include secret and call_id")
    secret, call_id = secret_and_call_id.rsplit("-", 1)
    if not secret or not call_id:
        raise RemoteStorageError("call spec is missing secret or call_id")
    server_address, base_url = resolve_storage_server(server_text)
    return secret, call_id, server_address, base_url


def _download_encrypted_bundle(base_url: 'str', call_id: 'str') -> 'bytes':
    url = f"{base_url}/call/{quote(call_id, safe='')}"
    try:
        response = requests.get(url, timeout=(5.0, 120.0))
    except requests.RequestException as exc:
        raise RemoteStorageError(f"call download failed: {exc}") from exc
    if response.status_code == 404:
        raise RemoteStorageError(f"call id not found: {call_id}")
    if response.status_code >= 400:
        raise RemoteStorageError(f"call download failed with status {response.status_code}")
    payload = response.content
    expected_sha256 = response.headers.get("X-Pycodex-Sha256", "").strip().lower() or None
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RemoteStorageError("downloaded bundle checksum mismatch")
    return payload


def _extract_bundle_bytes(bundle_bytes: 'bytes', destination: 'Path') -> 'None':
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    try:
        archive = zipfile.ZipFile(BytesIO(bundle_bytes))
    except zipfile.BadZipFile as exc:
        raise RemoteStorageError("decrypted bundle is not a valid zip archive") from exc
    with archive:
        for info in archive.infolist():
            member_name = info.filename
            if not member_name or member_name.endswith("/"):
                continue
            _normalize_member_path(member_name, field_name="bundle member")
            target_path = (destination_resolved / member_name).resolve()
            if target_path != destination_resolved and destination_resolved not in target_path.parents:
                raise RemoteStorageError("bundle contains unsafe paths")
        archive.extractall(destination)


def _resolve_extracted_home(extract_root: 'Path') -> 'Path':
    direct_config = extract_root / DEFAULT_ENTRY_CONFIG
    if direct_config.is_file():
        return extract_root
    children = [child for child in extract_root.iterdir() if child.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir() and (children[0] / DEFAULT_ENTRY_CONFIG).is_file():
        return children[0]
    raise RemoteStorageError("bundle is missing required config file after extraction")


def _load_cached_metadata(metadata_path: 'Path') -> 'typing.Dict[str, object]':
    if not metadata_path.is_file():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _check_storage_server(server_address: 'str', base_url: 'str') -> 'None':
    parsed = urlparse(base_url)
    health_url = f"{parsed.scheme}://{parsed.netloc}{HEALTHCHECK_PATH}"
    try:
        response = requests.get(health_url, timeout=(3.0, 3.0))
    except requests.RequestException as exc:
        raise RemoteStorageError(
            f"storage server preflight failed for {server_address}: {exc}"
        ) from exc
    if response.status_code >= 400:
        raise RemoteStorageError(
            f"storage server preflight failed for {server_address}: status {response.status_code}"
        )


def _discover_project_root(start: 'typing.Union[Path, None]' = None) -> 'Path':
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "pycodex").is_dir():
            return candidate
    return current


def _normalize_member_path(value: 'str', field_name: 'str') -> 'str':
    path = PurePosixPath(value)
    if not value or path.is_absolute():
        raise RemoteStorageError(f"{field_name} must be relative")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise RemoteStorageError(f"{field_name} contains invalid path segments")
    return path.as_posix()
