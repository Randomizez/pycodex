import json
import os
import sys
import threading
import time
import typing


class TrajectoryDumpWriter:
    ENV_VAR = "PYCODEX_DUMP"

    def __init__(self, root_dir: 'str') -> 'None':
        self._root_dir = os.path.abspath(root_dir)
        self._dump_path = os.path.join(self._root_dir, "dump.jsonl")
        self._lock = threading.Lock()
        os.makedirs(self._root_dir, exist_ok=True)

    @classmethod
    def from_env(cls) -> 'typing.Union[TrajectoryDumpWriter, None]':
        root_dir = str(os.environ.get(cls.ENV_VAR, "") or "").strip()
        if not root_dir:
            return None
        return cls(root_dir)

    def wrap_stream(self, outcomming_stream):
        def iter_stream():
            capture = _TrajectoryCapture(self, time.time())
            try:
                for chunk in outcomming_stream:
                    capture.observe_chunk(chunk)
                    yield chunk
            finally:
                capture.flush()

        return iter_stream()

    def _append_record(self, record: 'typing.Dict[str, object]') -> 'None':
        serialized = json.dumps(record, ensure_ascii=False)
        with self._lock:
            os.makedirs(self._root_dir, exist_ok=True)
            with open(self._dump_path, "a", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.write("\n")


class _TrajectoryCapture:
    def __init__(
        self,
        writer: 'TrajectoryDumpWriter',
        send_timestamp: 'float',
    ) -> 'None':
        self._writer = writer
        self._send_timestamp = float(send_timestamp)
        self._prefill_token_ids = None
        self._decode_token_ids = []
        self._closed = False

    def observe_chunk(self, payload: 'object') -> 'None':
        if not isinstance(payload, dict):
            return
        if self._prefill_token_ids is None and "prompt_token_ids" in payload:
            normalized_prefill = _normalize_token_ids(payload.get("prompt_token_ids"))
            if normalized_prefill is not None:
                self._prefill_token_ids = normalized_prefill

        choices = payload.get("choices") or []
        if not isinstance(choices, list):
            return
        for raw_choice in choices:
            if not isinstance(raw_choice, dict):
                continue
            normalized_decode = _normalize_token_ids(raw_choice.get("token_ids"))
            if normalized_decode:
                self._decode_token_ids.extend(normalized_decode)

    def flush(self) -> 'None':
        if self._closed:
            return
        self._closed = True
        record = {
            "tokens": {
                "prefill": list(self._prefill_token_ids or []),
                "decode": list(self._decode_token_ids),
            },
            "send_timestamp": self._send_timestamp,
        }
        try:
            self._writer._append_record(record)
        except Exception as exc:
            print(
                "responses_server: failed to append PYCODEX_DUMP trajectory: %s"
                % exc,
                file=sys.stderr,
            )


def _normalize_token_ids(raw_value: 'object') -> 'typing.Union[typing.List[int], None]':
    if not isinstance(raw_value, list):
        return None
    token_ids = []
    for value in raw_value:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        token_ids.append(value)
    return token_ids
