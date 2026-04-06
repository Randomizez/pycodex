
from dataclasses import dataclass
import threading
import time
import typing


@dataclass(frozen=True, )
class StoredResponse:
    response_id: 'str'
    session_id: 'typing.Union[str, None]'
    model: 'str'
    created_at: 'float'


class SessionStore:
    def __init__(self) -> 'None':
        self._lock = threading.Lock()
        self._next_response_number = 1
        self._responses: 'typing.Dict[str, StoredResponse]' = {}

    def create_response(self, session_id: 'typing.Union[str, None]', model: 'str') -> 'StoredResponse':
        with self._lock:
            response_id = f"resp_{self._next_response_number:08d}"
            self._next_response_number += 1
            stored = StoredResponse(
                response_id=response_id,
                session_id=session_id,
                model=model,
                created_at=time.time(),
            )
            self._responses[response_id] = stored
            return stored

    def get_response(self, response_id: 'str') -> 'typing.Union[StoredResponse, None]':
        with self._lock:
            return self._responses.get(response_id)
