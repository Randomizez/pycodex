
from .config import CompatServerConfig
from .payload_processors import post_process_outcomming_request
from .session_store import SessionStore
from .stream_router import StreamRouter
from .trajectory_dump import TrajectoryDumpWriter
import typing


class ResponseServer:
    def __init__(
        self,
        config: 'CompatServerConfig',
        session_store: 'typing.Union[SessionStore, None]' = None,
        stream_router: 'typing.Union[StreamRouter, None]' = None,
    ) -> 'None':
        self._config = config
        self._session_store = session_store or SessionStore()
        self._stream_router = stream_router or StreamRouter(config)
        self._trajectory_dump = TrajectoryDumpWriter.from_env()

    @property
    def config(self) -> 'CompatServerConfig':
        return self._config

    @property
    def session_store(self) -> 'SessionStore':
        return self._session_store

    @property
    def stream_router(self) -> 'StreamRouter':
        return self._stream_router

    def list_models(self) -> 'typing.Dict[str, object]':
        return self._stream_router.list_models()

    def start_response_stream(
        self,
        request_body: 'typing.Dict[str, object]',
        request_headers: 'typing.Dict[str, str]',
    ):
        outcomming_request = self._stream_router.build_outcomming_request(request_body)
        if self._trajectory_dump is not None:
            # vLLM surfaces prompt/decode token IDs only when this flag is set.
            outcomming_request["return_token_ids"] = True
        outcomming_request = post_process_outcomming_request(
            outcomming_request,
            self._config.model_provider,
        )
        custom_tool_names = self._stream_router.collect_custom_tool_names(request_body)
        session_id = (
            request_headers.get("x-client-request-id")
            or str(request_body.get("prompt_cache_key", "")).strip()
            or None
        )
        stored_response = self._session_store.create_response(
            session_id=session_id,
            model=str(outcomming_request["model"]),
        )
        return self._stream_router.route_stream(
            stored_response,
            outcomming_request,
            custom_tool_names,
            self._trajectory_dump,
        )
