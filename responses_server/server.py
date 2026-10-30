import typing
from dataclasses import dataclass

from .config import CompatServerConfig
from .payload_processors import post_process_outcomming_request
from .request_translation import ToolNameMap, build_outcomming_request
from .session_store import SessionStore
from .stream_router import StreamRouter
from .trajectory_dump import TrajectoryDumpWriter
from .transport import OutcommingTransport


@dataclass(frozen=True)
class PreparedResponse:
    request_body: "typing.Dict[str, object]"
    outcomming_request: "typing.Dict[str, object]"
    tool_names: "ToolNameMap"


class ResponseServer:
    def __init__(
        self,
        config: "CompatServerConfig",
        session_store: "typing.Union[SessionStore, None]" = None,
        stream_router: "typing.Union[StreamRouter, None]" = None,
    ) -> "None":
        self.config = config
        self.session_store = session_store or SessionStore()
        self.stream_router = stream_router or StreamRouter(OutcommingTransport(config))
        self._trajectory_dump = TrajectoryDumpWriter.from_env()

    def list_models(self) -> "typing.Dict[str, object]":
        return self.stream_router.transport.list_models()

    def prepare_response(
        self, request_body: "typing.Dict[str, object]",
    ) -> "PreparedResponse":
        tool_names = ToolNameMap()
        return PreparedResponse(
            request_body,
            build_outcomming_request(request_body, tool_names),
            tool_names,
        )

    def start_response_stream(
        self,
        prepared: "PreparedResponse",
        request_headers: "typing.Dict[str, str]",
    ):
        outcomming_request = prepared.outcomming_request
        if self._trajectory_dump is not None:
            # vLLM surfaces prompt/decode token IDs only when this flag is set.
            outcomming_request["return_token_ids"] = True
        outcomming_request = post_process_outcomming_request(
            outcomming_request,
            self.config.model_provider,
        )
        session_id = (
            request_headers.get("x-client-request-id")
            or str(prepared.request_body.get("prompt_cache_key", "")).strip()
            or None
        )
        stored_response = self.session_store.create_response(
            session_id=session_id,
            model=str(outcomming_request["model"]),
        )
        return self.stream_router.route_stream(
            stored_response,
            outcomming_request,
            prepared.tool_names,
            self._trajectory_dump,
        )
