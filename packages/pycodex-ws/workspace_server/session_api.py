"""Resource endpoints for custom frontends, scoped to an existing session."""

import base64
import json

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.routing import Route

from pycodex.utils.history import serialize_history_item


def _encode_cursor(value):
    return base64.urlsafe_b64encode(
        json.dumps(value, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")


def _decode_cursor(value):
    try:
        cursor = json.loads(base64.b64decode(value, altchars=b"-_", validate=True))
        if (
            not isinstance(cursor, dict)
            or not isinstance(cursor.get("source"), str)
            or type(cursor.get("before")) is not int
            or cursor["before"] < 0
        ):
            raise ValueError("invalid history cursor")
        return cursor
    except (ValueError, TypeError, UnicodeError) as error:
        raise HTTPException(422, "invalid history cursor") from error


def history_page(session_id, revision, history, before, limit):
    cursor = _decode_cursor(before) if before is not None else None
    source = "{0}:history:{1}".format(session_id, revision)
    if cursor is not None and cursor["source"] != source:
        raise HTTPException(409, "active history changed; restart pagination")
    end = cursor["before"] if cursor is not None else len(history)
    if end > len(history):
        raise HTTPException(409, "history cursor is beyond the current history")
    start = max(0, end - limit)
    next_cursor = None
    if start:
        next_cursor = _encode_cursor({"source": source, "before": start})
    return {
        "source": "active_history",
        "revision": revision,
        "items": [
            {"index": index, "type": "response_item", "payload": serialize_history_item(item)}
            for index, item in enumerate(history[start:end], start)
        ],
        "next_cursor": next_cursor,
    }


def session_api_routes(resolve_manager, read_json, prefix=""):
    async def query(request):
        manager = resolve_manager(request)
        session_id = request.path_params["session_id"]
        resource = request.url.path.rsplit("/", 1)[-1]
        options = dict(request.query_params)
        if "submission_id" in request.path_params:
            resource = "submission"
            options["submission_id"] = request.path_params["submission_id"]
        if resource == "history":
            try:
                limit = int(options.get("limit", "100"))
            except ValueError as error:
                raise HTTPException(422, "limit must be between 1 and 200") from error
            if not 1 <= limit <= 200:
                raise HTTPException(422, "limit must be between 1 and 200")
            options["limit"] = limit
        try:
            link = manager.get(session_id)
            result = await link.api_query(resource, options)
        except KeyError as error:
            raise HTTPException(404, "session, agent or submission not found") from error
        return JSONResponse(
            dict(result, api_version=1, session_id=session_id),
            headers={"Cache-Control": "no-store"},
        )

    async def command(request):
        payload = await read_json(request)
        if (
            set(payload) - {"command", "argument"}
            or not isinstance(payload.get("command"), str)
            or not isinstance(payload.get("argument", ""), str)
        ):
            raise HTTPException(422, "expected command and optional argument strings")
        session_id = request.path_params["session_id"]
        try:
            link = resolve_manager(request).get(session_id)
            result = await link.submit_command(
                payload["command"], payload.get("argument", ""),
                request.query_params.get("agent_id"),
            )
        except KeyError as error:
            raise HTTPException(404, "session or agent not found") from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        return JSONResponse(
            dict(result, api_version=1, session_id=session_id), status_code=202,
            headers={"Cache-Control": "no-store"},
        )

    base = prefix + "/api/sessions/{session_id}"
    return [
        Route(base + "/" + name, query, methods=["GET"])
        for name in ("state", "models", "history", "agents", "background-work")
    ] + [
        Route(base + "/submissions/{submission_id}", query, methods=["GET"]),
        Route(base + "/commands", command, methods=["POST"]),
    ]
