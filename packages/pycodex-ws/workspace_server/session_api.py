"""Resource endpoints for custom frontends, scoped to an existing session."""

import base64
import hashlib
import json

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.routing import Route

from pycodex.utils.history import serialize_history_item
from pycodex.utils.session_persist import read_rollout_history_page


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


def _history_digest(items):
    return hashlib.sha256(
        json.dumps(items, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def history_page(session_id, path, history, before, limit):
    cursor = _decode_cursor(before) if before is not None else None
    if path is not None:
        stat = path.stat()
        source = "{0}:rollout:{1}:{2}".format(session_id, stat.st_dev, stat.st_ino)
    else:
        source = session_id + ":memory"
    if cursor is not None and cursor["source"] != source:
        raise HTTPException(409, "history source changed; restart pagination")
    offset = cursor["before"] if cursor is not None else None
    if path is not None:
        try:
            page = read_rollout_history_page(path, offset, limit)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        next_before = page.pop("next_before")
    else:
        items = [serialize_history_item(item) for item in history]
        end = len(items) if offset is None else offset
        if cursor is not None and (
            end > len(items) or cursor.get("prefix") != _history_digest(items[:end])
        ):
            raise HTTPException(409, "active history changed; restart pagination")
        start = max(0, end - limit)
        page = {"items": [
            {"index": index, "type": "response_item", "payload": items[index]}
            for index in range(start, end)
        ]}
        next_before = start if start else None
    next_cursor = None
    if next_before is not None:
        next_value = {"source": source, "before": next_before}
        if path is None:
            next_value["prefix"] = _history_digest(items[:next_before])
        next_cursor = _encode_cursor(next_value)
    return dict(
        page, next_cursor=next_cursor,
        source="rollout" if path is not None else "active_history",
    )


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
