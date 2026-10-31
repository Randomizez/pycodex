"""Runtime reference text for the user to review and send."""

import json
import shlex
from urllib.parse import quote, urlsplit

from pycodex_router.protocol import normalize_api_url


def runtime_info_prompt(
    session_id, summary, workspace_id, local_url, router,
    agent_socket_path=None,
):
    lines = [
        "Runtime: pycodex / pycodex-ws (reference only).",
        "Workspace: {0}; session: {1}; API ID: {2}\ncwd: {3}".format(
            workspace_id or "(single)", summary["title"] or "(untitled)",
            session_id, summary["cwd"],
        ),
    ]
    if local_url is not None:
        path = "/w/" + quote(workspace_id, safe="") if workspace_id else ""
        lines.append("Local L={0}; current workspace=L{1}".format(normalize_api_url(local_url), path))
        if workspace_id:
            lines.append("GET L/api/workspaces -> workspaces[].id/sessions; B=L/w/<id> (case-sensitive).")
    if agent_socket_path is not None:
        lines.append("Access: curl --unix-socket {0} URL ...".format(shlex.quote(str(agent_socket_path))))
    if router is not None and router.connected:
        if router.api_url is None:
            lines.append("Router: connected; API URL not supplied (availability unknown).")
        else:
            url = normalize_api_url(router.api_url)
            lines.append("Router: R={0}; device=/n/{1}".format(url, router.node_id))
            if agent_socket_path is not None:
                lines.append("G=L/router; GET G{0}/api/workspaces -> workspaces[].url; B=G+url (same socket).".format(urlsplit(url).path))
            else:
                lines.append("GET R/api/workspaces -> workspaces[].url; resolve against R to get B.")
    lines += [
        "B=target workspace URL (no trailing /); use its session_id.",
        "GET B/api/sessions -> sessions[].id/title",
        "POST B/api/session/message "
        + json.dumps({"session_id": "<target>", "prompt": "From " + session_id + ": <task; reply address>"})
        + " -> submission_id (accepted)",
        "Use plain messages (steer) by default. Add /queue only for an independent follow-up task "
        "that should wait for current work to finish.",
        "GET B/api/session?session_id=<target> -> snapshot.turns; match submission_id for status/response/error.",
    ]
    return "\n".join(lines)
