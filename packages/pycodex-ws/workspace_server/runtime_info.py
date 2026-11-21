"""Runtime reference text for the user to review and send."""

import json
from pathlib import Path
from urllib.parse import quote, urlsplit

import pycodex
from pycodex_router.protocol import normalize_api_url


def _curl_header_command(header_path):
    text = str(header_path)
    is_windows_path = (len(text) >= 2 and text[1] == ":") or "\\" in text
    return '{0} -H "@{1}"'.format("curl.exe" if is_windows_path else "curl", text)


def source_directories():
    return "pycodex={0}; workspace_server={1}".format(
        Path(pycodex.__file__).resolve().parent,
        Path(__file__).resolve().parent,
    )


def runtime_info_prompt(
    session_id, workspace_id, local_url, agent_header_path=None,
):
    path = "/w/" + quote(workspace_id, safe="") if workspace_id else ""
    help_target = "{0}{1}/api/help?session_id={2}".format(
        normalize_api_url(local_url) if local_url is not None else "",
        path,
        quote(session_id, safe=""),
    )
    access = (
        '{0} "{1}"'.format(_curl_header_command(agent_header_path), help_target)
        if agent_header_path is not None
        else "GET {0}".format(help_target)
    )
    return "Collaboration info: {0}\nSource: {1}".format(access, source_directories())


def agent_help_text(
    multi, local_url, router, agent_header_path=None,
    workspace_id=None, session_id=None, summary=None,
):
    lines = [
        "Runtime: pycodex / pycodex-ws (reference only).",
        "Source: {0}".format(source_directories()),
    ]
    if session_id is not None and summary is not None:
        lines.append(
            "Workspace: {0}; session: {1}; API ID: {2}\ncwd: {3}".format(
                workspace_id or "(single)",
                summary["title"] or "(untitled)",
                session_id,
                summary["cwd"],
            )
        )
    if local_url is not None:
        path = "/w/" + quote(workspace_id, safe="") if workspace_id else ""
        if workspace_id or not multi:
            lines.append("Local L={0}; current workspace=L{1}".format(normalize_api_url(local_url), path))
        else:
            lines.append("Local L={0}".format(normalize_api_url(local_url)))
        if multi:
            lines.append("GET L/api/workspaces -> workspaces[].id/sessions; B=L/w/<id> (case-sensitive).")
    if agent_header_path is not None:
        lines.append(
            "Access: {0} URL ... (header file: X-Pycodex-Agent-Token: <token>)".format(
                _curl_header_command(agent_header_path)
            )
        )
    if router is not None and router.connected:
        if router.api_url is None:
            lines.append("Router: connected; API URL not supplied (availability unknown).")
        else:
            url = normalize_api_url(router.api_url)
            lines.append("Router: R={0}; device=/n/{1}".format(url, router.node_id))
            if agent_header_path is not None:
                lines.append(
                    "G=L/router; GET G{0}/api/workspaces -> workspaces[].url; B=G+url (same header).".format(
                        urlsplit(url).path
                    )
                )
            else:
                lines.append("GET R/api/workspaces -> workspaces[].url; resolve against R to get B.")
    sender = session_id or "<sender_id>"
    lines += [
        "B=target workspace URL (no trailing /); use its session_id.",
        "GET B/api/sessions -> sessions[].id/title",
        "POST B/api/session/message "
        + json.dumps({"session_id": "<target>", "prompt": "From " + sender + ": <task; reply address>"})
        + " -> submission_id (accepted)",
        "Use plain messages (steer) by default. Add /queue only for an independent follow-up task "
        "that should wait for current work to finish.",
        "GET B/api/session?session_id=<target> -> snapshot.turns; match submission_id for status/response/error.",
    ]
    return "\n".join(lines) + "\n"
