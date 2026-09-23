"""Compare real Codex requests with the production pycodex runtime, offline."""

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

from pycodex.bootstrap import build_agent, build_model, build_runtime
from pycodex.compat import ThreadingHTTPServer
from tests.compare_steer_request_bodies import scripted_origin_server
from tests.fake_responses_server import CaptureStore, build_proxy_handler


def response_events(items, index, model):
    response = {
        "id": "resp_fixture_{0}".format(index),
        "object": "response",
        "model": model,
        "status": "in_progress",
    }
    events = [("response.created", {"response": response})]
    for item in items:
        events.append(("response.output_item.done", {"item": item}))
    events.append(
        (
            "response.completed",
            {
                "response": dict(response, status="completed", output=items),
            },
        )
    )
    return "".join(
        "event: {0}\ndata: {1}\n\n".format(
            name,
            json.dumps(dict(payload, type=name), ensure_ascii=False),
        )
        for name, payload in events
    )


def response_sequence(scenario, model):
    answer = {
        "type": "message",
        "id": "msg_fixture_answer",
        "role": "assistant",
        "phase": "final_answer",
        "content": [{"type": "output_text", "text": "ALIGNED"}],
    }
    if scenario == "tool":
        first = [
            {
                "type": "reasoning",
                "id": "rs_fixture",
                "summary": [{"type": "summary_text", "text": "Use the fixture tool."}],
                "encrypted_content": "fixture-encrypted-reasoning",
            },
            {
                "type": "function_call",
                "id": "fc_fixture",
                "call_id": "call_fixture",
                "name": "exec_command",
                "arguments": '{ "cmd": "printf aligned", "max_output_tokens": 100 }',
            },
        ]
    else:
        first = [answer]
    return (
        response_events(first, 0, model),
        response_events([answer], 1, model),
    )


def write_config(path, model, base_url, personality, effort, sandbox):
    lines = [
        "model = {0}".format(json.dumps(model)),
        'model_provider = "capture"',
        'approval_policy = "never"',
        "sandbox_mode = {0}".format(json.dumps(sandbox)),
        "model_reasoning_effort = {0}".format(json.dumps(effort)),
        'developer_instructions = "Preserve the fixture context."',
    ]
    if personality != "default":
        lines.append("personality = {0}".format(json.dumps(personality)))
    lines.extend(
        [
            "[model_providers.capture]",
            'name = "Offline context capture"',
            "base_url = {0}".format(json.dumps(base_url)),
            'wire_api = "responses"',
            "requires_openai_auth = false",
            "request_max_retries = 0",
            "stream_max_retries = 0",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_requests(root):
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*_POST_*.json"))
        if path.name.endswith("_responses.json")
    ]


def run_command(command, workspace, environment, timeout):
    process = subprocess.Popen(
        command,
        cwd=str(workspace),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        start_new_session=True,
    )
    try:
        output, error = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, error = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, error = process.communicate(timeout=5)
        raise RuntimeError(
            "Codex capture timed out:\n" + output[-2000:] + error[-4000:]
        )
    if process.returncode:
        raise RuntimeError("Codex capture failed: " + error)


def run_codex(workspace, capture_root, environment, timeout, resume):
    run_command(
        ["codex", "exec", "--skip-git-repo-check", "--json", "First fixture prompt."],
        workspace,
        environment,
        timeout,
    )
    if resume:
        session_id = load_requests(capture_root)[0]["body"]["prompt_cache_key"]
        run_command(
            [
                "codex",
                "exec",
                "resume",
                "--skip-git-repo-check",
                "--json",
                session_id,
                "Second fixture prompt.",
            ],
            workspace,
            environment,
            timeout,
        )


async def run_pycodex(config, workspace, resume):
    runtime = build_runtime(
        build_agent(build_model(str(config)), config, cwd=workspace)
    )
    await runtime.start(str(config))
    try:
        receipt = await runtime.submit_input("First fixture prompt.", "cli")
        await receipt.future
    finally:
        await runtime.close()
    if resume:
        path = runtime.agent.session_file_path
        runtime = build_runtime(
            build_agent(build_model(str(config)), config, cwd=workspace)
        )
        runtime.resume(path)
        await runtime.start(str(config))
        try:
            receipt = await runtime.submit_input("Second fixture prompt.", "cli")
            await receipt.future
        finally:
            await runtime.close()


def capture(
    root,
    scenario,
    model,
    timeout,
    personality="pragmatic",
    effort="high",
    sandbox="danger-full-access",
):
    home = root / "home"
    codex_home = home / ".codex"
    workspace = root / "workspace" / "nested"
    codex_home.mkdir(parents=True)
    workspace.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(workspace.parent)], check=True)
    (codex_home / "AGENTS.md").write_text("Global fixture rules.\n", encoding="utf-8")
    (workspace.parent / "AGENTS.md").write_text(
        "Root fixture rules.\n", encoding="utf-8"
    )
    (workspace / "AGENTS.override.md").write_text(
        "Nested fixture rules.\n", encoding="utf-8"
    )
    skill = codex_home / "skills" / "fixture" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: fixture\ndescription: Fixture skill instructions.\n---\nUse fixture data.\n",
        encoding="utf-8",
    )
    environment = dict(os.environ, HOME=str(home), CODEX_HOME=str(codex_home))
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        environment.pop(key, None)
    captured = {}
    for label in ("codex", "pycodex"):
        store = CaptureStore(root / label)
        with scripted_origin_server(
            model_id=model,
            response_bodies=response_sequence(scenario, model),
            first_delay_seconds=0,
        ) as origin:
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                build_proxy_handler(store, origin.base_url, timeout),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            config = codex_home / "config.toml"
            write_config(
                config,
                model,
                "http://127.0.0.1:{0}/v1".format(server.server_port),
                personality,
                effort,
                sandbox,
            )
            try:
                if label == "codex":
                    run_codex(
                        workspace,
                        store.root,
                        environment,
                        timeout,
                        scenario == "resume",
                    )
                else:
                    asyncio.run(run_pycodex(config, workspace, scenario == "resume"))
            finally:
                server.shutdown()
                thread.join()
                server.server_close()
        captured[label] = load_requests(store.root)
    return captured


def normalize_context(body, identifiers):
    result = {"instructions": body.get("instructions"), "input": body["input"]}
    result = json.loads(json.dumps(result))
    for item in result["input"]:
        identifier = item.get("id")
        client_item = item.get("role") in {"user", "developer"} or item["type"] in {
            "function_call_output",
            "custom_tool_call_output",
        }
        if (
            client_item
            and identifier is not None
            and re.fullmatch(r"(msg|at|fco|cto)_[0-9a-f-]{36}", identifier)
        ):
            if identifier not in identifiers:
                identifiers[identifier] = "{0}_client_{1}".format(
                    identifier.split("_", 1)[0], len(identifiers)
                )
            item["id"] = identifiers[identifier]
        if item.get("type") == "function_call_output" and isinstance(
            item.get("output"), str
        ):
            item["output"] = re.sub(
                r"(?m)^(Chunk ID: |Wall time: ).*$",
                r"\1<dynamic>",
                item["output"],
            )
    return result


def shared_context(body):
    result = normalize_context(body, {})
    included = []
    exclusions = []
    for index, item in enumerate(result["input"]):
        location = "$.input[{0}]".format(index)
        if item["type"] == "additional_tools":
            exclusions.append(location + ": tool declarations compared separately")
            continue
        if item.get("role") == "developer":
            content = []
            for part in item["content"]:
                text = part.get("text", "")
                marker = next(
                    (
                        tag
                        for tag in (
                            "collaboration_mode",
                            "multi_agent_role",
                            "multi_agent_mode",
                        )
                        if text.startswith("<" + tag + ">")
                    ),
                    None,
                )
                if marker is not None:
                    exclusions.append(location + ": " + marker)
                else:
                    content.append(part)
            item["content"] = content
            if not content:
                continue
        identifier = item.get("id", "")
        if (
            item.get("role") in {"user", "developer"}
            or item["type"] in {"function_call_output", "custom_tool_call_output"}
        ) and re.fullmatch(r"(msg|fco|cto)_client_\d+", identifier):
            exclusions.append(location + ".id: client-generated identity")
            del item["id"]
        included.append(item)
    result["input"] = included
    return result, exclusions


def tool_declarations(body):
    if "tools" in body:
        return body["tools"]
    return next(
        item["tools"] for item in body["input"] if item["type"] == "additional_tools"
    )


def difference_paths(left, right, path="$"):
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                result.append(path + "." + key)
            else:
                result.extend(difference_paths(left[key], right[key], path + "." + key))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [path + ".length"]
        return [
            difference
            for index, (left_item, right_item) in enumerate(zip(left, right))
            for difference in difference_paths(
                left_item, right_item, "{0}[{1}]".format(path, index)
            )
        ]
    return [] if left == right else [path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument(
        "--personality",
        choices=("default", "none", "pragmatic", "friendly"),
        default="pragmatic",
    )
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--sandbox",
        choices=("danger-full-access", "read-only", "workspace-write"),
        default="danger-full-access",
    )
    parser.add_argument(
        "--scenario", choices=("plain", "resume", "tool"), action="append"
    )
    parser.add_argument("--timeout-seconds", type=float, default=30)
    args = parser.parse_args()
    root = args.root or Path(tempfile.mkdtemp(prefix="pycodex-context-compare-"))
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "codex_version": subprocess.check_output(
            ["codex", "--version"],
            universal_newlines=True,
        ).strip(),
        "model": args.model,
        "personality": args.personality,
        "reasoning_effort": args.reasoning_effort,
        "sandbox_mode": args.sandbox,
        "scenarios": {},
    }
    for scenario in args.scenario or ("plain", "resume", "tool"):
        captured = capture(
            root / scenario,
            scenario,
            args.model,
            args.timeout_seconds,
            args.personality,
            args.reasoning_effort,
            args.sandbox,
        )
        identifiers = {"codex": {}, "pycodex": {}}
        contexts = {
            label: [
                normalize_context(request["body"], identifiers[label])
                for request in requests
            ]
            for label, requests in captured.items()
        }
        differences = difference_paths(contexts["codex"], contexts["pycodex"])
        shared = {
            label: [shared_context(request["body"]) for request in requests]
            for label, requests in captured.items()
        }
        shared_differences = difference_paths(
            [context for context, exclusions in shared["codex"]],
            [context for context, exclusions in shared["pycodex"]],
        )
        report["scenarios"][scenario] = {
            "request_counts": {
                label: len(requests) for label, requests in captured.items()
            },
            "context_differences": differences,
            "shared_context_differences": shared_differences,
            "shared_context_exclusions": {
                label: [exclusions for context, exclusions in requests]
                for label, requests in shared.items()
            },
            "tool_declaration_differences": [
                difference_paths(
                    tool_declarations(left["body"]), tool_declarations(right["body"])
                )
                for left, right in zip(captured["codex"], captured["pycodex"])
            ],
            "header_differences": [
                difference_paths(left["headers"], right["headers"])
                for left, right in zip(captured["codex"], captured["pycodex"])
            ],
            "non_context_differences": [
                difference_paths(
                    {
                        key: value
                        for key, value in left["body"].items()
                        if key not in ("input", "instructions", "prompt_cache_key")
                    },
                    {
                        key: value
                        for key, value in right["body"].items()
                        if key not in ("input", "instructions", "prompt_cache_key")
                    },
                )
                for left, right in zip(captured["codex"], captured["pycodex"])
            ],
        }
        print("{0}: {1}".format(scenario, differences or "context equal"))
        print(
            "  shared context:", shared_differences or "equal (see explicit exclusions)"
        )
    path = root / "comparison.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Report:", path)
    return int(
        any(item["context_differences"] for item in report["scenarios"].values())
    )


if __name__ == "__main__":
    raise SystemExit(main())
