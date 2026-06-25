import asyncio
import json
from dataclasses import asdict

from .protocol import AgentEvent
from .runtime import CliSubmissionQueue
from .runtime_services import create_agent_runtime_environment
from .utils import CliSessionView, uuid7_string
from .utils.compactor import compact_agent
from .utils.session_persist import (
    SessionRolloutRecorder,
    conversation_history_to_turns,
    list_resumable_sessions,
    load_resumed_session,
    resolve_codex_home,
)
import typing


EXIT_COMMANDS = {"/exit", "/quit"}
HISTORY_COMMAND = "/history"
TITLE_COMMAND = "/title"
MODEL_COMMAND = "/model"
QUEUE_COMMAND = "/queue"
RESUME_COMMAND = "/resume"
COMPACT_COMMAND = "/compact"
LINK_COMMAND = "/link"
UNLINK_COMMAND = "/unlink"
EXTRA_COMMANDS_LINE = (
    "Extra commands: /history, /title, /model, /resume, /compact, /link, /unlink"
)


def format_turn_output(result, json_mode: "bool") -> "str":
    if json_mode:
        return json.dumps(asdict(result), ensure_ascii=False, indent=2)
    return result.output_text or ""


async def prompt_request_user_input(
    view,
    payload: "typing.Dict[str, object]",
) -> "typing.Union[typing.Dict[str, object], None]":
    view.finish_stream()
    view.write_line("[request_user_input] waiting for user response")
    answers: "typing.Dict[str, typing.Dict[str, typing.List[str]]]" = {}
    for question in payload.get("questions", []):
        if not isinstance(question, dict):
            continue
        header = str(question.get("header", "")).strip()
        question_text = str(question.get("question", "")).strip()
        question_id = str(question.get("id", "")).strip()
        if header:
            view.write_line(f"[{header}] {question_text}")
        else:
            view.write_line(question_text)

        options = question.get("options") or []
        if isinstance(options, list):
            for index, option in enumerate(options, start=1):
                if not isinstance(option, dict):
                    continue
                label = str(option.get("label", "")).strip()
                description = str(option.get("description", "")).strip()
                view.write_line(f"  {index}. {label} - {description}")
        view.write_line("  0. Other")

        try:
            raw_answer = await view.get_prompt("answer> ")
        except EOFError:
            return None
        answer_text = raw_answer.strip()
        if not answer_text:
            return None

        selected_answer = answer_text
        if answer_text.isdigit() and isinstance(options, list):
            choice = int(answer_text)
            if 1 <= choice <= len(options):
                option = options[choice - 1]
                if isinstance(option, dict):
                    selected_answer = (
                        str(option.get("label", "")).strip() or answer_text
                    )
            elif choice == 0:
                try:
                    raw_answer = await view.get_prompt("other> ")
                except EOFError:
                    return None
                selected_answer = raw_answer.strip()
                if not selected_answer:
                    return None

        answers[question_id] = {"answers": [selected_answer]}

    return {"answers": answers}


async def prompt_request_permissions(
    view,
    payload: "typing.Dict[str, object]",
) -> "typing.Union[typing.Dict[str, object], None]":
    view.finish_stream()
    view.write_line("[request_permissions] user approval required")
    reason = payload.get("reason")
    if reason:
        view.write_line(f"Reason: {reason}")
    view.write_line("Requested permissions:")
    view.write_line(
        json.dumps(payload.get("permissions", {}), ensure_ascii=False, indent=2)
    )
    view.write_line("Choose: [n] deny / [t] grant for turn / [s] grant for session")
    try:
        raw_answer = await view.get_prompt("permissions> ")
    except EOFError:
        return None

    answer = raw_answer.strip().lower()
    if answer in {"t", "turn", "y", "yes"}:
        return {
            "permissions": payload.get("permissions", {}),
            "scope": "turn",
        }
    if answer in {"s", "session"}:
        return {
            "permissions": payload.get("permissions", {}),
            "scope": "session",
        }
    return {
        "permissions": {},
        "scope": "turn",
    }


async def run_interactive_session(
    queue: "CliSubmissionQueue",
    json_mode: "bool",
    config_path: "typing.Union[str, None]" = None,
    view=None,
    view_factory=None,
    show_banner: "bool" = True,
) -> "int":
    worker = asyncio.create_task(queue.run_forever())
    context_window_tokens = queue._agent._context_manager.resolve_model_context_window()
    if view is None:
        factory = view_factory or CliSessionView
        view = factory()
    view.set_context_window_tokens(context_window_tokens)
    model_client = queue._agent._model_client
    codex_home = resolve_codex_home(config_path)
    queue.set_event_handler(view.handle_event)
    pending_turn_tasks: "typing.Set[asyncio.Task[None]]" = set()
    runtime_environment = queue._agent.runtime_environment
    if runtime_environment is None:
        runtime_environment = create_agent_runtime_environment()
        queue._agent.runtime_environment = runtime_environment
    runtime_environment.request_user_input_manager.set_handler(
        lambda payload: prompt_request_user_input(view, payload)
    )
    runtime_environment.request_permissions_manager.set_handler(
        lambda payload: prompt_request_permissions(view, payload)
    )
    if show_banner:
        view.write_line("pycodex interactive mode. Type /exit to quit.")
        view.write_line(EXTRA_COMMANDS_LINE)
    feishu_link = None
    try:

        def has_pending_turn_tasks() -> "bool":
            pending_turn_tasks.difference_update(
                task for task in tuple(pending_turn_tasks) if task.done()
            )
            return bool(pending_turn_tasks)

        async def run_manual_compact() -> "None":
            current_agent = queue._agent
            if not current_agent.history:
                view.write_line("Nothing to compact.")
                return

            compact_turn_id = uuid7_string()

            def handle_compact_stream_event(event) -> "None":
                if event.kind not in {"token_count", "stream_error"}:
                    return
                view.handle_event(
                    AgentEvent(
                        kind=event.kind,
                        turn_id=compact_turn_id,
                        payload=dict(event.payload),
                    )
                )

            view.write_line("Compacting conversation history...")
            compact_result = await compact_agent(
                current_agent,
                handle_compact_stream_event,
                True,
            )
            if compact_result is None:
                view.write_line("Nothing to compact.")
                return
            view.load_session_history(
                getattr(view, "_title", None),
                conversation_history_to_turns(compact_result.history),
            )
            view.write_line(compact_result.display_text())

        async def wait_for_turn_result(future) -> "None":
            try:
                result = await future
            except Exception as exc:  # pragma: no cover - defensive surface
                if str(exc) == "submission interrupted":
                    return
                view.show_error(str(exc))
                return

            if json_mode:
                view.write_line(format_turn_output(result, True))

        while True:
            try:
                raw_line = await view.poll_prompt()
            except EOFError:
                break
            if raw_line is None:
                await asyncio.sleep(0.05)
                continue

            prompt_text = raw_line.strip()
            if not prompt_text:
                continue
            if prompt_text in EXIT_COMMANDS:
                break
            if prompt_text == HISTORY_COMMAND:
                view.show_history()
                continue
            if prompt_text == TITLE_COMMAND:
                view.show_title()
                continue
            if prompt_text.startswith(f"{TITLE_COMMAND} "):
                title = prompt_text[len(TITLE_COMMAND) :].strip()
                if not title:
                    view.write_line("Usage: /title <title>")
                    continue
                set_session_title = getattr(view, "set_session_title", None)
                if callable(set_session_title):
                    set_session_title(title)
                else:
                    view.write_line(f"Session: {title}")
                continue
            if prompt_text == RESUME_COMMAND:
                sessions = list_resumable_sessions(codex_home)
                if not sessions:
                    view.write_line("No resumable sessions found.")
                    continue
                view.write_line("Available sessions:")
                for index, session in enumerate(sessions, start=1):
                    view.write_line(f"[{index}] {session['preview']}")
                continue
            if prompt_text.startswith(f"{RESUME_COMMAND} "):
                if has_pending_turn_tasks():
                    view.write_line(
                        "Cannot resume while work is running or queued."
                    )
                    continue
                resume_target = prompt_text[len(RESUME_COMMAND) :].strip()
                try:
                    resumed = load_resumed_session(codex_home, resume_target)
                    queue._agent.replace_history(resumed["history"])
                    if hasattr(model_client, "_session_id"):
                        model_client._session_id = str(resumed["session_id"])
                    queue._agent.set_rollout_recorder(
                        SessionRolloutRecorder.resume(resumed["rollout_path"])
                    )
                    view.load_session_history(
                        str(resumed["title"]),
                        tuple(resumed["turns"]),
                    )
                    show_resumed_session = getattr(view, "show_resumed_session", None)
                    if callable(show_resumed_session):
                        show_resumed_session(str(resumed["title"]))
                    else:
                        view.write_line(f"Resumed session: {resumed['title']}")
                        view.show_history()
                except Exception as exc:  # pragma: no cover - defensive surface
                    view.show_error(str(exc))
                continue
            if prompt_text == COMPACT_COMMAND:
                if has_pending_turn_tasks():
                    view.write_line(
                        "Cannot compact while work is running or queued."
                    )
                    continue
                try:
                    await run_manual_compact()
                except Exception as exc:  # pragma: no cover - defensive surface
                    view.show_error(str(exc))
                continue
            if prompt_text.startswith(f"{LINK_COMMAND} "):
                link_target = prompt_text[len(LINK_COMMAND) :].strip()
                if not link_target:
                    view.write_line("Usage: /link <feishu-email|open_id|chat_id>")
                    continue
                if feishu_link:
                    view.write_line("A Feishu card is already linked. Use /unlink first.")
                    continue
                try:
                    from .feishu_link import PycodexRuntimeLink

                    view.write_line(f"Linking Feishu card to current session: {link_target}")
                    link = await PycodexRuntimeLink(
                        queue,
                        link_target,
                    ).start_async()
                    feishu_link = link
                    view.write_line(
                        "Linked Feishu card: session_key={0} message_id={1}".format(
                            link.session_key,
                            link.message_id or "-",
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive surface
                    view.show_error(str(exc))
                continue
            if prompt_text == UNLINK_COMMAND:
                if not feishu_link:
                    view.write_line("No Feishu card is linked.")
                    continue
                feishu_link.detach()
                feishu_link = None
                view.write_line("Unlinked Feishu card.")
                continue
            if prompt_text.startswith(f"{QUEUE_COMMAND} "):
                queued_text = prompt_text[len(QUEUE_COMMAND) :].strip()
                if not queued_text:
                    view.write_line("Usage: /queue <message>")
                    continue
                try:
                    submission_id, future = await queue.enqueue_user_turn(
                        queued_text, queue="enqueue"
                    )
                    view.show_steer_queued(submission_id, queued_text)
                    turn_task = asyncio.create_task(wait_for_turn_result(future))
                    pending_turn_tasks.add(turn_task)
                except Exception as exc:  # pragma: no cover - defensive surface
                    view.show_error(str(exc))
                continue
            if prompt_text == MODEL_COMMAND:
                view.write_line(
                    f"Current model: {getattr(model_client, 'model', None) or 'unavailable'}"
                )
                models = await model_client.list_models()
                view.write_line(f"Available models: {', '.join(models)}")
                continue
            if prompt_text.startswith(f"{MODEL_COMMAND} "):
                if has_pending_turn_tasks():
                    view.write_line(
                        "Cannot change model while work is running or queued in steer mode."
                    )
                    continue
                model_name = prompt_text[len(MODEL_COMMAND) :].strip()
                if not model_name:
                    view.write_line("Usage: /model <model>")
                    continue

                model_client.model = model_name
                view.write_line(f"Switched model to {model_name}.")
                continue

            try:
                steered = has_pending_turn_tasks()
                submission_id, future = await queue.enqueue_user_turn(
                    prompt_text,
                    queue="steer",
                )
                if steered:
                    view.schedule_steer_inserted(submission_id, prompt_text)
                turn_task = asyncio.create_task(wait_for_turn_result(future))
                pending_turn_tasks.add(turn_task)
                continue
            except Exception as exc:  # pragma: no cover - defensive surface
                view.show_error(str(exc))
                continue
    finally:
        if feishu_link:
            feishu_link.detach()
            feishu_link.stop()
        runtime_environment.request_user_input_manager.set_handler(None)
        runtime_environment.request_permissions_manager.set_handler(None)
        await queue.shutdown()
        await worker
        if pending_turn_tasks:
            await asyncio.gather(*pending_turn_tasks, return_exceptions=True)
        view.close()

    return 0
