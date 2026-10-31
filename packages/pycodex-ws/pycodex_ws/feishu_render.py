"""Pure Feishu card layout and content rendering."""

import typing

CARD_OUTPUT_LIMIT = 6500
CARD_OUTPUT_MODE_MARKDOWN = "markdown"
CARD_OUTPUT_MODE_CODE = "code"


def render_card(
    state, output_mode: str = CARD_OUTPUT_MODE_MARKDOWN
) -> "typing.Dict[str, object]":
    status = "Closed" if state.display.closed else state.status
    input_disabled = not state.accepts_input or state.detached or state.display.closed
    output = _truncate(state.output_text, CARD_OUTPUT_LIMIT) or "Ready."
    working_output = _truncate(state.working_output, CARD_OUTPUT_LIMIT)
    color = _status_color("Detached" if state.detached else status or "Idle")
    body_elements = [
        _output_box(
            "answer_box",
            "answer_md",
            _render_output_content(output, output_mode),
            "grey-50",
        ),
    ]
    if working_output:
        body_elements.append(
            _output_box(
                "working_output_box",
                "working_output_md",
                _render_output_content(working_output, output_mode),
                "green-50",
            )
        )
    if state.activity_text:
        body_elements.append(
            {
                "tag": "markdown",
                "element_id": "activity_md",
                "content": _render_output_content(
                    _truncate(state.activity_text, CARD_OUTPUT_LIMIT), output_mode
                ),
            }
        )
    if state.user_prompt:
        body_elements.insert(
            0,
            {
                "tag": "markdown",
                "element_id": "user_prompt_md",
                "content": _render_output_content(
                    "user> " + _truncate(state.user_prompt, CARD_OUTPUT_LIMIT),
                    output_mode,
                ),
            },
        )
    history = state.earlier_turns()
    if history:
        index = (
            len(history) - 1 if state.history_index is None else state.history_index
        )
        body_elements.insert(
            0,
            _history_panel(
                history,
                index,
                state.history_index is not None,
                output_mode,
                not state.detached,
            ),
        )
    card = {
        "schema": "2.0",
        "config": {"update_multi": True},
        "header": {
            "title": {
                "tag": "plain_text",
                "content": state.display.title or "pycodex",
            },
            "template": color,
        },
        "body": {
            "elements": body_elements,
        },
    }
    if not state.detached:
        card["body"]["elements"].append(
            {
                "tag": "input",
                "element_id": "prompt_input",
                "name": "prompt",
                "input_type": "text",
                "width": "fill",
                "disabled": input_disabled,
                "placeholder": {
                    "tag": "plain_text",
                    "content": state.prompt_text + (status or state.model_name),
                },
                "behaviors": [{"type": "callback", "value": {"action": "send"}}],
                "value": {"action": "send"},
            },
        )
    return card


def _status_color(status: str) -> str:
    normalized = status.lower()
    if normalized in {"detached", "closed"}:
        return "grey"
    if normalized in {"idle", "idle: sleeping"}:
        return "green"
    return "blue"


def _truncate(text: str, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 32)] + "\n...[truncated]"


def _render_output_content(output: str, output_mode: str) -> str:
    if output_mode == CARD_OUTPUT_MODE_CODE:
        return "```text\n{0}\n```".format(str(output or "").replace("```", "'''"))
    return output


def _history_panel(history, index, expanded, output_mode, interactive):
    prompt, response = history[index]
    content = _truncate(
        "user> {0}\n\nassistant> {1}".format(prompt, response), CARD_OUTPUT_LIMIT
    )
    columns = []
    for label, target, disabled in (
        ("Previous", index - 1, index == 0),
        ("Next", index + 1, index == len(history) - 1),
    ):
        columns.append(
            {
                "tag": "column",
                "width": "auto",
                "elements": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": label},
                        "type": "default",
                        "disabled": disabled or not interactive,
                        "behaviors": [
                            {
                                "type": "callback",
                                "value": {
                                    "action": "history",
                                    "history_index": target,
                                },
                            }
                        ],
                    }
                ],
            }
        )
    return {
        "tag": "collapsible_panel",
        "element_id": "history_panel",
        "expanded": expanded,
        "header": {
            "icon": {
                "tag": "standard_icon",
                "token": "down-small-ccm_outlined",
                "size": "16px 16px",
            },
            "title": {
                "tag": "plain_text",
                "content": "History · {0} / {1}".format(index + 1, len(history)),
            },
        },
        "elements": [
            {
                "tag": "markdown",
                "element_id": "history_md",
                "content": _render_output_content(content, output_mode),
            },
            {"tag": "column_set", "columns": columns},
        ],
    }


def _output_box(
    box_id: str,
    markdown_id: str,
    content: str,
    background_style: str,
) -> "typing.Dict[str, object]":
    return {
        "tag": "column_set",
        "element_id": box_id,
        "background_style": background_style,
        "horizontal_spacing": "8px",
        "horizontal_align": "left",
        "columns": [
            {
                "tag": "column",
                "width": "auto",
                "elements": [
                    {
                        "tag": "markdown",
                        "element_id": markdown_id,
                        "content": content,
                    },
                ],
                "vertical_spacing": "8px",
                "horizontal_align": "left",
                "vertical_align": "top",
            }
        ],
    }
