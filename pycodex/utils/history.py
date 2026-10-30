"""Pure conversation-history projections, independent of rollout storage."""

import typing

from ..protocol import AgentMessage, AssistantMessage, ConversationItem, UserMessage


def conversation_history_to_turns(
    history: "typing.Iterable[ConversationItem]",
) -> "typing.Tuple[typing.Tuple[str, str], ...]":
    turns: "typing.List[typing.Tuple[str, str]]" = []
    current_user_text: "typing.Union[str, None]" = None
    current_assistant_text = ""
    for item in history:
        if isinstance(item, (UserMessage, AgentMessage)):
            if current_user_text is not None or current_assistant_text:
                turns.append((current_user_text or "", current_assistant_text))
            current_user_text = item.text
            current_assistant_text = ""
            continue
        if isinstance(item, AssistantMessage):
            current_assistant_text = item.text
    if current_user_text is not None or current_assistant_text:
        turns.append((current_user_text or "", current_assistant_text))
    return tuple(turns)
