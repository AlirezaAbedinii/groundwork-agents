"""One text rendering of a LangChain message list.

``render_messages`` is the single text form of a conversation, used wherever a
call needs a stable string: mock-fixture matching and exact-fixture keys,
replay keys, and the ``llm_calls.prompt`` column the trace explorer shows.
It is deterministic (one block per message, in order, JSON keys sorted) so the
same conversation always renders to the same text.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def render_messages(messages: list[BaseMessage]) -> str:
    """Render *messages* as text, one block per message, blocks separated by a blank line.

    Line kinds::

        [system] <content>
        [user] <content>
        [assistant] <text>                          (line omitted when the text is empty)
        -> called <name>(<json args, sorted keys>)  (one per tool call on that AIMessage)
        -> <name> returned: <content>               (ToolMessage, status success)
        -> <name> failed: <content>                 (ToolMessage, status error)
    """
    blocks: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage):
            lines = [f"[assistant] {message.text}"] if message.text else []
            for call in message.tool_calls:
                lines.append(f"-> called {call['name']}({json.dumps(call['args'], sort_keys=True)})")
            if lines:
                blocks.append("\n".join(lines))
        elif isinstance(message, ToolMessage):
            verb = "failed" if message.status == "error" else "returned"
            blocks.append(f"-> {message.name or 'tool'} {verb}: {message.text}")
        elif message.type == "system":
            blocks.append(f"[system] {message.text}")
        elif message.type == "human":
            blocks.append(f"[user] {message.text}")
        else:
            blocks.append(f"[{message.type}] {message.text}")
    return "\n\n".join(blocks)
