# Context

这份文档记录当前和上游 Codex 对齐时，模型请求 context 的形状。

这不是抓包原文，也不是实现笔记；它更接近一份“对齐契约”：

- `Codex ...`：写上游 Codex 的基准形状。
- `Our ... Diff`：只写 `pycodex` 相对这个基准的差异。
- 已经通过 fake server 抓到的部分，标成 `status = "captured"`。
- 还没抓到、只是根据当前代码推导出的部分，标成 `status = "derived"`。

写法约定：

- 统一用 Python `dict` / `list` / `f-string` 风格表示。
- 不再使用 `capture(name)` 这类命名占位。
- 动态值统一写成语义占位，例如：
  - `"{uuid7() for each session}"`
  - `"{uuid7() for each turn(user input)}"`
  - `"{get_cwd()}"`
  - `"{date(YYYY-MM-DD)}"`
  - `"{IANA timezone, *like Asia/Hong_Kong}"`

```python
InstalledSkills = [
    *{本地用户安装的其他 skills},
    (
        "openai-docs",
        "Use when the user asks how to build with OpenAI products or APIs and needs up-to-date official documentation with citations, help choosing the latest model for a use case, or explicit GPT-5.4 upgrade and prompt-upgrade guidance; prioritize OpenAI docs MCP tools, use bundled references only as helper context, and restrict any fallback browsing to official OpenAI domains.",
        "~/.codex/skills/.system/openai-docs/SKILL.md",
    ),
    (
        "skill-creator",
        "Guide for creating effective skills. This skill should be used when users want to create a new skill (or update an existing skill) that extends Codex's capabilities with specialized knowledge, workflows, or tool integrations.",
        "~/.codex/skills/.system/skill-creator/SKILL.md",
    ),
    (
        "skill-installer",
        "Install Codex skills into $CODEX_HOME/skills from a curated list or a GitHub repo path. Use when a user asks to list installable skills, install a curated skill, or install a skill from another repo (including private repos).",
        "~/.codex/skills/.system/skill-installer/SKILL.md",
    ),
]
```

## 0. `instructions` 和 `input` 是两回事

```python
ResponsesRequest = {
    "instructions": f"""{base_instructions}""",
    "input": [
        *conversation_items,
    ],
}
```

这里必须明确：

- `instructions` 是 Responses API 顶层字段。
- `instructions` 不是 `message`，也不在 `input` 数组里。
- `instructions` 更接近 base instructions，也就是基础系统提示。
- `input` 才是本轮真正发给模型的结构化上下文数组。
- `input` 里面会出现：
  - `developer` / `user` / `assistant` message
  - `function_call` / `function_call_output`
  - `custom_tool_call` / `custom_tool_call_output`

```python
PromptLikeShape = {
    "instructions": f"""{base_prompt_only}""",
    "input": [
        developer_message,
        contextual_user_message,
        latest_user_turn,
        *optional_prior_assistant_items,
        *optional_tool_call_items,
        *optional_tool_result_items,
    ],
}
```

## 1. 首轮请求

```python
FirstTurn = {
    "status": "captured",
    "path": "default CLI non-exec path with initial prompt",
}
```

### 1.1 Codex 基准

```python
CodexFirstTurnRequest = {
    "headers": {
        "originator": "codex-tui",
        "user-agent": "{*like codex-tui/0.115.0 (Ubuntu 22.4.0; x86_64) xterm-256color (codex-tui; 0.115.0)}",
        "x-client-request-id": "{uuid7() for each session}",
        "session_id": "{same as x-client-request-id}",
        "x-codex-turn-metadata": json.dumps(
            {
                "turn_id": "{uuid7() for each turn(user input)}",
                "sandbox": "none",
            },
            separators=(",", ":"),
        ),
        "x-codex-beta-features": "guardian_approval",
    },
    "body": {
        "model": "{configured model slug, *like gpt-5.4}",
        "instructions": f"""{(
            # source:
            # https://github.com/openai/codex/blob/main/codex-rs/core/models.json
            get_from("codex-rs/core/models.json[config.model].model_messages.instructions_template")
            .replace(
                "{{ personality }}",
                get_from("codex-rs/core/models.json[config.model].model_messages.instructions_variables[config.personality]"),
            )
        )}""",
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""<permissions instructions>
{get_from("codex-rs/protocol/src/prompts/permissions/sandbox_mode/danger_full_access.md").replace("{network_access}", "enabled")}
{get_from("codex-rs/protocol/src/prompts/permissions/approval_policy/never.md")}
</permissions instructions>""",
                    },
                    {
                        "type": "input_text",
                        "text": f"""<collaboration_mode>
{get_from("upstream collaboration default prompt")}
</collaboration_mode>""",
                    },
                    {
                        "type": "input_text",
                        "text": f"""<skills_instructions>
## Skills
A skill is a set of local instructions to follow that is stored in a `SKILL.md` file. Below is the list of skills that can be used. Each entry includes a name, description, and file path so you can open the source for full instructions when using a specific skill.
### Available skills
{"".join(f"- {name}: {description} (file: {path})\n" for name, description, path in InstalledSkills)}### How to use skills
{get_from("upstream fixed skills guidance")}
</skills_instructions>""",
                    },
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""# AGENTS.md instructions for {get_cwd()}

<INSTRUCTIONS>
{get_from("~/.codex/AGENTS.md")}

--- project-doc ---

{get_from("./AGENTS.md")}
</INSTRUCTIONS>""",
                    },
                    {
                        "type": "input_text",
                        "text": f"""<environment_context>
  <cwd>{get_cwd()}</cwd>
  <shell>{get_shell_name() *like bash}</shell>
  <current_date>{date(YYYY-MM-DD)}</current_date>
  <timezone>{IANA timezone, *like Asia/Hong_Kong}</timezone>
</environment_context>""",
                    },
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""{latest_user_turn_text}""",
                    },
                ],
            },
        ],
        "tools": [
            "exec_command",
            "write_stdin",
            "update_plan",
            "request_user_input",
            "apply_patch",
            "web_search",
            "view_image",
            "spawn_agent",
            "send_input",
            "resume_agent",
            "wait_agent",
            "close_agent",
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "store": False,
        "stream": True,
        "include": [
            "reasoning.encrypted_content",
        ],
        "prompt_cache_key": "{same as x-client-request-id}",
        "reasoning": {
            "effort": "{configured reasoning effort, *like medium}",
            "summary": "{configured reasoning summary, *like auto}",
        },
        "text": {
            "verbosity": "{configured verbosity, *like medium}",
        },
    },
}
```

### 1.2 Our 相对 Codex 的差异

当前这条 fake-server capture 下：

当前默认 CLI 的 non-exec 首轮请求，在逻辑 request body 上已经基本对齐。

仍值得记录的区别：

- `instructions` 的来源文件不同：Codex 从 `codex-rs/core/models.json` 取，`pycodex` 从 `./pycodex/prompts/models.json` 取。
- `permissions` prompt 的来源目录不同：Codex 从 `codex-rs/protocol/src/prompts/permissions/...` 取，`pycodex` 从 `./pycodex/prompts/permissions/...` 取。
- `collaboration_mode` block 的来源不同：Codex 用上游协作提示模板，`pycodex` 用 `./pycodex/prompts/collaboration_default.md` / `./pycodex/prompts/collaboration_plan.md`。
- `skills guidance` 的来源不同：Codex 用上游固定 guidance，`pycodex` 用 `./pycodex/context.py::SKILLS_GUIDANCE`。
- `tools` 的构造来源不同：Codex 从上游 runtime tool registry 出来，`pycodex` 从 `./pycodex/prompts/exec_tools.json + ToolSpec.serialize()` 出来。

### 1.3 首轮请求不变量

```python
FirstTurnInvariants = {
    "prompt_cache_key": "{uuid7() for each session}",
    "x-client-request-id": "{same as prompt_cache_key}",
    "session_id": "{same as prompt_cache_key}",
    "turn_id": "{uuid7() for each turn(user input)}",
    "reuse_rule": [
        "prompt_cache_key == x-client-request-id == session_id",
        "turn_id 只出现在 x-codex-turn-metadata 里",
    ],
    "input_shape": [
        "input[0] = developer message",
        "input[1] = contextual user message",
        "input[2] = latest user turn",
    ],
}
```

## 2. 工具 follow-up 请求

```python
ToolFollowUpTurn = {
    "status": "captured in part",
    "path": "second turn after an earlier assistant reply",
}
```

这一节已经抓到了两类真实包：

- 上游 `codex-tui` 首轮后，通过 `resume` 继续发第二轮请求。
- `pycodex` REPL 在同一个本地会话里连续发送两轮请求。

这两类抓包已经足够确认第二轮 `input` 的核心历史形状；但 `workspaces` 是否应在
第二轮继续出现，还需要同进程上游轨迹再确认。

### 2.1 Codex 基准：当前已抓到的第二轮事实

```python
CodexSecondTurnCapturedFacts = {
    "headers.originator": "codex-tui",
    "headers.user-agent": "{*like codex-tui/<version> (...)}",
    "headers.x-client-request-id": "{same session uuid7 as first request}",
    "headers.session_id": "{same session uuid7 as first request}",
    "headers.x-codex-turn-metadata.turn_id": "{new uuid7 for the second turn}",
    "body.prompt_cache_key": "{same session uuid7 as first request}",
    "body.instructions": "{same as first request}",
    "body.tools": "{same as first request}",
    "body.input": [
        "developer message",
        "contextual user message",
        "first user turn",
        "assistant reply to first turn",
        "second user turn",
    ],
}
```

### 2.2 `input` 新增 item 的形状

```python
AssistantMessageItem = {
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "output_text",
            "text": f"""{assistant_text}""",
        },
    ],
}

FunctionToolCallItem = {
    "type": "function_call",
    "name": f"{tool_name}",
    "arguments": json.dumps(tool_args, ensure_ascii=False),
    "call_id": f"{call_id}",
}

FunctionToolResultItem = {
    "type": "function_call_output",
    "call_id": f"{call_id}",
    "output": f"""{serialized_tool_output}""",
}

CustomToolCallItem = {
    "type": "custom_tool_call",
    "name": f"{tool_name}",
    "input": f"""{freeform_tool_input}""",
    "call_id": f"{call_id}",
}

CustomToolResultItem = {
    "type": "custom_tool_call_output",
    "name": f"{tool_name}",
    "call_id": f"{call_id}",
    "output": f"""{freeform_tool_output}""",
}
```

### 2.3 Our 相对 Codex 的差异

当前已经抓到 `pycodex` REPL 的第二轮请求。

当前结论：

- 第二轮请求里的 `input` 历史形状已经和当前抓到的上游轨迹对齐。
- 第二轮请求现在也会把上游返回的 `reasoning` item 一起带回下一次请求。
- `prompt_cache_key` 在两轮之间会复用。
- 第二轮会生成新的 `turn_id`。

仍待进一步确认的点：

- 上游同进程 TUI 第二轮是否稳定继续携带 `workspaces`；目前抓到的 upstream
  `resume` 第二轮没有，而 `pycodex` REPL 第二轮当前会继续带。

当前代码已知行为：

- 同一个 `AgentLoop.run_turn(...)` 内会复用同一个 `Prompt.turn_id`。
- 同一个 `ResponsesModelClient` 内会复用同一个 `session_id / prompt_cache_key`。
- 第二轮请求仍然使用同一组 tools。
- 当 Responses API 返回 `reasoning` item 时，`pycodex` 现在会把它保留进 history，并在下一轮原样回传。

## 3. 交互模式请求

```python
InteractiveTurn = {
    "status": "derived",
    "path": "interactive REPL after the first submitted turn",
}
```

这一节当前还没有通过 fake server 抓到真实包，所以这里只记录默认 CLI 非 `exec`
路径已经确认的事实之外，REPL 连续多轮部分仍待验证的内容。

### 3.1 Codex 基准：当前已确认的 non-exec 事实

```python
CodexNonExecConfirmedFacts = {
    "headers.originator": "codex-tui",
    "headers.user-agent": "{*like codex-tui/<version> (<os_name> <os_version>; <arch>) <terminal> (codex-tui; <version>)}",
    "body.input[0].content[1]": "<collaboration_mode>...</collaboration_mode>",
    "body.tools": [
        "exec_command",
        "write_stdin",
        "update_plan",
        "request_user_input",
        "apply_patch",
        "web_search",
        "view_image",
        "spawn_agent",
        "send_input",
        "resume_agent",
        "wait_agent",
        "close_agent",
    ],
}
```

### 3.2 Our 相对 Codex 的差异

默认 CLI 的 non-exec 首轮请求现在已经切到 `codex-tui`，并且会注入
`<collaboration_mode>`。

当前代码已知差异：

- `collaboration_mode` 内容来自 `./pycodex/prompts/collaboration_default.md` / `./pycodex/prompts/collaboration_plan.md`。
- 默认 CLI 的 non-exec 路径当前使用 12 个 exec-mode tools。
- REPL 连续多轮路径还没有单独 fake-server capture，所以现在不能声称它已经完全和 Codex 对齐。

## 4. Tool Schema

### 4.1 Codex 基准：当前已对齐的 exec-mode tools

```python
ExecModeToolOrder = [
    "exec_command",
    "write_stdin",
    "update_plan",
    "request_user_input",
    "apply_patch",
    "web_search",
    "view_image",
    "spawn_agent",
    "send_input",
    "resume_agent",
    "wait_agent",
    "close_agent",
]
```

### 4.2 schema 形状

```python
FunctionToolSchema = {
    "type": "function",
    "name": f"{tool_name}",
    "description": f"""{tool_description}""",
    "parameters": {
        *parameter_schema,
    },
    "strict": False,
    "output_schema": {
        *output_schema,
    },
}

CustomToolSchema = {
    "type": "custom",
    "name": f"{tool_name}",
    "description": f"""{tool_description}""",
    "format": {
        "type": "grammar",
        "syntax": "lark",
        "definition": f"""{grammar_definition}""",
    },
}

ProviderBuiltinToolSchema = {
    "type": "web_search",
    "external_web_access": True,
}
```

### 4.3 Our 相对 Codex 的差异

在已经对比过的 non-interactive `exec` 路径上，最终发出去的 tool schema 已经没有差异。

当前实现方式：

- 不再使用 prompt 级别的 `serialized_tools` override。
- 在工具层直接复用上游 snapshot。
- snapshot 文件位于 `./pycodex/prompts/exec_tools.json`。
