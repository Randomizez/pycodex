# Alignment

This document records the current prompt/context alignment work between
`pycodex` and upstream Codex from `https://github.com/openai/codex`.

## Scope

The comparison in this pass focuses on the model-visible prompt assembly:

- `instructions`
- `input` items
- developer/contextual user message shape
- `AGENTS.md` / environment-context injection

It does not claim full request parity for every runtime mode yet.

## Comparison method

The alignment pass compares the real outbound `/responses` request body from
both CLIs under the same provider/config conditions.

In practice, this is done by routing both upstream Codex and `pycodex` to the
same local capture server, then diffing:

- `instructions`
- `input`
- developer/contextual-user message shape
- model-visible tool subset for the compared mode

The repository copy of that helper server now lives at
`tests/fake_responses_server.py`. It supports both:

- fake mode: return a fixed local SSE response while recording requests
- proxy mode: forward to a real upstream Responses endpoint while recording
  both requests and responses

The temporary capture artifacts used during debugging are intentionally not part
of the repository contract and are not documented here as stable project files.

## Result

As of this snapshot, prompt/context parity is achieved for the non-interactive
`exec` comparison:

- `instructions` match exactly
- `input` match exactly

In other words, the model-visible prompt dump for `pycodex` and upstream Codex
is currently identical for this comparison scenario.

## Current non-prompt status

After prompt/context parity, the next comparison layer is the full outbound
request shape. That work is in progress.

At the time of writing:

- turn-loop parity is now aligned at the default behavior level: `pycodex`
  no longer stops after a fixed 12 iterations, and instead keeps going until
  the turn naturally converges like upstream Codex
- request body parity is aligned for the compared `exec` path, modulo dynamic
  identifiers
- the default CLI non-exec first turn now follows the captured upstream
  `codex-tui` context path (`originator = codex-tui` plus
  `<collaboration_mode>` in the developer message)
- the second-turn non-exec history shape is now aligned at the message level:
  prior user turn, reasoning item, and assistant reply are threaded back in the
  same order as the captured upstream resume path
- transport/header parity is now aligned for the compared path, including the
  sub-agent `x-openai-subagent` header and the observed `workspaces` omission
  on later sub-agent turns
- tool schema parity is aligned for the compared exec-mode tool subset

The current implementation already matches:

- exec-mode tool subset size and membership for the compared path
- `include = ["reasoning.encrypted_content"]`
- model-visible prompt fields (`instructions` and `input`)
- request-scoped `prompt_cache_key`
- session-scoped request id headers
- turn metadata header shape (`turn_id` + `sandbox`)
- mode-aware `originator` header
- exact exec-mode tool schema payloads via vendored snapshot at the tool layer
- `User-Agent` string for the compared non-interactive path

The main remaining deltas are now outside the prompt dump itself:

- dynamic run-specific values such as generated session ids and turn ids
- behavior outside the compared non-interactive `exec` path and the captured
  default two-turn TUI path, especially other runtime modes not yet captured

## Proxy tool-schema compare

当前仓库新增了一个真实 proxy 抓包比较脚本：

- `uv run python tests/compare_tool_schemas.py`

它会：

- 用 `tests/fake_responses_server.py` 的 proxy 模式分别抓 upstream Codex 和
  `pycodex` 的首个 `/responses` 请求
- 从 `tests/TESTS.md` 的真实 smoke tool 表读取工具顺序
- 逐个比较这条被抓到的 request path 里真正暴露给模型的 tool schema

在当前默认 CLI non-exec / `codex-tui` 这条被抓到的路径上，已经确认 schema
一致的工具有：

- `exec_command`
- `write_stdin`
- `update_plan`
- `request_user_input`
- `apply_patch`
- `web_search`
- `view_image`
- `spawn_agent`
- `send_input`
- `resume_agent`
- `wait_agent`
- `close_agent`

同一条被抓到的路径下，当前 upstream Codex 和 `pycodex` 都没有暴露这些工具：

- `shell`
- `shell_command`
- `exec`
- `wait`
- `request_permissions`
- `grep_files`
- `read_file`
- `list_dir`

这里的结论只针对当前被抓到的默认 `codex-tui` request path；它不等价于说这些
工具在上游全局不存在，只说明这次实际 context capture 没把它们带进首轮请求。

## Tool-call / tool-result schema compare

首轮 `tools` snapshot 之外，当前又补做了一轮“真实触发工具调用之后”的 schema
对比。这里比较的不是模型会不会稳定选中某个工具，而是：

- 当模型真的发起该工具调用时，stream 里的 `tool_call` item 外层 schema 是否一致
- 当客户端把工具执行结果喂回下一轮时，请求里的 `tool_result` item 外层 schema
  是否一致

这一步故意只比较外层 envelope / 字段 shape，不把模型生成的具体参数值、补丁正文、
搜索 query 文本等内容差异误判为 schema 差异。

另外，Codex 实际选参本身带随机性：同一个 request body，模型有时会显式补出默认参数，
有时不会。为了把“模型随机发挥”和“协议 / context 没对齐”区分开，这里的
`round-trip same` 采用两层标准：

- 先确认真实工具调用链路已经发生，且 `tool_call` / `tool_result` 外层 shape 一致
- 如果默认参数显式与否会抖动，再补一条更强约束的 prompt，把参数固定住后再看
  round-trip 文本包装是否也一致

当前已手动完成的真实对比样本：

- `web_search`
  - `web_search_call` item schema 一致
  - 这是 provider-native tool，没有单独的客户端 `tool_result` round-trip
  - 当前仓库对 provider-native tool 分两层处理：
    - `pycodex` 主路径仍然直接向模型暴露 upstream snapshot 里的
      `{"type":"web_search","external_web_access":true}` payload，不把它改写成
      本地 function tool
    - `responses_server` compat 路径为了接到下游 `/v1/chat/completions` backend，
      会在 provider 侧把 incoming `web_search` 改写成一个同名 mock function
      tool；如果下游模型选择了这个 tool，server 会先回放一个
      `web_search_call` item，再自动补一轮空结果
      `{"results":[],"mock":true,...}` 的 tool follow-up，继续后续 chat
      completions 采样
- `exec_command`
  - `function_call` item schema 一致
  - 下一轮里的 `function_call_output` schema 一致
  - 当前样本里，`arguments` 默认值和 tool result 文本包装也已经对齐；
    剩下的差异主要是动态值，例如 `id` / `call_id` / `chunk id` / `wall time`
- `update_plan`
  - `function_call` item schema 一致
  - 下一轮里的 `function_call_output` schema 一致
  - 参数值会因模型自由生成 explanation / step 文案而不同，但外层 shape 一致
- `request_user_input`
  - 当前默认 `codex-tui` / Default mode 路径下，强制触发这个 tool call 时，
    upstream Codex 和 `pycodex` 都会回传同一个固定错误：
    `request_user_input is unavailable in Default mode`
  - 这说明 Default-mode 下的 `function_call` / `function_call_output` round-trip
    已对齐
  - Plan-mode happy path 现在也已按 upstream 源码建模：handler 会要求每个问题都带
    非空 `options`、自动给每个问题补 `isOther=true`，并把结构化答案序列化成
    JSON 字符串回传到下一轮 `function_call_output.output`，同时补 `success=true`
  - 当前仓库已经新增 deterministic proxy compare 脚本
    `uv run python tests/compare_request_user_input_roundtrip.py`
  - 该脚本会用同一套固定 origin SSE + proxy capture，同步比较 upstream Codex
    和 `pycodex` 的 Plan-mode round-trip。当前在本机已安装的
    `codex-cli 0.115.0` 上，first request 的 Plan-mode collaboration prompt
    与 `function_call` 已一致；second request 里唯一剩余的 live schema 差异是
    `pycodex` 的 `function_call_output` 多了 `success=true`
  - 这和当前 GitHub `openai/codex` `main` 分支源码并不完全一致：源码里的
    `FunctionCallOutputPayload` 与 `request_user_input` handler 都允许/传递
    `success`。因此这里需要区分“本机 installed Codex 0.115.0 live capture”
    和“upstream main 源码建模”两层结论
- `view_image`
  - `function_call` item schema 一致
  - 下一轮里的 `function_call_output` schema 一致
  - 当前样本里，两边都会把 tool result 回传成同一个 `input_image` 列表，
    `image_url` data URL 也一致；当前抓到的默认样本没有显式 `detail`
- `spawn_agent`
  - 当前先补齐了一个最小 validation-path：当模型在没有 `message` / `items` 的情况下
    强制调用 `spawn_agent` 时，upstream Codex 和 `pycodex` 现在都会回传同一个固定错误：
    `Provide one of: message or items`
  - happy path 也已经补抓到：两边都会把结果回传成 JSON 字符串，字段名同为
    `agent_id` / `nickname`
  - 当前 `pycodex` 也已经改成 uuid7 agent id，并接上了与 upstream 同一批候选名的
    默认昵称池；剩余差异主要只在具体抽到哪个昵称这类动态值
- `send_input`
  - `function_call` item schema 一致
  - 下一轮里的 `function_call_output` schema 一致
  - 当前样本里回传的都是紧凑 JSON 字符串 `{submission_id: ...}`；差异仅剩动态
    `submission_id`
- `wait_agent`
  - `function_call` item schema 一致
  - 下一轮里的 `function_call_output` schema 一致
  - 当前样本里，`status` / `timed_out` 结构一致；差异主要只剩动态 agent id
- `close_agent`
  - `function_call` item schema 一致
  - 下一轮里的 `function_call_output` schema 一致
  - 当前仓库已把返回键名从 `previous_status` 改成 `status`，与 upstream 当前 happy
    path 对齐
- `resume_agent`
  - 真实 happy path 已补抓：子 agent 完成、`close_agent`、`resume_agent`、再
    `send_input` 的完整链路现在已经对齐
  - 当前样本里，upstream 在 `resume_agent` 之后返回的是
    `{"status":"pending_init"}`，`pycodex` 已同步改成这个状态值
  - 同一条链路里还顺手补齐了两个 request-level 差异：
    sub-agent request 现在只暴露 upstream 那 6 个工具
    `exec_command` / `write_stdin` / `update_plan` / `apply_patch` /
    `web_search` / `view_image`，并且不再注入 `<collaboration_mode>` developer
    block
  - request body 里的 `prompt_cache_key` 现在也改成和 upstream 一样：
    parent thread 维持自己的稳定 session id，而 sub-agent thread 则改用
    `agent_id` 本身，不再错误复用 parent 的 cache key
  - 这 6 个 sub-agent tool schema 现在也已经固化到
    `pycodex/prompts/subagent_tools.json`，并由测试逐字节锁定
- `sub-agent notification`
  - 在 `wait_agent` 之后，upstream 会向 parent thread history 额外注入一条
    `user` message：
    `<subagent_notification>\n{...}\n</subagent_notification>`
  - 这条 higher-level history shape 当前也已经在 `pycodex` 里补齐
- `write_stdin`
  - `function_call` item schema 一致
  - 下一轮里的 `function_call_output` schema 一致
  - 为了排除模型随机显式补默认参数，这一项额外用更强约束的 prompt 固定了
    `yield_time_ms` / `max_output_tokens` / `tty`，当前受控样本里 tool result 文本包装
    也已对齐，只剩 `session_id` / `call_id` / `chunk id` / `wall time` 这类动态值
  - 本地实现现在也已补齐 upstream unified-exec 的两个默认截断细节：省略
    `max_output_tokens` 时默认走 `10_000` token 预算；长时间未轮询时，未读输出缓冲
    也会像 upstream 一样只保留 `1 MiB` 的 head/tail
- `apply_patch`
  - `custom_tool_call` item schema 一致
  - 下一轮里的 `custom_tool_call_output` schema 一致
  - 当前样本里，tool result 文本包装也已经对齐到 upstream developer-tool 风格：
    `Exit code` / `Wall time` / `Output` 三段式前缀已补齐；剩余差异主要只和被修改的
    具体相对路径有关
  - 在 `responses_server` -> downstream `/v1/chat/completions` 兼容路径上，当前实现
    也 follow upstream 早期 chat-completions 做法：wire 上不再直接透传 `custom`
    tool，而是把它适配成同名 function tool（单个 `input: string` 参数），再在
    server 边界把 chat backend 的 function tool call 重新翻回
    `custom_tool_call`

### Per-tool status

下面这个表把 `tests/TESTS.md` 里的工具全部列出来。状态说明：

- `not exposed`：在当前默认 `codex-tui` 首轮 request path 下两边都没把这个工具带进 `tools`
- `first-request same`：首轮 `tools` schema 已确认一致
- `round-trip same`：真实触发后的 `tool_call` / `tool_result` 外层 schema 已确认一致
- `pending`：这条工具链还没有补完真实触发对比

| tool | current status | note |
|---|---|---|
| `shell` | `not exposed` | 当前默认 `codex-tui` 首轮路径不带这个工具 |
| `shell_command` | `not exposed` | 当前默认 `codex-tui` 首轮路径不带这个工具 |
| `exec_command` | `round-trip same` | `function_call` / `function_call_output` 外层 shape 一致；默认 `10_000` token 截断和未读输出 `1 MiB` head/tail cap 也已补齐，仅剩动态值差异 |
| `write_stdin` | `round-trip same` | `function_call` / `function_call_output` 外层 shape 一致；默认 `10_000` token 截断和未读输出 `1 MiB` head/tail cap 也已补齐，仅剩动态值差异 |
| `exec` | `not exposed` | 当前默认 `codex-tui` 首轮路径不带这个工具 |
| `wait` | `not exposed` | 当前默认 `codex-tui` 首轮路径不带这个工具 |
| `web_search` | `round-trip same` | `web_search_call` shape 一致；provider-native tool 无单独客户端 `tool_result` |
| `update_plan` | `round-trip same` | `function_call` / `function_call_output` 外层 shape 一致 |
| `request_user_input` | `round-trip same (Default mode); Plan mode delta` | Default-mode unavailable 路径已 capture 对齐；Plan-mode deterministic proxy compare 已补做：本机 installed `codex-cli 0.115.0` 的 live capture 里，`function_call` 已一致，`function_call_output` 仅差 `pycodex` 多带 `success=true` |
| `request_permissions` | `not exposed` | 当前默认 `codex-tui` 首轮路径不带这个工具 |
| `apply_patch` | `round-trip same` | `custom_tool_call` / `custom_tool_call_output` 外层 shape 一致；当前样本里输出包装也已对齐，仅剩具体文件路径差异 |
| `grep_files` | `not exposed` | 当前默认 `codex-tui` 首轮路径不带这个工具 |
| `read_file` | `not exposed` | 当前默认 `codex-tui` 首轮路径不带这个工具 |
| `list_dir` | `not exposed` | 当前默认 `codex-tui` 首轮路径不带这个工具 |
| `view_image` | `round-trip same` | `function_call` / `function_call_output` 外层 shape 一致；当前样本里 `input_image` data URL 也一致 |
| `spawn_agent` | `round-trip same` | validation-path 与 happy-path 都已补抓；剩余主要是动态 agent id / nickname 值 |
| `send_input` | `round-trip same` | `function_call` / `function_call_output` 外层 shape 一致；仅剩动态 `submission_id` |
| `resume_agent` | `round-trip same` | 已补抓真实 happy path；`resume_agent` 后的 `pending_init` 返回值、sub-agent tool 子集、sub-agent context 都已对齐 |
| `wait_agent` | `round-trip same` | `function_call` / `function_call_output` 外层 shape 一致；仅剩动态 agent id |
| `close_agent` | `round-trip same` | `function_call` / `function_call_output` 外层 shape 一致；parent-thread notification message 也已补齐 |

### Redacted example: request-level diff categories

The remaining request diff is roughly of this form:

```text
same:
- instructions
- input
- include
- exec-mode tool subset membership
- request context field presence
- exec-mode tool schemas
- user-agent semantics and compared string

different:
- dynamic request metadata values
- transport-layer header casing / normalization
- paths and modes not yet aligned beyond non-interactive `exec`
```

## Redacted examples

The exact captured prompt is intentionally not embedded here because local
`AGENTS.md` contents can contain machine- or user-specific details. Instead,
this section records the shape of the aligned context using redacted examples.

### Example: developer message shape

After alignment, the compared non-interactive path uses one `developer`
message with multiple `content` items, schematically like this:

```text
role=developer
content[0]:
<permissions instructions>
Filesystem sandboxing defines ...
Approval policy is currently never ...
</permissions instructions>

content[1]:
<skills_instructions>
## Skills
- skill-a: ...
- skill-b: ...
</skills_instructions>
```

### Example: contextual user message shape

The contextual user prompt is one `user` message with two `content` items:

```text
role=user
content[0]:
# AGENTS.md instructions for /workspace/project

<INSTRUCTIONS>
[merged global and repo instructions]
</INSTRUCTIONS>

content[1]:
<environment_context>
  <cwd>/workspace/project</cwd>
  <shell>bash</shell>
  <current_date>YYYY-MM-DD</current_date>
  <timezone>Region/City</timezone>
</environment_context>
```

### Example: parity criterion

The alignment check is based on the model-visible prompt structure, not on any
particular local path. In redacted form, the equality target is:

```text
official.instructions == pycodex.instructions
official.input == pycodex.input
```

with any machine-specific values normalized or compared under the same config.

## Main fixes that enabled parity

### 1. Base instructions source

`pycodex` no longer relies on a simplified static prompt for this path.

Instead, `pycodex/context.py` now resolves base instructions from:

- config override
- config `base_instructions`
- config `model_instructions_file`
- vendored upstream `pycodex/prompts/models.json`

This is what made the upstream GPT-5.4 pragmatic instructions match exactly.

### 2. Developer message shape

The developer message now matches upstream structure:

- one `developer` message
- multiple `content` items inside that message

For the compared `exec` path, the two content blocks are:

- `<permissions instructions> ... </permissions instructions>`
- `<skills_instructions> ... </skills_instructions>`

### 3. Contextual user message shape

The contextual user prompt now matches upstream structure:

- one `user` message
- two `content` items inside that message

Those two blocks are:

- `# AGENTS.md instructions for ... <INSTRUCTIONS> ... </INSTRUCTIONS>`
- `<environment_context> ... </environment_context>`

### 4. `AGENTS.md` merge behavior

`ContextManager` now merges:

- `~/.codex/AGENTS.md`
- repo `AGENTS.md`
- deeper `AGENTS.override.md` / `AGENTS.md`

in the same model-visible order used for the compared upstream run.

### 5. Skills section

`ContextManager` now scans the Codex skills directories and renders the same
`<skills_instructions>` block shape used by upstream Codex for this path.

### 6. Environment context details

The `<environment_context>` block now matches upstream formatting and values,
including the IANA timezone name (`Asia/Hong_Kong` rather than `HKT`).

### 7. Exec-mode tool exposure

For non-interactive `pycodex`, `get_tools(exec_mode=True)` now matches the
upstream `codex exec` tool subset so prompt comparison is done against the same
tool surface.

## Files involved

Primary implementation files:

- `pycodex/context.py`
- `pycodex/agent.py`
- `pycodex/cli.py`
- `pycodex/protocol.py`
- `pycodex/model.py`

Vendored upstream prompt data:

- `pycodex/prompts/models.json`
- `pycodex/prompts/permissions/sandbox_mode/`
- `pycodex/prompts/permissions/approval_policy/`
- `pycodex/prompts/exec_tools.json`

Tests:

- `tests/test_context.py`
- `tests/test_cli.py`
- `tests/test_model.py`

## What is still out of scope here

Prompt parity is not the same thing as full request parity.

At the time this file was written, the remaining request-level differences are
outside the prompt/context dump itself, for example:

- dynamic request metadata values such as generated session ids and turn ids
- behavior outside the currently aligned non-interactive `exec` path
- broader runtime features such as sandbox / approvals / compact / memory

Those are the next alignment target after the prompt/context pass.

## Steer semantics

当前 upstream Codex 的 steer 行为需要和显式 queue 区分开看：

- TUI 层语义：
  - 开启 steer 后，`Enter` 是 `Submitted`，`Tab` 才是 `Queued`
  - 代码：`tui/src/bottom_pane/chat_composer.rs:401`、
    `tui/src/bottom_pane/chat_composer.rs:2130`
- TUI 提交层：
  - `Submitted` 直接走 `submit_user_message(...)`
  - `Queued` 走本地 `queue_user_message(...)`，等 idle 后才 `maybe_send_next_queued_input()`
  - 代码：`tui/src/chatwidget.rs:2694`、`tui/src/chatwidget.rs:3146`、
    `tui/src/chatwidget.rs:3606`

更关键的是 core 侧的实际打断语义：

- 新的 `Op::UserTurn` 进入 core 后，先构造新的 `TurnContext` / 更新 session config，
  然后优先尝试 `inject_input(items)`
  - 代码：`core/src/codex.rs:1163`、`core/src/codex.rs:2583`、
    `core/src/codex.rs:2648`
- 如果当前有 active turn，`inject_input(...)` 会把新输入塞进当前
  `ActiveTurn.turn_state.pending_input`，而不是立刻 `spawn_task(...)`
  - 代码：`core/src/codex.rs:2145`
- 当前 `run_turn(...)` 会在每次 sampling request 结束后检查
  `has_pending_input()`，若存在则 `needs_follow_up = true` 并继续下一轮
  - 取 pending input 并先写入 history：`core/src/codex.rs:3388`
  - sampling 完成后触发 follow-up：`core/src/codex.rs:4322`

因此，对 upstream steer 更准确的理解是：

- 它不是在任意时刻硬杀当前模型/工具调用；而是等“当前这次向模型发起的请求”结束后，再把 steer 插进下一轮 prompt
- 这通常表现为：
  - 最近一次模型响应 batch 完成（`ResponseEvent::Completed`），或
  - 如果这一轮里模型先发起了工具调用，则要等工具跑完、结果回流给模型、并且这次模型请求结束
- 但它**不等价于**“任意一次工具结束就立刻插入”
  - 如果当前 turn 还卡在长工具 / 长 sampling request 中，新输入只会先进入
    `pending_input`
  - 要等当前这次模型请求自然结束后，下一轮 prompt 才会带上这条 steer 输入

关于 history 保留/丢弃：

- 已经形成 `ResponseItem` 并通过 `handle_output_item_done(...)` 记录进 history 的内容会保留
  - 代码注释已明确说明“even if the turn is later cancelled”
  - 代码：`core/src/stream_events_utils.rs:24`
  - 非工具项立即落 history：`core/src/stream_events_utils.rs:79`
  - 工具调用项立即落 history：`core/src/stream_events_utils.rs:53`
- 新的 steer 输入本身也会在下一轮 sampling 前先写入 history
  - 代码：`core/src/codex.rs:3397`
- 会丢的是还没形成最终 `ResponseItem` 的半截流式内容（例如只有 delta、尚未 item done）

关于 `Replaced` 与 `Interrupted`：

- 真正的 `TurnAbortReason::Replaced` 出现在 `spawn_task(...)` 前的
  `abort_all_tasks(TurnAbortReason::Replaced)`，不是 active-turn steer 的首选路径
  - 代码：`core/src/tasks/mod.rs:113`
- `Interrupted` 会写 `<turn_aborted>` marker 到 model-visible history；
  `Replaced` 不会
  - 代码：`core/src/tasks/mod.rs:258`
- app-server thread history 对 `Replaced` 的外显语义是：
  - 前一个 turn 状态记为 interrupted
  - 但该 turn 已经存在的 user/assistant items 保留
  - 测试：`app-server-protocol/src/protocol/thread_history.rs:404`

当前 `pycodex` 的对齐策略：

- 我们优先对齐“下一次发出去的请求体”，尤其是 `input` 里的 messages 顺序，而不是强求内部控制流和 upstream 完全同构
- 当前实现里，`AgentRuntime` 负责 steer/queue 语义；内部实现已经收成两个同构 queue：`enqueue` 与 `steer` 走同一个 runtime API，只是调度优先级和 steer 的 interrupt/coalesce 规则不同
- 运行中收到 steer 时，runtime 会先请求当前 `AgentLoop` 在安全边界停下，再把多个 steer 文本批量交给下一次 `run_turn(texts, turn_id=...)`
- 因此，下一次请求体里的 `input` 现在可以把多个 steer 文本按顺序并到 history 尾部，并且继续沿用同一个 `turn_id`；这一点已经明显比旧版 `cancel + 单条新 turn` 更接近 upstream
- 通过 `tests/compare_steer_request_bodies.py` 的 fake/proxy capture 对比，当前 steer 首轮/次轮 request body 在忽略 `prompt_cache_key` 后已与本机 installed `codex-cli 0.115.0` 对齐；这里比较的是“默认 steer”路径，因此脚本会先去掉本机用户配置里的顶层 `service_tier`，避免把本地 fast-mode 设置误记成 steer 差异。同一 steer turn 的 follow-up request 仍需继续带 `workspaces`
- 仍未完全一致的点主要是内部控制流：本地实现仍是在 runtime 层结束一次 `run_turn(...)` 再启动下一次；upstream 则更倾向于在同一个 active turn 里继续 follow-up
