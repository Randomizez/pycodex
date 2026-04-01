# Tests

这个文件记录当前 `pycodex` 的测试面，以及每类测试的预期行为。

## 自动化测试

### `tests/test_agent.py`

- `test_agent_loop_runs_tool_then_returns_final_message`
  - 预期：`AgentLoop` 能完成一轮 `ToolCall -> ToolResult -> 下一轮模型回复` 闭环。
- `test_parallel_tools_share_one_model_round`
  - 预期：支持并行工具批执行，不会把两个可并行工具串行跑掉。
- `test_runtime_submission_loop_processes_turn_and_shutdown`
  - 预期：`AgentRuntime` 能正常处理用户提交和 shutdown。

### `tests/test_builtin_tools.py`

覆盖当前已接入的本地工具实现：

- `shell`
  - 预期：按 argv 执行命令；能返回工作目录、退出码、stdout/stderr；超时能正确报出。
- `shell_command`
  - 预期：按 shell script 字符串执行命令。
- `exec_command`
  - 预期：长命令会返回运行中的 `session_id`。
- `write_stdin`
  - 预期：能复用已有 session，向 stdin 写入并拿到后续输出。
- `exec`
  - 预期：能运行 raw JavaScript；支持返回 completed 或 running cell 状态。
- `wait`
  - 预期：能等待 `exec` cell 的后续输出或完成态。
- `web_search`
  - 预期：按 provider-native tool 形式序列化到 Responses API payload 中，而不是走本地 handler round-trip。
- `update_plan`
  - 预期：更新 `PlanStore`，并返回固定确认文本 `Plan updated`。
- `request_user_input`
  - 预期：Default mode 下立即返回固定错误 `request_user_input is unavailable in Default mode`；Plan mode 下要求每个问题都带非空 `options`、自动补 `isOther=true`，并回传 JSON 字符串答案 + `success=true`；CLI 侧的问题收集 helper 另有单测单独覆盖。
- `request_permissions`
  - 预期：把权限请求转发给交互层，并返回带 `scope` 的权限响应。
- `apply_patch`
  - 预期：支持 add/delete/update；整体验证失败时不留下部分写入。
- `grep_files`
  - 预期：只返回匹配文件路径，不把不匹配文件混进结果。
- `read_file`
  - 预期：按 1-based 行号切片读取，并带行号前缀。
- `list_dir`
  - 预期：列出绝对路径目录树片段。
- `view_image`
  - 预期：返回 data URL，并把工具结果序列化成 `input_image` content item。
- `spawn_agent`
  - 预期：happy path 下能创建子 agent 并通过 `wait_agent` / `close_agent` 正常回收；当前还额外校验了空参数时要返回固定校验错误 `Provide one of: message or items`。
- `send_input`
  - 预期：能给已有子 agent 继续发消息。
- `resume_agent`
  - 预期：关闭后的子 agent 能被恢复并继续接收输入。
- `wait_agent`
  - 预期：能等待子 agent 到达完成态。
- `close_agent`
  - 预期：能关闭子 agent，并返回关闭前状态。

### `tests/test_cli.py`

- 参数解析、TTY 判定、交互模式 `/history` / `/title` 行为。
- `get_tools()` 注册结果必须和当前接入的 builtin tools 集合一致。
- `get_tools(exec_mode=True)` 必须收敛到官方 `codex exec` 的 tool 子集。
- CLI 启动前会加载与 config 同目录的 `.env`，但会过滤掉 `CODEX_` 前缀变量。

### `tests/test_context.py`

- `ContextManager` 的 base instructions 优先级：
  - 预期：`override > config.base_instructions > model_instructions_file > 默认 prompt`。
- `ContextManager` 的 model instructions 解析：
  - 预期：能从 vendored `models.json` 按 `model + personality` 解析出与上游一致的 base instructions。
- `ContextManager` 的 prompt 拼接：
  - 预期：developer 上下文单独作为一条 `developer` message，且内部保留多个 `content` item；
  - 预期：`AGENTS.md` 指令和 `<environment_context>` 合并为一条 contextual user message，且内部保留两个 `content` item；
  - 预期：`~/.codex/AGENTS.md`、repo 根 `AGENTS.md`、当前目录 `AGENTS.override.md` 能按顺序合并；
  - 预期：permissions / skills / collaboration prompt 的顺序和 shape 接近上游 Codex。
- `AgentLoop` + `ContextManager` 集成：
  - 预期：上下文只注入到本轮 `Prompt.input`，不会污染持久 history。

### `tests/test_model.py`

- `ResponsesProviderConfig` 能读 `~/.codex/config.toml` 风格配置。
- `ResponsesModelClient` 构造的 payload 要正确包含：
  - `developer` / `user` message 形式的上下文注入
  - `prompt_cache_key`
  - Codex 风格请求 headers（如 request id / turn metadata / originator）
  - 普通 function tools
  - freeform/custom tools
  - `output_schema`
  - `custom_tool_call` / `custom_tool_call_output`
- 其中有一个 hardcoded reference 测试会直接固定一份示例 request body JSON，用来防止 payload format 被悄悄改坏。
- SSE 解析要能恢复 assistant message、function tool call、custom tool call。
- 如果 Responses 流里出现 `reasoning` item，协议层也要保留它，并允许下一轮请求把它原样回传。

### `tests/test_cli.py`

- `get_tools(exec_mode=True)` 除了校验工具集合本身，还会把 `model_visible_specs()` 的序列化结果和 vendored `pycodex/prompts/exec_tools.json` 逐字节对比。
- 预期：exec-mode tool schema 对齐发生在工具定义层，而不是通过 prompt 级 override 旁路注入。
- 非交互 `run_cli(...)` 的 capture 测试会直接验证默认 CLI 非 `exec` 路径现在发的是 `codex-tui`，并且 developer message 里包含 `<collaboration_mode>`。

### `tests/fake_responses_server.py`

- 这是 prompt/context 对齐时使用的本地 fake Responses API server。
- 代码放在 `tests/` 下，临时抓包输出默认仍然写到 `.tmp/prompt_capture`。
- 预期：同一台 fake server 可以同时接 upstream Codex 和 `pycodex` 的请求，然后把 `/models` 和 `/responses` 的交互稳定记录下来。
- 现在也支持 proxy 模式：传 `--proxy-base-url <upstream>` 后，会把请求真实转发到上游，同时把上游响应一起落盘。

### `tests/test_fake_responses_server.py`

- 本地起一个“上游小服务器”和一个 proxy capture server，验证 proxy 模式会真实转发 `/responses`，并把 response status / headers / body 记进 capture 文件。

### `tests/compare_tool_schemas.py`

- 用 proxy 模式分别抓 upstream Codex 和 `pycodex` 的真实首轮 `/responses` 请求。
- 工具顺序直接从本文件里的真实 smoke table 读取，逐个比较 request-visible tool schema。
- 当前比较的是被抓到的那条具体 context path；如果某个工具在这条路径下根本没有暴露出来，会明确记成 `not exposed`，而不是误判为 schema 相等。
- 真正的 `round-trip same` 仍然要靠单独的手工抓包确认；如果某个工具的默认参数显式与否有随机性，需要用更强约束的 prompt 固定参数，再比较 `tool_call` / `tool_result`。

### `tests/compare_request_user_input_roundtrip.py`

- 用 deterministic origin SSE + proxy capture 比较 upstream Codex 和 `pycodex` 的 Plan-mode `request_user_input` round trip。
- upstream 侧通过 tmux 驱动真实 TUI：先发 `/plan`，再发 prompt，再按数字选项回答；`pycodex` 侧则直接用本地 runtime 的 Plan mode 跑同一条链路。
- 当前在本机 installed `codex-cli 0.115.0` 上，first request 的 Plan-mode prompt 和 `function_call` 已一致；second request 里唯一剩余 live schema 差异是 `pycodex` 的 `function_call_output` 多带 `success=true`。

### `tests/compare_steer_request_bodies.py`

- 用 deterministic fake origin + proxy capture 比较 upstream Codex 和 `pycodex` 在 steer 流程下的两轮 `/responses` request body。
- upstream 和 `pycodex` 都通过 tmux 驱动真实交互会话：先发第一条 prompt，再在第一条 request 仍未返回时发第二条 steer prompt。
- 这个场景里不能用 proxy capture 文件出现的时刻当作“第一条 request 已到达”的信号，因为 proxy handler 只有在整条 upstream response 读完后才会落盘；脚本现在改为同步等待 fake origin 收到第 1 条 POST，再注入 steer。
- 为了比较“默认 steer”而不是本机用户配置带来的 fast-mode 噪音，脚本会给 upstream 和 `pycodex` 都生成临时 config，并显式去掉顶层 `service_tier`。

## 真实模型 smoke tests

下面这些测试是用真实 `pycodex` + `~/.codex/config.toml` 跑过的 prompt 级 smoke，用来确认模型在真实对话里会主动选对工具。

说明：

- 文件类工具测试时使用仓库内临时目录 `.tmp_tool_smoke`。
- `spawn_agent` 相关工具有时会出现额外一轮自检或重复清理；只要目标工具被调用且最终回复符合预期，就记为通过。

| tool name | test prompt | expected behavior |
|---|---|---|
| `shell` | 必须且只需调用 `shell`，运行 `['bash','-lc','printf SHELL_OK']`，最后只回复 `SHELL_OK` | 应调用 `shell`，最终回复 `SHELL_OK` |
| `shell_command` | 必须且只需调用 `shell_command`，运行 `printf SHELL_COMMAND_OK`，最后只回复 `SHELL_COMMAND_OK` | 应调用 `shell_command`，最终回复 `SHELL_COMMAND_OK` |
| `exec_command` | 必须且只需调用 `exec_command`，运行 `printf EXEC_COMMAND_OK`，最后只回复 `EXEC_COMMAND_OK` | 应调用 `exec_command`，最终回复 `EXEC_COMMAND_OK` |
| `write_stdin` | 先 `exec_command` 跑 `read line; printf "$line"`，再 `write_stdin` 发送 `WRITE_STDIN_OK\n` | 应依次调用 `exec_command`、`write_stdin`，最终回复 `WRITE_STDIN_OK` |
| `exec` | 只调用 `exec`，运行 `text('EXEC_SMOKE_OK')`，最后只回复 `EXEC_SMOKE_OK` | 应调用 `exec`，最终回复 `EXEC_SMOKE_OK` |
| `wait` | 先 `exec` 运行一次会先 `yield_control()` 再输出 `WAIT_OK` 的脚本，再 `wait` 拿到后续输出 | 应依次调用 `exec`、`wait`，最终回复 `WAIT_OK` |
| `web_search` | 只调用 `web_search` 搜索一个明确问题，再基于搜索结果简短作答 | 应调用 `web_search`，并基于搜索结果回复 |
| `update_plan` | 先调用 `update_plan` 设两步计划，再只回复 `TOOL_OK` | 应调用 `update_plan`，最终回复 `TOOL_OK` |
| `request_user_input` | 在当前默认 `codex-tui` / Default mode 路径里强制调用一次 `request_user_input` | 应调用 `request_user_input`，并把 unavailable 错误回传给下一轮；另外还有独立 tool-level 测试覆盖 Plan-mode happy path（JSON 字符串输出 + `success=true`） |
| `request_permissions` | 只调用 `request_permissions`，请求 `network.enabled=true`；CLI 侧输入 `t` | 应调用 `request_permissions`，最终回复 `REQUEST_PERMISSIONS_OK` |
| `apply_patch` | 只调用 `apply_patch`，把目标文件里的 `before` 改成 `APPLY_PATCH_OK` | 应调用 `apply_patch`，最终回复 `APPLY_PATCH_OK` |
| `grep_files` | 只调用 `grep_files` 搜索 `NEEDLE_123` | 应调用 `grep_files`，最终回复 `grep_target.txt` |
| `read_file` | 只调用 `read_file` 读取目标文件 | 应调用 `read_file`，最终回复 `READ_FILE_OK` |
| `list_dir` | 只调用 `list_dir` 列出目标目录 | 应调用 `list_dir`，最终回复 `child` |
| `view_image` | 只调用 `view_image` 查看目标图片 | 应调用 `view_image`，最终回复 `VIEW_IMAGE_OK` |
| `spawn_agent` | `spawn_agent` 创建子 agent，再 `wait_agent` / `close_agent` 清理 | 至少应调用 `spawn_agent`、`wait_agent`、`close_agent`，最终回复 `SPAWN_AGENT_OK` |
| `send_input` | `spawn_agent` 后再 `send_input` 发第二条消息 | 至少应调用 `spawn_agent`、`send_input`、`wait_agent`、`close_agent`，最终回复 `SEND_INPUT_OK` |
| `resume_agent` | 子 agent 完成并关闭后再 `resume_agent`，然后继续 `send_input` | 至少应调用 `spawn_agent`、`wait_agent`、`close_agent`、`resume_agent`、`send_input`，最终回复 `RESUME_AGENT_OK` |
| `wait_agent` | 创建子 agent 后显式等待 | 至少应调用 `spawn_agent`、`wait_agent`、`close_agent`，最终回复 `WAIT_AGENT_OK` |
| `close_agent` | 创建子 agent 后显式关闭 | 至少应调用 `spawn_agent`、`wait_agent`、`close_agent`，最终回复 `CLOSE_AGENT_OK` |

## 当前通过情况

- `uv run pytest`：当前通过（113 tests）。
- 已对 `exec` 做过真实模型 smoke，当前通过。
- 已对 `pycodex "请只回复当前目录的 basename，不要解释。"` 做过真实配置 smoke；预期回复应等于当前 checkout 目录名。
- 已通过 tmux 驱动的真实 upstream/`pycodex` TUI capture 补抓 `request_user_input`：当前 Default mode 下，两边都会回传固定错误 `request_user_input is unavailable in Default mode`。
- `request_user_input` 的 Plan-mode happy path 现在也已有 deterministic proxy compare：`uv run python tests/compare_request_user_input_roundtrip.py` 会在本机 installed `codex-cli 0.115.0` 上复现真实链路。当前结论是 `function_call` 已一致，`function_call_output` 仅差 `pycodex` 多带 `success=true`。
- 已手工补抓 `write_stdin` 的 round-trip；在固定参数的 prompt 下，upstream Codex 和 `pycodex` 的 `function_call` / `function_call_output` 外层 schema 一致，tool result 文本包装也一致。
- `exec_command` / `write_stdin` 的本地 unified-exec 默认截断也已补齐：省略 `max_output_tokens` 时默认走 `10_000` token 预算；长时间未轮询的未读输出缓冲会保留 upstream 同款 `1 MiB` head/tail。
- 已补抓 `apply_patch` 的 `custom_tool_call_output`；当前 `pycodex` 已对齐 upstream 的 `Exit code` / `Wall time` / `Output` 文本包装。
- 已补抓 `view_image` 的 round-trip；当前 upstream Codex 和 `pycodex` 都会把结果回传成 `function_call_output.output = [input_image]`，其中 `image_url` data URL 一致。
- 已补抓 `spawn_agent` 的最小 validation-path；当前 upstream Codex 和 `pycodex` 在缺少 `message/items` 时都会回传 `Provide one of: message or items`。
- 已补抓 sub-agent happy path：`spawn_agent` / `send_input` / `wait_agent` / `close_agent` 当前在真实链路里的 `function_call_output` 外层 shape 已和 upstream 对齐；`pycodex` 也已补齐默认昵称和 `<subagent_notification>` 回灌。
- 已补抓 `resume_agent` 的真实 happy path：当前 `pycodex` 已对齐 upstream 的 `{"status":"pending_init"}` 返回值；同一条链路里的 sub-agent request 也已收敛到 upstream 那 6 个工具，并去掉了 `<collaboration_mode>` developer block。
- sub-agent request body 里的 `prompt_cache_key` 现在也已对齐：parent thread 用自己的稳定 session id，sub-agent thread 改为使用 `agent_id`。
- sub-agent request 的 `x-openai-subagent: collab_spawn` header，以及后续 turn 不再携带 `workspaces` 的 metadata 细节，也已对齐 upstream。
- 已补抓默认 CLI 主线程的两轮无工具对话：当前 upstream Codex 和 `pycodex` 的首轮/次轮 request body 与 header shape 一致；第二轮都会省略 `workspaces`。
- sub-agent 那 6 个工具的 schema 现在已固化到 `pycodex/prompts/subagent_tools.json`，并有 snapshot 测试锁定。
- 其余文件/agent/交互类工具 smoke 见上表；`web_search` 当前主要通过 payload/事件层测试验证接入，是否在真实模型里被主动选中还受具体 prompt 与 provider 行为影响。

## 后续新增工具时的记录要求

新增 tool 后，至少补两类记录：

1. 一个 `pytest` 级别的实现测试，验证本地 handler 的确定性行为。
2. 一个真实 `pycodex` prompt 级 smoke，验证模型在真实对话里会选到该工具，并记录 prompt 与预期最终回复。
