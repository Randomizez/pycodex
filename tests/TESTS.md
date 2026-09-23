# Tests

这个文件记录当前 `pycodex` 的测试面，以及每类测试的预期行为。

## CI 环境

- `test.yml` 和 `publish.yml` 的 Python 3.6 job 使用
  `python:3.6.15-slim-bullseye`，系统依赖仅从
  `https://archive.debian.org/debian` 的 `bullseye main` 归档安装。
  不使用镜像内旧的 security / updates 源，也不假定 `bullseye-security`
  归档存在。这个冻结环境仅用于旧版本兼容性测试，不提供持续的安全更新。
- 归档源使用 `check-valid-until=no` 允许读取已过有效期的固定索引，
  但继续校验仓库签名；`APT::Update::Error-Mode=any` 保证索引更新失败时
  立即停止，不会继续拿旧索引安装已经不存在的包。
- 如果已推送的 tag 在 CI 初始化阶段失败，修复 workflow 后应从包含修复的
  branch 手动运行 `publish`，并选择正确的 `repository`。直接重跑旧的
  tag workflow 仍使用旧提交；不要为了 CI 修复强制移动已公开的 tag。

## 自动化测试

### Context 抓包审计（2026-09-23）

基线为本机 `codex-cli 0.153.4`；脚本使用隔离的 localhost fixture，
不读取用户账号配置、不调用真实模型：

```bash
env -u VIRTUAL_ENV uv sync --dev
env -u VIRTUAL_ENV uv run --dev python -m tests.compare_context_requests
env -u VIRTUAL_ENV uv run --dev python -m tests.compare_context_requests \
  --model gpt-6-astra:caishi-azure --personality default \
  --reasoning-effort xhigh --scenario plain --scenario resume
```

- 首轮、文件 resume、reasoning/tool follow-up 走生产装配和真实 HTTP。
- 默认创建新的临时目录并打印 `comparison.json`；`--root` 必须使用新的目录。
- 只对客户端动态 UUID、Chunk ID、Wall time 做动态值规范化。
  raw context 有差异时仍退出 `1`，不能把 shared context 一致当成整包一致。
- 报告同时保留 shared-context exclusions、工具声明、其他 body 和 header 差异。
- `test_context.py` 覆盖最长前缀/namespace、skills alias/顺序/YAML policy/
  model guidance 和 filesystem XML；`test_model.py` 覆盖 suffix wire model、
  lite prefix UUIDv5、assistant phase/分段及 tool 原始参数的 resume/compact 回放。
- `test_fake_responses_server.py` 验证审计投影不吞 provider ID/phase 差异，
  不会把 user 正文中的协作标签当作可排除的 developer 指令。
- 完整边界及已知差异见 `docs/CONTEXT.md`。

### 测试分层与去重

- 小幅文案、布局、样式改动做页面检查，不搭建专用 Node/DOM 模拟测试；
  状态或队列语义只保留必要的行为回归，按变更范围运行相关用例。
- 会话命令、问答、权限和订阅契约集中在 `test_session_backend.py`；
  CLI/Web 不再分别重测每一个后端命令。
- Agent/Runtime 保持单向控制：Agent 无队列引用，裸 `stop_asap` 不取消已发出的
  工具且保持 call/result 配对；Runtime 结算停止回执并启动下一批输入。
  compaction/overflow 期间停止、steer 合并、关闭排空及子 Agent 接入在
  `test_agent_state.py`、`test_runtime_state.py` 和 `test_agent.py` 覆盖。
- pre-turn、mid-turn 和 overflow compact 中收到停止请求时，不再发后续采样，
  不增加未发出请求的 iteration；溢出恢复只重试一次，不重跑已完成工具。
- `test_cli.py` 保留入口、装配、portable round-trip、steer/queue 反馈和流式展示；
  验证普通/JSON 回执不创建额外任务，移除逐个 argparse 字段、
  相同 buffer 状态和重复恢复流程的微测试。
- `test_workspace_server.py` 以路由安全、工作区 CRUD、tab 持久化、线程隔离为主；
  验证启动/清理失败仍回收线程，取消关闭调用方不打断当前 turn；
  `/history` 整条命令只产生一条控制消息，不逐行挤占聊天记录；
  不再把 CSS 颜色、函数名等实现细节当成独立回归。
- 恢复失败按缺失/空文件/损坏/无 session id 选代表状态，不再对不改变执行路径的
  persisted/closed 标志做完整笛卡尔积。数据落盘原子性、路径隔离、SSE 协议、
  截断、clock 和自然关闭等独立风险回归保留。
- `test_agent_state.py` 验证 `replace_history` 已移除，fork/resume 装配 recorder
  失败不改变原会话；延迟写入、恢复续写和无参 resume 的状态保留继续覆盖。
- 小测试文件并入所属组件：终端 API 兼容检查并入 `test_cli.py`，
  provider 迟到回调检查并入 `test_model.py`。
- 事件字段/冻结约束、纯文本和有状态展示在 `test_events.py` 验证，身份补全不修改原事件由
  `test_session_backend.py` 覆盖；`test_workspace_server.py` 固定 Web 序列化形状。

### `tests/test_events.py`

- 事件 `visualize()` 的命令、问答、权限、provider web search 和 compact 文案。
- 无状态格式化 helpers 位于 `utils/event_helpers.py`，继续在这里覆盖其文本/颜色输出和事件调用。
- 工具成功/失败/空输出、后台 exec session、Python heredoc、plan 进度及子 Agent 摘要；
  不在各前端重复遍历这些文本规则。
- `EventDisplay` 的订阅端状态隔离、着色、流缓冲/重试/fatal 顺序、steer 反馈、
  昵称替换、输入提示、快照恢复和非事件输出；不依赖终端或 SDK。
- compact summary 从计数派生，手动 compact 的结果只由命令事件输出一次，
  自动 compact 不结束 turn；ANSI 只出现在展示输出，不进入事件数据和纯文本视图。
- CLI 仅验证事件委派与实际 log/status/prompt 执行，不再重复遍历流重试规则。

### `tests/test_feishu_card.py`

- 飞书复用独立的无色 `EventDisplay`；未知事件只需实现 `render` 即能展示，
  无需增加飞书分支。测试验证回调到卡片字段的适配、近期记录长度上限和流式区域。
- 覆盖快照恢复不重复 active turn、重试移除 partial、fatal 保留 partial、
  问答输入提示和关闭/detach；保留 Markdown 降级及请求传输回归。
- 运行时隔离 HOME，避免测试读取真实飞书 refresh token。

### `tests/test_session_backend.py`

- CLI/Web/飞书的全部会话命令使用相同后端、不落为模型 user message；
  compact 只调用摘要请求，fork 同步 Agent/provider identity、保留 history/usage；
  未续写即关闭不建空文件，续写只在新文件记录 metadata 和继承历史。
- 独立 attach/detach、多前端模型/标题同步、迟加入前端恢复 active stream；
  飞书 busy 时接受 steer，显式 queue 顺序不变。
- 多题、选项标签、Other、权限 scope、结构化回答、空答/超时/detach/close
  取消及过期 request id。
- 单 worker、未启动/并发/重复关闭、工具关闭钩子只执行一次；
  关闭等待已开始命令，调用方取消不打断 worker，清理错误通过 worker 统一回传。
- 停止接纳不额外入队，拒绝 exec/clock 新唤醒；工具在已有请求排空后清理，
  包括排队中的请求重新设置 clock 的情况，清理期间禁止恢复会话。
- `test_runtime_state.py` 验证子 Agent 清理报错后仍关闭其余子 Agent，
  首个错误向调用方传播，其余错误向事件循环报告。
- 子进程验证后端装配不导入前端、Web 不导入 CLI；IPython 保持裸 Agent。

### `tests/test_agent.py`

- `test_agent_loop_runs_tool_then_returns_final_message`
  - 预期：`AgentLoop` 能完成一轮 `ToolCall -> ToolResult -> 下一轮模型回复` 闭环。
- `test_parallel_tools_share_one_model_round`
  - 预期：支持并行工具批执行，不会把两个可并行工具串行跑掉。
- `test_runtime_submission_loop_processes_turn_and_shutdown`
  - 预期：队列通过 `start()` / `close()` 处理用户提交并自然关闭，不另建 worker。

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
  - 预期：不做模式门控，要求每个问题都有非空 `options`，自动补 `isOther=true`，并回传 JSON 字符串答案 + `success=true`；没有交互 handler 时返回取消结果。后端负责问题收集，CLI/Web/飞书接入测试验证选项和 Other 通过统一输入通道返回。
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

### `tests/test_context.py`

- `ContextManager` 的 base instructions 优先级：
  - 预期：`override > config.base_instructions > model_instructions_file > 默认 prompt`。
- `ContextManager` 的 model instructions 解析：
  - 预期：能从 vendored `models.json` 按 `model + personality` 解析出与上游一致的 base instructions。
- `ContextManager` 的 prompt 拼接：
  - 预期：developer 上下文单独作为一条 `developer` message，且内部保留多个 `content` item；
  - 预期：`AGENTS.md` 指令和 `<environment_context>` 合并为一条 contextual user message，且内部保留两个 `content` item；
  - 预期：`~/.codex/AGENTS.md`、repo 根 `AGENTS.md`、当前目录 `AGENTS.override.md` 能按顺序合并；
  - 预期：permissions / skills prompt 的顺序和 shape 接近上游 Codex。
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
  - 不在 request-visible function tool payload 里序列化 `output_schema`
  - `custom_tool_call` / `custom_tool_call_output`
- 其中有一个 hardcoded reference 测试会直接固定一份示例 request body JSON，用来防止 payload format 被悄悄改坏。
- SSE 解析要能恢复 assistant message、function tool call、custom tool call。
- 如果 Responses 流里出现 `reasoning` item，协议层也要保留它，并允许下一轮请求把它原样回传。

### `tests/test_cli.py`

- 覆盖参数和 stdin/TTY 选择、单次调用和 JSON 输出、命令不进模型、失败退出码。
- 装配验证 chat/messages/vLLM/managed transport、模型选择、`codex-tui` originator、
  context 注入、工具选择和 schema 边界。
- 用本地 storage server 验证 `--put`/`--call` round-trip、默认 GBK locale 下读取 UTF-8，
  以及 `.env` 加载时过滤 `CODEX_` 前缀变量。
- 终端验证 steer/queue 反馈、重试不重复输出、fatal 保留 partial、context 和后台状态；
  清理失败也解除视图订阅；终端库参数兼容也在本文件中验证。
- 子进程内使用真实 prompt_toolkit 输入验证单次 Ctrl+C、Ctrl+D 和 POSIX SIGINT：
  空闲时正常退出，有 active/queued 请求时先排空再清理一次，不取消请求或泄漏输入任务异常。
  `[closing]` 提示及不提前 flush partial 输出的规则由 `test_events.py` 覆盖。

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

### `tests/compare_steer_request_bodies.py`

- 用 deterministic fake origin + proxy capture 比较 upstream Codex 和 `pycodex` 在 steer 流程下的两轮 `/responses` request body。
- upstream 和 `pycodex` 都通过 tmux 驱动真实交互会话：先发第一条 prompt，再在第一条 request 仍未返回时发第二条 steer prompt。
- 这个场景里不能用 proxy capture 文件出现的时刻当作“第一条 request 已到达”的信号，因为 proxy handler 只有在整条 upstream response 读完后才会落盘；脚本现在改为同步等待 fake origin 收到第 1 条 POST，再注入 steer。
- 为了比较“默认 steer”而不是本机用户配置带来的 fast-mode 噪音，脚本会给 upstream 和 `pycodex` 都生成临时 config，并显式去掉顶层 `service_tier`。
- 所需 tmux/capture helper 直接放在本脚本内。完整请求比较会保留 pycodex 移除协作指令、修改问答工具可用性描述产生的有意差异，不再承诺整包相同。

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
| `request_user_input` | 在交互 CLI 中调用一次 `request_user_input` 并选择一个答案 | 应把 JSON 字符串答案和 `success=true` 回传给下一轮；没有 handler 的非交互调用返回取消结果 |
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

- 本地验证使用隔离 HOME/CODEX_HOME 的 `uv run --dev pytest`；数量以实际 collection 为准，
  不把历史通过数量当成当前覆盖范围。
- 已对 `exec` 做过真实模型 smoke，当前通过。
- 已对 `pycodex "请只回复当前目录的 basename，不要解释。"` 做过真实配置 smoke；预期回复应等于当前 checkout 目录名。
- `request_user_input` 的模式限制和模式专用对齐脚本已删除；当前通过本地工具、共享后端及前端接入测试验证正常答复、无 handler 取消、options 校验和超时参数 clamp。
- 已手工补抓 `write_stdin` 的 round-trip；在固定参数的 prompt 下，upstream Codex 和 `pycodex` 的 `function_call` / `function_call_output` 外层 schema 一致，tool result 文本包装也一致。
- `exec_command` / `write_stdin` 的本地 unified-exec 默认截断也已补齐：省略 `max_output_tokens` 时默认走 `10_000` token 预算；长时间未轮询的未读输出缓冲会保留 upstream 同款 `1 MiB` head/tail。
- 已补抓 `apply_patch` 的 `custom_tool_call_output`；当前 `pycodex` 已对齐 upstream 的 `Exit code` / `Wall time` / `Output` 文本包装。
- 已补抓 `view_image` 的 round-trip；当前 upstream Codex 和 `pycodex` 都会把结果回传成 `function_call_output.output = [input_image]`，其中 `image_url` data URL 一致。
- 已补抓 `spawn_agent` 的最小 validation-path；当前 upstream Codex 和 `pycodex` 在缺少 `message/items` 时都会回传 `Provide one of: message or items`。
- 已补抓 sub-agent happy path：`spawn_agent` / `send_input` / `wait_agent` / `close_agent` 当前在真实链路里的 `function_call_output` 外层 shape 已和 upstream 对齐；`pycodex` 也已补齐默认昵称和 `<subagent_notification>` 回灌。
- 已补抓 `resume_agent` 的真实 happy path：当前 `pycodex` 已对齐 upstream 的 `{"status":"pending_init"}` 返回值；同一条链路里的 sub-agent request 也已收敛到 upstream 那 6 个工具。
- sub-agent request body 里的 `prompt_cache_key` 现在也已对齐：parent thread 用自己的稳定 session id，sub-agent thread 改为使用 `agent_id`。
- sub-agent request 的 `x-openai-subagent: collab_spawn` header，以及后续 turn 不再携带 `workspaces` 的 metadata 细节，也已对齐 upstream。
- 已补抓默认 CLI 主线程的两轮无工具对话：当前 upstream Codex 和 `pycodex` 的首轮/次轮 request body 与 header shape 一致；第二轮都会省略 `workspaces`。
- sub-agent 那 6 个工具的 schema 现在来自类内 `BaseTool` spec，并由 CLI serialization 测试覆盖；`pycodex/prompts/subagent_tools.json` 已删除。
- 其余文件/agent/交互类工具 smoke 见上表；`web_search` 当前主要通过 payload/事件层测试验证接入，是否在真实模型里被主动选中还受具体 prompt 与 provider 行为影响。

## 后续新增工具时的记录要求

新增 tool 后，至少补两类记录：

1. 一个 `pytest` 级别的实现测试，验证本地 handler 的确定性行为。
2. 一个真实 `pycodex` prompt 级 smoke，验证模型在真实对话里会选到该工具，并记录 prompt 与预期最终回复。
