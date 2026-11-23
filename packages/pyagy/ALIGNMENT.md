# pyagy 对齐说明

本文档记录 `pyagy` 与官方 Antigravity CLI（**`agy 1.5.14`**，客户端指纹 `google-genai-sdk/1.79.1 gl-go/go1.28-20261102-RC01 cl/1012564305 +298f53e483 X:fieldtrack,boringcrypto,nogolistexportnewformat`）的对齐现状。

对照基准通过同一个本地 HTTP Capture Server 抓取双端真实发出的 Gemini `streamGenerateContent` 请求、工具往返与多轮会话轨迹核验（验证日期：`2026-11-22`）。下文先列出当前**未对齐或仅形式对齐**的部分，再列出**已对齐**的部分。

---

## 一、未对齐与仅形式对齐部分

### 1.1 纯形式对齐的禁用/占位工具（3 个）

以下 3 个工具在 `pyagy/prompts/tool_declarations.json` 中完整保留了官方名称、描述与 `parametersJsonSchema`，模型侧看到的 14 个工具声明与官方完全一致，但本地仅注册为禁用/占位类，调用时直接返回错误：

| 工具名 | 实现位置 | 调用返回与说明 |
| --- | --- | --- |
| `ask_question` | `pyagy/tools/interaction_tools.py` (`AskQuestionPlaceholderTool`) | 固定返回 `Encountered error in tool execution: ask_question is disabled`；保留工具声明但不触发交互提问或弹窗。 |
| `run_workflow` | `pyagy/tools/interaction_tools.py` (`RunWorkflowPlaceholderTool`) | 固定返回 `Encountered error in tool execution: tool is not enabled in pyagy CLI mode`；未实现 Python 工作流脚本宿主与多智能体编排原语。 |
| `swarm` | `pyagy/tools/interaction_tools.py` (`SwarmPlaceholderTool`) | 固定返回 `Encountered error in tool execution: tool is not enabled in pyagy CLI mode`；未实现共享消息板、成员名册及集群协作生命周期。 |

### 1.2 已实现工具中的局部形式对齐与行为差异（5 项）

其余 11 个工具已实现主体执行逻辑，但以下参数或子能力目前仅在 Schema / 提示词层面对齐，运行时存在简化或忽略：

1. **`edit_file` 的 `ArtifactMetadata` 参数未生效**
   - `edit_file` 的 Schema 包含 `ArtifactMetadata`（`Summary`、`UserFacing`、`RequestFeedback`），但 `EditFileTool` 未读取该参数，传入时会被静默忽略（`write_file` 则会将其写入 `.system_generated/artifacts/<name>.metadata.json`）。
   - 此外，CLI 与 `pycodex-ws` Web 前端目前均未消费 `UserFacing` 与 `RequestFeedback`（例如计划文档的交互确认按钮）。
2. **`read_file` 仅对图片实现二进制多模态，未覆盖 PDF 与音视频**
   - 工具描述声明可读取 `text, image, pdf, video, or audio file`。
   - 实际 `ReadFileTool` 仅识别 `png / jpeg / gif / webp` 四类图片并封装为 Gemini `inlineData`；传入 PDF、音频或视频文件时不会转为多模态输入，而是按 `utf-8 (errors="replace")` 当作普通文本解码并按行截断。
3. **`define_subagent` 的 `EnableMCPTools` 参数与子 Agent 原生生图工具**
   - `EnableMCPTools` 仅保存在 `SubagentBlueprint` 中，`_resolve_subagent_prompt_and_tools` 组装子 Agent 工具集时未读取该字段（`pyagy` 当前未接入 MCP Server 工具）。
   - 官方自定义子 Agent 与内置 `image-generator` 具备原生生图工具（如 `generate_image`）；`pyagy` 未实现原生生图工具，`image-generator` 目前仅挂载 `read_file`、`write_file`、`run_command`、`send_message`、`schedule`、`manage_task`。
4. **`search_web` 的检索通道与无 Grounding 降级行为**
   - `SearchWebTool` 通过复用当前下游 Gemini 接口额外发起一条带 `tools: [{"googleSearch": {}}]` 的请求并提取 `groundingChunks` 引用。
   - 若下游代理返回 HTTP `400` 或 `501`（不支持 Google Search Grounding），实现会静默回退为不带 `googleSearch` 的普通模型生成请求，此时回答仅依赖模型参数知识而非真实联网检索。
5. **`research` 子 Agent 沙箱与 CLI 权限/模式开关**
   - 遵循仓库整体不做 OS 沙箱强制和命令审批拦截的约定：`research` 子 Agent 的“无网络只读沙箱”仅由系统提示词与只读工具集约束，其 `run_command` 仍在宿主 `bash` 中直接执行。
   - CLI 参数 `--mode`（`accept-edits` / `auto` / `plan`）与 `--dangerously-skip-permissions` 仅做命令行兼容解析，不触发独立的权限审批弹窗或 Plan 状态切换。

### 1.3 旁路请求、HTTP 传输层与本地持久化差异

1. **省略官方 `SAFETY_RATER` 旁路请求**
   - 官方 `agy 1.5.14` 在模型返回 `run_command` 后、真正执行命令前，会额外向 `/v1beta/models/gemini-3.1-flash-lite-preview:streamGenerateContent?alt=sse` 发送一条无 `tools`、带 `X-Server-Timeout: 8` 且 `thinkingLevel: "MINIMAL"` 的安全评估请求（`You are a safety and security rater overseeing Jetski...`）。
   - `pyagy` 不发送该旁路请求，直接执行工具调用。
2. **JSON 原始字节转义、上行 `gzip` 压缩与 HTTP 连接头**
   - 双端请求体解压并解析后的 JSON 结构、键顺序与字段值完全一致；但官方基于 Go `encoding/json.Encoder`，会将 `<`、`>`、`&` 转义为 `\u003c`、`\u003e`、`\u0026` 并在请求体末尾追加 `\n`，且默认明文发送请求体、复用 HTTP/1.1 长连接。
   - `pyagy` 使用 Python `json.dumps(..., ensure_ascii=False, separators=(",", ":"))` 直接输出原字符且无末尾换行，并默认附带 `Content-Encoding: gzip` 压缩上行请求体（下游若返回 `415` 或 JSON 解码 `400` 则自动回退为明文重发并缓存，也可设 `PYCODEX_DISABLE_REQUEST_GZIP=1` 禁用）；底层 `urllib.request` 会附带 `Connection: close`。
3. **本地会话存储格式与 `transcript.jsonl` 详略度**
   - 官方使用内部 protobuf 会话文件，并在 `transcript.jsonl` / `transcript_full.jsonl` 的 `PLANNER_RESPONSE` 步骤中写入完整 `thinking`、`content` 与 `tool_calls` 字段。
   - `pyagy` 使用 `.system_generated/pyagy_session.json` 保存完整历史与恢复元数据（含 `anchor_tokens` 与对应的 `anchor_history_len` 锚定边界；缺少 `anchor_history_len` 的旧版存档在恢复时自动定位至最后一条模型响应条目之后，确保尾部未采样消息计入增量估算）；运行时实时追加的 `PLANNER_RESPONSE` 步骤目前仅记录 `step_index`、`created_at` 与 `model`（完整推理及工具记录保存在 `pyagy_session.json` 及 Web 模式的 Codex rollout 文件中）。

### 1.4 `pycodex-ws` Web 模式下的显式上下文扩展

当配置 `harness = "pyagy"` 从 `pycodex-ws` 启动 Web 会话时，为支持工作区功能会按 `ContextConfig` 动态注入两项额外上下文（纯 `pyagy` CLI 模式默认关闭，保持与官方 CLI 0 diff）：

1. **`contents[0]` 前置工作区上下文消息**：注入 `<environment_context>` 及（当 `enable_board=True` 时）实时 Canvas 画布路径说明，首条真实 `<USER_REQUEST>` 顺移至 `contents[1]`。
2. **合并 Codex 技能目录**：当 `include_skills_instructions=True` 且提供 `codex_home` 时，`systemInstruction` 的 `<skills>` 会合并挂载 `~/.codex/skills` 下的技能条目。

---

## 二、已对齐部分

在纯 CLI 默认模式下，将官方 `agy 1.5.14` 与 `pyagy` 同时指向同一个本地 Capture Server，多轮主 Agent 与子 Agent 请求解析后的 JSON 负载及关键协议头已实现逐字段一致（由 `packages/pyagy/tests/test_wire_parity.py` 回归覆盖）。

### 2.1 Gemini Wire 请求结构与客户端指纹

- **端点路径**：统一发送 `POST /v1beta/models/{api_model}:streamGenerateContent?alt=sse`。
- **请求头与传输压缩**：固定发送 `Content-Type: application/json`、`Accept-Encoding: gzip`、`x-goog-api-key`，默认启用 `Content-Encoding: gzip` 上行压缩（带不支持网关注册回退）及响应流 `gzip` 解压，并将 `User-agent` 与 `x-goog-api-client` 锁定为 `agy 1.5.14` 的 `google-genai-sdk/1.79.1 ...` 指纹（自动忽略 provider 配置中旧版本的同名 header 覆盖）。
- **顶层字段顺序**：固定按 `['contents', 'generationConfig', 'systemInstruction', 'tools']` 顺序序列化（显式指定 `tool_choice` 时追加 `toolConfig`）。

### 2.2 模型目录与 `generationConfig` 映射

- **25 个官方模型**：完整对齐 `OFFICIAL_MODELS` 的 slug、人类可读显示名（如 `Gemini 4 Carbon (High)`）及底层 `api_model` 映射，支持 `model:route` 路由后缀原样透传。
- **推理配置分档**：
  - Gemini 4（Neon / Carbon / Hydrogen / Argon）与 Gemini 3.8 / 3.7 / 3.6 Flash：`maxOutputTokens: 65536`，`thinkingConfig: {"includeThoughts": true, "thinkingLevel": "LOW|MEDIUM|HIGH|XHIGH"}`（仅 `neon` 与 `hydrogen` 保留 `XHIGH`，其余自动收敛至 `HIGH`）。
  - `gemini-3.1-pro-high` / `gemini-3.1-pro-low`：映射至 `gemini-3.1-pro-preview`，`maxOutputTokens: 65535`，`thinkingConfig` 使用 `thinkingBudget`（`low -> 1024`，`high -> -1`）。
  - 子 Agent 模型分档：`inherit` 继承父模型与显式 effort，`flash_lite -> gemini-3.6-flash-low`，`flash -> gemini-3.8-flash-high`，`pro -> gemini-4-carbon-high`。

### 2.3 12 段式 `systemInstruction`、技能与规则发现

- **指令装配**：`systemInstruction` 外层固定 `role: "user"`，正文严格按静态前缀（`<identity>`、`<artifacts>`、`<conversation_transcript>`、`<guidelines>`、`<work_efficiently>`、`<communication_style>`、`<slash_commands>`、`<messaging>`、`<subagents>`）+ `<skills>` + `<user_rules>` + `<user_information>` 拼接。
- **技能发现与过滤**：默认挂载 5 个内置技能（`agy-customizations`、`antigravity-guide`、`run-workflow`、`swarm_monitor`、`swarm_pane`），并按优先级扫描 `~/.gemini/config/skills` 及所有工作区从 Git 仓库根到当前目录祖先链下的 `.agents/.agent/_agents/_agent/skills`，支持过滤 `disable-model-invocation: true` 并按技能名排序。
- **规则发现与多工作区**：沿 Git 仓库根到工作区目录的祖先链按序收集 `AGENTS.md` 与 `GEMINI.md`，使用 `<RULE[绝对路径]>` 标签包裹（不混入 `~/.codex/AGENTS.md`）；支持 `--add-dir` 多工作区去重排序。

### 2.4 多轮历史、Thought 签名与工具往返格式

- **用户消息锚定包装**：自动包裹 `<USER_REQUEST>` 与带本地时区秒级时间戳的 `<ADDITIONAL_METADATA>`；首轮或模型显示名发生变更时附加 `<USER_SETTINGS_CHANGE>`，未变更轮次不重复附加，历史轮次保持原时间戳不变。
- **流式思考与签名还原**：
  - 纯文本回复：合并流式 `thought` 文本分片，并将 `thought` 与正文分片上的 `thoughtSignature`（含流末尾空文本签名分片）按字节拼接后置于最终正文 part。
  - 工具调用回复：前置 `thought` 仅在自身携带签名时保留，无签名前置 `thought` 自动丢弃；保留各 `functionCall` 原生签名，若整轮无任何签名则在首个 `functionCall` 填充 `skip_thought_signature_validator`；每个 `functionCall` 固定键序 `['args', 'id', 'name']` 并对 `args` 递归按键名排序。
- **`functionResponse` 与系统消息拼接**：
  - 工具返回外层固定 `role: "model"`，同一批次连续工具结果合并为一个 `contents` 条目，且绝不并入前一条 `functionCall` 模型消息。
  - 成功结果写入 `response.output`，失败结果写入 `response.error`（自动补齐 `Encountered error in tool execution: ` 前缀）；图片工具结果写入 `functionResponse.parts[].inlineData`，字段顺序保持 `['id', 'name', 'parts', 'response']`。
  - 工具执行期间到达的一个或多个后台任务 / 子 Agent `<SYSTEM_MESSAGE>` 自动以 `\n\n` 追加到紧邻的最后一个 `functionResponse` 文本末尾。

### 2.5 14 个工具声明与 11 个已实现工具核心行为

- **工具声明表**：全部 14 个工具的排列顺序、描述文本及 `parametersJsonSchema`（含 `$schema`、`additionalProperties`、`required` 等）与官方首轮请求完全一致。
- **文件读写编辑**：`read_file`（绝对路径校验、空文件提示、行号切片、`ByteOffset` 续读、单次 800 行 / 46080 字节上限、PNG/JPEG/GIF/WebP 图片多模态读取）、`write_file`（自动建父目录、默认防覆盖、`Overwrite`/`Append` 互斥校验、Artifact 元数据落盘）、`edit_file`（全量精确替换、替换次数及起始行与增删行数统计）。
- **命令执行与定时任务**：`run_command`（POSIX 使用 PTY 会话与进程组信号清理，Windows 回退到标准管道、`CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`、PowerShell UTF-8 `OutputEncoding` 初始化与 `taskkill /F /T` 清理进程树，并将工具描述中的运行环境切换为 `Operating System: windows`；双端统一注入 `TERM=dumb`、`PAGER=cat`、`GIT_PAGER=cat`、`JJ_PAGER=cat`、`GIT_EDITOR=false`、`HGEDITOR=false`、`JJ_EDITOR=false`、`GIT_TERMINAL_PROMPT=0`、`ANTIGRAVITY_AGENT=1`、`ANTIGRAVITY_CONVERSATION_ID` 及默认 `PYTHONUTF8=1`，在 Windows 下对环境变量名做大小写无关去重，并在主 Shell 退出而后台孙进程仍持有 PTY/管道句柄时最多等待 1 秒排空即判定命令完成，避免交互式分页器或孤儿孙进程导致永久卡死；`WaitMsBeforeAsync` 500~10000ms 同步等待或转后台任务、按 Gemini 4 / 3.x / 3.1 Pro 区分后台返回提示语、完成或取消后自动投递 `<SYSTEM_MESSAGE>`）、`manage_task`（`list` / `status` / `send_input` / `kill`）、`schedule`（单次倒计时支持 `TimerCondition` `never`/`any`/`<sender>` 来信早停，5 段式 Cron 支持抖动、`ExactTiming`、`IsDaemon` 与 `MaxIterations`）。`background_work_count` 包含所有仍能唤醒会话的后台命令和定时任务，打印模式只等待非守护任务。`background_work_snapshot` 省略原始命令行与提示词。
- **工具输出 10 KiB 中间截断与步骤文件落盘**：同步 `run_command`、后台 `run_command` 完成时的 `<SYSTEM_MESSAGE>` 结果正文，以及其余非 `read_file` 纯文本工具输出超过 `10,240` UTF-8 字节时，统一按官方格式保留首尾合法 UTF-8 切片与 `\n<truncated X of Y bytes from the middle of the output>\n` 标记，并将完整原始输出写入 `.system_generated/steps/<step_index>/output.txt`（同步与后台完成的 `run_command` 始终落盘步骤输出文件，`read_file` 保持独立的 800 行 / 46,080 字节限制）。
- **本地恢复扩展**：按会话保存待触发的 `schedule`。恢复时沿用任务 ID、下一次截止时间、取消条件及已触发次数。已过期的 cron 补发一次后按原表达式继续；已取消和完成的任务在关闭前即移除恢复记录。
- **子 Agent 与跨会话通信**：`define_subagent`（自定义蓝图与写/子 Agent 工具开关）、`invoke_subagent`（支持 `self` / `research` / `image-generator` 及自定义类型、`Workspace` 支持 `inherit` / `branch` 目录克隆 / `share` Git worktree 隔离、按类型构造专属系统提示词与 `<subagent_reminder>`）、`send_message`（父子及同级双向消息投递与唤醒、正常完成且已主动汇报时抑制重复空闲通知；若子 Agent 轮次异常失败或中断，或等待后台任务/孙 Agent 清零后进入空闲，均确保唤醒父会话）、`manage_subagents`（直属子 Agent 列表及递归终止清理派生工作区）；运行中的直属子 Agent 同步通过 `AgySubagentWorkTracker` 纳入父会话 `background_work_count` 与 `background_work_snapshot`。

### 2.6 上下文压缩、会话恢复与 CLI 模式

- **自动与手动压缩**：按模型阈值（Gemini 4 Neon/Carbon/Argon `512000`，其余官方模型 `256000`，支持环境变量 `AGY_CLI_AUTO_COMPACT_OVERRIDE` 整数或 `1%..100%` 覆盖）以 `promptTokenCount + candidatesTokenCount + thoughtsTokenCount` 锚点加尾部未采样增量估算在调用前自动触发压缩（对外 `usage.input_tokens` 按官方语义扣除 `cache_read_tokens`）；支持 `/compact` 与 `/compact <follow-up>` 手动压缩（附加 `ON-DEMAND COMPACTION` 后缀），压缩请求自动切换模型对应的 `compaction_max_output_tokens`（`32768` 或 `16384`）。若普通请求或压缩请求本身遇到 `ContextLengthExceeded`，自动回退至超长 `<SYSTEM_MESSAGE>` / 工具结果截断与最旧工具结果裁剪以完成恢复。
- **`<CONTEXT_SUMMARY>` 与恢复**：压缩后历史替换为标准 `<CONTEXT_SUMMARY>`（包含最近 10 条用户请求或父 Agent 消息、按 ID 排序的已创建子 Agent 摘要块及日志路径说明），同时写入 `CHECKPOINT` 步骤；`--continue` / `--conversation` 跨进程恢复历史后在下一轮用户消息后自动附带重启提醒 `<SYSTEM_MESSAGE>`。
- **CLI 输入输出模式**：支持 `-p` 非交互打印、交互式 REPL（`/model`、`/effort`、`/compact`、`/status`）、`models` / `agents` 子命令，以及 `--input-format stream-json` 与 `--output-format text|json|stream-json` 流式事件输出。
