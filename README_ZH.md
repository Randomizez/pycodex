# pycodex

中文 README。English version: `README.md`

0.3.0 发版准备与 Python API 迁移说明：`docs/RELEASE_0.3.0.md`。

PyPI distributions：

- 主包：`python-codex`
- Workspace 安装别名：`pycodex-ws`

Import 路径仍保持 `pycodex`；CLI 命令为 `pycodex` 和 `pycodex-ws`。

`pycodex-ws` 是一个薄 metapackage，精确依赖同版本的 `python-codex`。
实际代码和 console script 继续由 `python-codex` 提供，两个 distribution
不会重复安装同名模块。

这个仓库把上游 Codex（`https://github.com/openai/codex`）里最核心的 agent
闭环先抽成一个尽量小的 Python 版本，重点保留两层结构：

- `submission_loop`：顺序消费提交的操作。
- `run_turn`：在单个 turn 内持续执行 `模型采样 -> 工具调用 -> 把工具结果喂回模型`，直到拿到最终回答。

对应的 Rust 参考点：

- `codex-rs/core/src/codex.rs` 里的 `submission_loop`
- `codex-rs/core/src/codex.rs` 里的 `run_turn`
- `codex-rs/core/src/codex.rs` 里的 `run_sampling_request`
- `codex-rs/core/src/tools/router.rs` 里的 `ToolRouter`
- `codex-rs/core/src/stream_events_utils.rs` 里的 `handle_output_item_done`

## 快速开始

安装完整包或 workspace 安装别名：

```bash
pip install python-codex
pip install pycodex-ws
```

先安装开发依赖：

```bash
uv sync
```

试一下真实入口：

```bash
uv run pycodex "Reply with exactly OK."
uv run pycodex
```

## 设计取舍

这里不是对 Rust 版本做 1:1 移植，而是先收敛到一个最小可复用内核：

1. 用一个很薄的 `ModelClient` 协议抽象模型侧。
2. 用 `ToolRegistry` 管理工具规格和执行器。
3. 用 `AgentLoop` 实现核心闭环。
4. 用 `AgentRuntime` 保留外层提交队列，方便以后继续对齐 Rust 的 `submission_loop`。

暂时刻意不包含：

- TUI / 流式增量渲染
- MCP / connectors / sandbox / approvals
- memory / compact / review mode
- 真实 OpenAI 适配器

这些都可以后续继续往上叠，但当前项目先把最核心的“工具增强推理主循环”钉住。

## 目录

- `pycodex/protocol.py`：最小的会话 item / prompt 协议
- `pycodex/events.py`：类型化事件、纯文本视图及有状态展示逻辑
- `pycodex/utils/event_helpers.py`：无状态的文本、颜色、摘要及结果格式化 helpers
- `pycodex/utils/image_utils.py`：图片加载、缩放和 data URL 转换
- `pycodex/model.py`：模型客户端协议和 Responses API 适配器
- `pycodex/cli.py`：单轮/交互命令行入口、终端 I/O 执行器和输入循环
- `pycodex/bootstrap.py`：与前端无关的模型、工具、Agent 和会话装配
- `pycodex/tools/base_tool.py`：`BaseTool`、`ToolRegistry`、`ToolContext`
- `pycodex/tools/`：具体工具实现
- `pycodex/agent.py`：主循环
- `pycodex/runtime.py`：统一会话命令、提交队列和前端事件订阅
- `tests/test_agent.py`：核心行为测试

## 当前对齐状态

当前前后端边界为 `CLI / Web / 飞书 -> AgentRuntime -> Agent`，不再额外套 Session。
所有会话命令、steer/queue 策略、模型/历史/标题状态、交互问答和权限响应、
worker 启停及连接清理都由后端统一处理，前端只做输入和展示适配。
Web 不再导入或运行 CLI shell，飞书运行中也能提交 steer 和回答问题。
IPython 是明确的例外：`ipython_agent()` 仍返回裸 Agent。

外部程序使用 `pycodex.bootstrap.build_model/build_agent/build_runtime` 装配，
通过 `await runtime.start()`、`runtime.attach(handler)`、
`await runtime.submit_input(text, sender)` 和 `await receipt.future` 交互。
`detach` 只移除一个前端，`close` 才停止接纳并自然排空会话。
队列统一通过 `start()` / `close()` 启停；重复或并发关闭复用 worker 的完成结果和异常，
不再入队特殊关闭请求。停止接纳后，worker 排空已有请求便退出，最后统一清理工具。
后台唤醒也遵守同一个接纳开关；取消关闭调用方不会取消 worker 或 Agent turn。
Agent 不保存 Runtime/Queue 引用，也不取输入或完成提交回执。steer 由 Runtime 调用
`agent.stop_asap()`：当前请求和已发出的工具先完成并记录，在安全边界抛出
`TurnInterrupted`，随后 Runtime 结算旧回执、启动下一批 `run_turn`。这不是 Task
取消；运行中的 steer 保留逻辑 turn id，但每次执行重新计数。裸 Agent 调用方直接
收到停止异常，待处理输入只有 Runtime worker 才能执行。
`/fork` 保留历史、分配新 session id 和延迟创建的新 rollout，不改写旧文件。
工作区标签页会在新 rollout 落盘后更新恢复路径；尚未落盘时保存源记录和待 fork
标记，重启后从源记录重新分叉，保留历史与标题，后续续写不会写回源文件。
完整契约见 `docs/RUNTIME.md`。

Python 回调统一接收 `pycodex.events` 中的具体 `Event` dataclass，
例如 `TurnStartedEvent` / `ToolCompletedEvent`；自定义模型客户端发送
`AssistantDeltaEvent` 等 `ModelEvent`。不再使用通用事件 payload 字典，
事件通过 `event.visualize()` 生成可复用的纯文本，通过 `event.render(display)`
执行有状态展示。每个 CLI 实例和飞书卡片独占 `EventDisplay`，事件体系负责着色、
流缓冲、状态和输入提示更新；前端只执行 log/status/prompt 回调及展示刷新，不维护
展示 handler mapping。飞书关闭颜色，展示有长度上限的近期记录和当前流式文本，
不再单独维护上一轮答案、工具、重试和问答的展示规则。Web 保留独立投影，
事件 JSON 格式不变。

当前进度可以分成两层看：

- prompt/context 对齐：
  - 2026-09-23 使用 Codex CLI 0.153.4 重抓：`gpt-5.4` 首轮、resume、
    工具 follow-up，以及带后缀 Astra 的首轮/resume，共享 context 一致；
  - 元数据按最长 slug 前缀匹配，wire model 不改名；提示和回放 metadata
    已更新，比较的明确排除项见 `docs/CONTEXT.md`。
- turn-loop 语义对齐：
  - `AgentLoop` 默认不再使用固定 12 轮上限；
  - 现在和上游一样，按 “还有没有 follow-up / tool handoff” 自然收敛；
  - 本地不再保留额外的 iteration-limit 参数。
- request-level 对齐：
  - **不宣称原始 request 严格相同**：客户端 item ID、工具目录、上游新版
    遥测和部分 permission profile 仍有差异；
  - 默认 CLI 继续使用 `codex-tui` 身份，但按用户要求移除了协作模式指令；
  - `tests/compare_context_requests.py` 分开报告原始差异和共享 context
    排除项，旧版 interactive 抓包不代表当前版本已经重新验证。
- 结构化问答：
  - `request_user_input` 保留工具声明，但调用固定返回
    `request_user_input is unavailable in Default mode`，不弹出问题；
  - 接入前端或注册交互 handler 不会启用它；通用 runtime 问答及权限服务保留；
  - `success` 只保留为本地执行状态，不发送给 API；旧 rollout 中的答案恢复后
    也遵守这条序列化规则。

协作模式的配置字段、提示模板和门控均已删除；普通 CLI/Web 交互、`update_plan`
和子 Agent 保留，不依赖协作模式。

更细的对齐说明见 `docs/ALIGNMENT.md`。

## 真实模型联调

如果本机已经有 Codex CLI 配置，可直接复用 `~/.codex/config.toml` 里的
`model`、`model_provider`、`base_url`、`env_key`：

```python
from pycodex import ResponsesModelClient

client = ResponsesModelClient.from_codex_config()
```

当前实现走 OpenAI-compatible Responses API 的流式 `/responses` 接口。这个点
已经用本机 `~/.codex/config.toml` 做过联调验证。

通过 CLI 启动时，`pycodex` 还会在读取配置前加载同目录下的 `.env`
（通常是 `~/.codex/.env`），方便把 provider key 之类的环境变量放在那里。
为对齐上游 Codex，`.env` 中以 `CODEX_` 开头的变量不会被导入。

## pycodex

`pycodex` 现在默认是一个最小交互式入口，内部通过 `AgentRuntime` 驱动 turn
提交循环，默认直接复用 `~/.codex/config.toml`：

```bash
pycodex
pycodex "Summarize this repo in one sentence."
printf 'Reply with exactly OK.' | pycodex
pycodex --json "Reply with exactly OK."
pycodex --profile model_proxy "Reply with exactly OK."
pycodex --vllm-endpoint http://127.0.0.1:18000 "Reply with exactly OK."
pycodex --put @127.0.0.1:5577
pycodex --put /data/.codex/@127.0.0.1:5577
pycodex --call SECRET-CALLID@127.0.0.1:5577 "只回复 OK。"
pycodex doctor
```

当前行为：

- 没有 argv prompt 且 stdin 是 TTY 时，进入交互模式
- 有 argv prompt 或 stdin 管道输入时，执行单轮请求
- 交互模式通过 `/exit`、`/quit`、空提示符上的 Ctrl+D 或一次 Ctrl+C 正常退出；
  `[closing]` 提示表示正在等待已接收的工作结束并清理资源，不取消模型或工具调用。
- 交互模式下会显示简洁阶段事件流，例如工具执行状态和模型回看工具结果
- assistant 文本会按流式 delta 直接打印
- 交互模式下支持 `/history`、`/title`、`/model` 和 `/resume`
- `/model <name>` 会切换当前交互会话后续请求使用的模型；`/model` 会显示当前模型和可选模型
- `/resume` 不带参数时会按首条用户消息预览列出当前可恢复的 session；`/resume 1`
  会恢复列表里的第 1 个 session
- `/resume <数字>` 会从 `CODEX_HOME/sessions` 读取选中的已记录 Codex rollout，
  并直接替换当前内存里的会话 history
- 新 session 现在会自动保存到 `CODEX_HOME/sessions/.../rollout-*.jsonl`，
  使用稳定的 session/thread id，并按 item 级别 append + flush，和 `/resume`
  读取的 rollout 格式保持一致
- 如果 workspace 根目录存在非空的 `TURN_HOOK.md`，每个已完成 turn 之后都会把
  刚结束的 history fork 成一个不落盘的临时 follow-up 会话，并把文件内容作为下一条
  user 指令提交；适合做 Feishu 通知这类副作用收尾动作
- 交互模式默认支持 steer：普通输入会走 runtime 的 steer 路径，当前请求会在下一个安全边界尽快停下，后续 steer 文本会按顺序并入下一次模型请求的 `input`；如需明确排队可用 `/queue <message>`，会打印 `[steer] queued: ...`，随后等该 turn 真正开始时再打印 `[steer] inserted: ...`
- 当前默认工具集由上游对齐子集和 pycodex 的 `clock` 扩展组成：`shell`、`shell_command`、`exec_command`、`write_stdin`、`clock`、`exec`、`wait`、`web_search`、`update_plan`、`request_user_input`、`request_permissions`、`spawn_agent`、`send_input`、`resume_agent`、`wait_agent`、`close_agent`、`apply_patch`、`grep_files`、`read_file`、`list_dir`、`view_image`
- `clock(period_m)` 为当前 Agent session 设置一个周期计时器，传 `null` 取消；每次回复后重新计时，到期后用包含当前时区时间的 `<clock_tick>` 消息唤醒 Agent
- 后台命令或 clock 正在等待时，空闲状态统一显示为 `idle: sleeping`
- workspace 只在当前活动 tab 上显示关闭按钮
- `--vllm-endpoint http://host:port` 会自动拉起一个本地 `responses_server` compat 层；当 path 为空时会内部补 `/v1`，继续把 `/responses` 请求转到下游 `/v1/chat/completions`。这条本地 compat 路径始终使用标准 Responses request shape，即使所选模型的 metadata 打开了 `responses_lite` 也不会切换成 lite wire format；启动时还会读取下游 `/v1/models`，自动使用返回列表中的最后一个 model id。当前对 `model_provider = "vllm"` 已补上 reasoning 兼容：会把 chat chunk 里的 `reasoning` / `reasoning_content` 翻回 Responses `reasoning` item，并把历史里的 `reasoning` item 回放成下游 assistant message 的 `reasoning` 字段；同时会向 vLLM 请求 streaming usage，并在最终 `response.completed.response.usage` 中回传
- `pycodex doctor` 会检查配置、`.env`、API key、DNS、TCP/TLS，以及可选的 live Responses API 请求

它目前主要用于：

- 验证 provider / model / auth 配置是否可用
- 调试 `ResponsesModelClient`
- 做最小单轮 / 多轮 smoke test

`doctor` 示例：

```bash
pycodex doctor
pycodex doctor --skip-live
pycodex doctor --json
```

## Portable Mode

`Portable Mode` 适合在新机器、新容器或新的调试镜像里，快速带起你平时使用的
`pycodex` 配置。

常见用法：

```bash
pycodex --put @127.0.0.1:5577
pycodex --put /data/.codex/@127.0.0.1:5577
uv run pycodex --call SECRET-CALLID@127.0.0.1:5577
uv run pycodex --call SECRET-CALLID@127.0.0.1:5577 "检查当前工作区并解释为什么启动失败。"
```

- `--put` 会打印一个可复用的 `SECRET-CALLID@host:port`，以及最终可直接执行的
  `pycodex --call ...` 命令
- 到新环境后，直接执行那条 `--call` 命令即可开始使用
- 这个模式适合快速恢复 `config.toml`、`.env`、`AGENTS.md` 和 `skills/`
- `--put /path/.codex/@host:port` 可以发布另一套 Codex home

## 示例

```python
import asyncio

from pycodex import (
    AgentLoop,
    BaseTool,
    ContextManager,
    ResponsesModelClient,
    ToolRegistry,
)


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo the provided text."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def run(self, context, args):
        del context
        return args["text"]


async def main() -> None:
    model = ResponsesModelClient.from_codex_config()
    context_manager = ContextManager.from_codex_config()

    tools = ToolRegistry()
    tools.register(EchoTool())

    agent = AgentLoop(model, tools, context_manager)
    result = await agent.run_turn("Call the echo tool with text=hello, then tell me what it returned.")
    print(result.output_text)


asyncio.run(main())
```

## 对齐清单

更细的说明见 `docs/ALIGNMENT.md`。这里保留一个高层 checklist，方便直接看当前进度。

### Tools 对齐

上游官方工具：

- [x] `shell` — 执行 argv 形式的 shell 命令。
- [x] `shell_command` — 执行字符串形式的 shell script。
- [x] `exec_command` — 启动带 session 的长命令执行。
- [x] `write_stdin` — 向已有执行 session 写入 stdin 或轮询输出。
- [x] `web_search` — 暴露 provider-native 的网页搜索能力。
- [x] `update_plan` — 更新任务计划并维护步骤状态。
- [x] `request_user_input` — 保留工具声明，返回上游 Default mode 的不可用提示。
- [x] `request_permissions` — 请求额外权限再继续执行。
- [x] `spawn_agent` — 创建并启动子 agent。
- [x] `send_input` — 给已有子 agent 继续发送输入。
- [x] `resume_agent` — 恢复已关闭的子 agent。
- [x] `wait_agent` — 等待子 agent 进入终态。
- [x] `close_agent` — 关闭不再需要的子 agent。
- [x] `apply_patch` — 用 freeform patch 精确修改文件。
- [x] `grep_files` — 按模式搜索文件内容。
- [x] `read_file` — 读取文件片段并保留行号语义。
- [x] `list_dir` — 列出目录树片段。
- [x] `view_image` — 把本地图片转成模型可见输入。

尚未单独建模的上游官方低频 / 特殊模式工具：

- [ ] `wait_infinite` — 长时间阻塞等待外部事件或后续输入。
- [ ] `spawn_agents_on_csv` — 按 CSV 批量创建子 agent 任务。
- [ ] `report_agent_job_result` — 上报批处理 agent job 的结果。
- [ ] `js_repl` — JavaScript REPL / code-mode 主入口。
- [ ] `js_repl_reset` — 重置 `js_repl` 的运行状态。
- [ ] `artifacts` — 生成或管理结构化工件输出。
- [ ] `list_mcp_resources` — 列出 MCP 资源。
- [ ] `list_mcp_resource_templates` — 列出 MCP 资源模板。
- [ ] `read_mcp_resource` — 读取 MCP 资源内容。
- [ ] `multi_tool_use.parallel` — 并行包装多个 developer tools 调用。

本仓库额外兼容层 / 过渡工具：

- [x] `clock` — pycodex 的周期性 Agent 唤醒扩展。
- [x] `exec` — 当前对 code-mode 的本地近似实现。
- [x] `wait` — 当前对 code-mode 等待行为的本地近似实现。

### 行为对齐

- [x] `AgentLoop` / `AgentRuntime` 主循环骨架 — turn 闭环和提交队列已成立。
- [x] 非交互 `exec` 路径的 `instructions` 对齐 — base instructions 已对齐上游。
- [x] 非交互 `exec` 路径的 `input` 对齐 — prompt input 已对齐上游。
- [x] developer/contextual-user message 的 shape 对齐 — message/content 结构已对齐。
- [x] `AGENTS.md` + `<environment_context>` 注入逻辑对齐 — 上下文拼接顺序已对齐。
- [x] 非交互 `exec` 路径的上游工具子集对齐 — 对齐子集已收敛；pycodex 额外暴露 `clock`。
- [x] `include = ["reasoning.encrypted_content"]` — reasoning include 字段已对齐。
- [x] `prompt_cache_key` — 请求级 prompt cache key 已补齐。
- [x] `x-client-request-id` — 请求 id header 已补齐。
- [x] `x-codex-turn-metadata` — turn id / sandbox header 已补齐。
- [x] `originator` — mode-aware originator header 已补齐。
- [x] `user-agent` 精确字符串对齐 — 非交互 `exec` 路径已对齐上游字符串。
- [x] 上游 exec-mode tool schema 的逐字段对齐 — 对齐工具使用类内 spec；`clock` 作为扩展单独记录。
- [ ] 交互模式与非 `exec` 路径的完整行为对齐 — non-exec 首轮 context 已切到 `codex-tui` 路径，但 REPL 连续多轮行为还未完全验证。
- [ ] sandbox / approvals / compact / memory 等外围行为对齐 — 外围系统仍在后续范围。
