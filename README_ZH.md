# pycodex

中文 README。English version: `README.md`

PyPI distribution name: `python-codex`  
Import path and CLI command remain `pycodex`.

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
- memory / compact / hooks / review mode
- 真实 OpenAI 适配器

这些都可以后续继续往上叠，但当前项目先把最核心的“工具增强推理主循环”钉住。

## 目录

- `pycodex/protocol.py`：最小的会话 item / prompt / event 协议
- `pycodex/model.py`：模型客户端协议和 Responses API 适配器
- `pycodex/cli.py`：`pycodex` 单轮命令行入口
- `pycodex/tools/base_tool.py`：`BaseTool`、`ToolRegistry`、`ToolContext`
- `pycodex/tools/`：具体工具实现
- `pycodex/agent.py`：主循环
- `pycodex/runtime.py`：外层提交队列
- `tests/test_agent.py`：核心行为测试

## 当前对齐状态

当前进度可以分成两层看：

- prompt/context 对齐：
  - 非交互 `exec` 路径下，`instructions` 和 `input` 已经对齐到上游 Codex；
  - 这一层现在主要由 `pycodex/context.py` 和 vendored prompt data 负责。
- turn-loop 语义对齐：
  - `AgentLoop` 默认不再使用固定 12 轮上限；
  - 现在和上游一样，按 “还有没有 follow-up / tool handoff” 自然收敛；
  - 本地不再保留额外的 iteration-limit 参数。
- request-level 对齐：
  - 非交互 `exec` 路径的 request body 已基本对齐；
  - 默认 CLI 的 non-exec 首轮请求现在也已切到 `codex-tui` + `<collaboration_mode>` 这条上游路径；
  - 默认 CLI 的两轮主线程对话 request/header 也已补抓并对齐，包括后续 turn 不再携带 `workspaces`；
  - 当前剩余重点主要转向更外围的行为分支，而不是这条已比较路径上的 request/header。
- tool round-trip 对齐：
  - `request_user_input` 的 Default-mode unavailable 路径已按真实 upstream capture 对齐；
  - Plan-mode happy path 现在也已按 upstream 源码补齐到工具/协议层：会强制 `isOther=true`、要求非空 `options`，并以 JSON 字符串 + `success=true` 回传结构化答案；
  - 新增了一个基于 `tests/fake_responses_server.py` proxy 模式的 deterministic round-trip compare 脚本 `tests/compare_request_user_input_roundtrip.py`；它在本机已安装的 `codex-cli 0.115.0` 上确认：Plan-mode live capture 里唯一剩余的 `function_call_output` schema 差异是 `pycodex` 多带了 `success=true`。

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
pycodex doctor
```

当前行为：

- 没有 argv prompt 且 stdin 是 TTY 时，进入交互模式
- 有 argv prompt 或 stdin 管道输入时，执行单轮请求
- 交互模式下支持 `/exit` 和 `/quit`
- 交互模式下会显示简洁阶段事件流，例如工具执行状态和模型回看工具结果
- assistant 文本会按流式 delta 直接打印
- 交互模式下支持 `/history`、`/title` 和 `/model`
- `/model <name>` 会切换当前交互会话后续请求使用的模型；`/model` 会显示当前模型和可选模型
- 交互模式默认支持 steer：普通输入会走 runtime 的 steer 路径，当前请求会在下一个安全边界尽快停下，后续 steer 文本会按顺序并入下一次模型请求的 `input`；如需明确排队可用 `/queue <message>`，会打印 `[steer] queued: ...`，随后等该 turn 真正开始时再打印 `[steer] inserted: ...`
- 当前默认注册一组与原版 Codex 一一对应的本地工具子集：`shell`、`shell_command`、`exec_command`、`write_stdin`、`exec`、`wait`、`web_search`、`update_plan`、`request_user_input`、`request_permissions`、`spawn_agent`、`send_input`、`resume_agent`、`wait_agent`、`close_agent`、`apply_patch`、`grep_files`、`read_file`、`list_dir`、`view_image`
- `--vllm-endpoint http://host:port` 会自动拉起一个本地 `responses_server` compat 层；当 path 为空时会内部补 `/v1`，继续把 `/responses` 请求转到下游 `/v1/chat/completions`。当前对 `model_provider = "vllm"` 已补上 reasoning 兼容：会把 chat chunk 里的 `reasoning` / `reasoning_content` 翻回 Responses `reasoning` item，并把历史里的 `reasoning` item 回放成下游 assistant message 的 `reasoning` 字段
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
- [x] `request_user_input` — 向用户发起结构化问题并等待回答。
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

- [x] `exec` — 当前对 code-mode 的本地近似实现。
- [x] `wait` — 当前对 code-mode 等待行为的本地近似实现。

### 行为对齐

- [x] `AgentLoop` / `AgentRuntime` 主循环骨架 — turn 闭环和提交队列已成立。
- [x] 非交互 `exec` 路径的 `instructions` 对齐 — base instructions 已对齐上游。
- [x] 非交互 `exec` 路径的 `input` 对齐 — prompt input 已对齐上游。
- [x] developer/contextual-user message 的 shape 对齐 — message/content 结构已对齐。
- [x] `AGENTS.md` + `<environment_context>` 注入逻辑对齐 — 上下文拼接顺序已对齐。
- [x] 非交互 `exec` 路径的工具子集对齐 — 暴露给模型的工具集合已收敛。
- [x] `include = ["reasoning.encrypted_content"]` — reasoning include 字段已对齐。
- [x] `prompt_cache_key` — 请求级 prompt cache key 已补齐。
- [x] `x-client-request-id` — 请求 id header 已补齐。
- [x] `x-codex-turn-metadata` — turn id / sandbox header 已补齐。
- [x] `originator` — mode-aware originator header 已补齐。
- [x] `user-agent` 精确字符串对齐 — 非交互 `exec` 路径已对齐上游字符串。
- [x] exec-mode tool schema 的逐字段对齐 — 当前通过工具层直接复用上游 snapshot。
- [ ] 交互模式与非 `exec` 路径的完整行为对齐 — non-exec 首轮 context 已切到 `codex-tui` 路径，但 REPL 连续多轮行为还未完全验证。
- [ ] sandbox / approvals / compact / memory 等外围行为对齐 — 外围系统仍在后续范围。
