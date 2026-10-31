# pycodex

[English](README.md)

pycodex 用 Python 实现 [Codex](https://github.com/openai/codex) 的 Agent 主循环，
提供 CLI、可复用的 Agent API 和浏览器工作区。项目对齐选定功能、模型与协议在
上游默认配置下的行为；支持范围和刻意保留的差异见[对齐文档](docs/ALIGNMENT.md)。

本地约定不实现 sandbox、不恢复 collaboration mode 交互模式切换；
子 Agent 不使用 OpenAI 专属模型/effort 白名单，也不要求补齐所有 OpenAI 模型目录条目。

| 包 | 内容 | Python |
| --- | --- | --- |
| `python-codex` | Agent、Runtime、CLI、工具、Responses 兼容服务和 portable 存储 | 3.6.2+ |
| `pycodex-ws` | 浏览器工作区、Canvas、文件/diff 查看器和可选飞书集成 | 3.7+ |

workspace 包会安装匹配版本的 core。Python 导入路径为 `pycodex`，
命令为 `pycodex` 和 `pycodex-ws`。

`exec` 工具在宿主进程内执行支持顶层 `await` 的 Python。每个 cell 独立维护
namespace、输出和 asyncio 任务记录，导入模块和进程状态共享。停止采用协作式
取消；阻塞操作通过 `tools.exec_command` 运行，直接访问文件时可使用 cell 的
`cwd` Path。详见 [exec 生命周期与限制](docs/RUNTIME.md#python-exec-cells)。

## 安装与启动

安装核心 CLI：

```bash
python3 -m pip install python-codex
pycodex
```

也可以安装包含 core 的浏览器工作区：

```bash
python3 -m pip install pycodex-ws
pycodex-ws --help
```

使用 `python3 -m pip install --upgrade pycodex-ws` 一起升级两个包。
如果命令不在 `PATH` 中，用安装时的解释器运行 `python3 -m pycodex`
或 `python3 -m workspace_server`。

## 模型配置

默认读取 `~/.codex/config.toml` 和同目录的 `.env`。根据需要配置 `model`、
`model_provider` 及 provider 的 `base_url`、`env_key`。
内置 `openai` provider 使用 `OPENAI_API_KEY`。认证使用 API key，
不读取 `auth.json` 或系统钥匙串中的 ChatGPT 登录凭据。

`--profile NAME` 支持同目录的 `NAME.config.toml` 和内嵌的
`[profiles.NAME]`，两者均递归合并到基础配置，包括 provider 和 feature
等嵌套表。顶层 `profile = "NAME"` 设置默认选择，显式 `--profile` 优先。
两种格式同名或选中的 profile 缺失时会报错；现有内嵌配置无需迁移。

```bash
pycodex --profile my_profile "概括这个仓库。"
pycodex --config /path/to/config.toml
pycodex doctor
```

默认使用流式 Responses API；OpenAI 模型直接使用原生 Responses。
接入 vLLM Chat Completions 后端时，在对应 provider 配置中设置
`use_chat_completion = true`。其他后端入口包括：

```bash
pycodex --vllm-endpoint http://127.0.0.1:18000
pycodex --profile my_profile --use-messages
```

详见[配置与上下文](docs/CONTEXT.md)和
[Responses 兼容服务](docs/responses_server/README.md)。
兼容服务支持 namespace 内的 function/custom 工具（包括上游 Codex 的
子 Agent 工具），并在调用返回时恢复 namespace。
正文和 reasoning 流包含完整条目生命周期事件，支持 Codex 实时展示。

## CLI

不带 prompt 时进入交互会话，也可以直接运行单轮任务：

```bash
pycodex "用一句话概括这个仓库。"
printf 'Reply with exactly OK.' | pycodex
pycodex --json "Reply with exactly OK."
```

| 命令 | 用途 |
| --- | --- |
| `/model`、`/effort` | 查看或切换模型及推理强度 |
| `/cd <directory>` | 空闲时切换当前会话的工作目录 |
| `/title`、`/history` | 管理标题、查看历史 |
| `/resume` | 列出已保存会话；传入序号或 UUID 恢复 |
| `/compact` | 压缩历史，释放上下文空间 |
| `/queue <message>` | 等当前 turn 完成后再执行 |

运行中提交普通输入，会在下一个安全采样边界插入。
斜杠输入遵循 Codex 的命令规则：`/data/project` 和 `/tmp/` 是普通消息；
`/modle` 或 `/tmp` 这样的未知单层名称会提示命令错误。
在输入开头加一个空格即可按文本发送，包括已知命令，例如 ` /model`。
这些规则同样适用于浏览器工作区和飞书。会话保存在
`CODEX_HOME/sessions`；配置 `model_auto_compact_token_limit` 可启用自动压缩。

Ctrl+D 或 Ctrl+C 发起退出，同时停止本会话及所有子 Agent：清空待执行任务，
当前执行在已发出的模型请求和工具操作完成后停止。关闭 session 使用相同行为；
关闭等待期间再次 Ctrl+C 立即退出。完整启动参数见 `pycodex --help`。

## 浏览器工作区

创建 `workspaces.json`：

```json
{
  "workspaces": [
    {"id": "demo", "work_dir": ".", "state_path": "demo", "enable_board": true}
  ]
}
```

```bash
pycodex-ws --listen 127.0.0.1:6007 --workspace-config workspaces.json
```

访问 `http://127.0.0.1:6007`。工作区同时提供对话、实时 HTML Canvas 和只读
文件/diff 查看器。点击回复中的文件链接打开视图，通过 **Insert into chat**
把选中的代码插入可编辑草稿。文本和图片视图的单文件上限为 10 MiB。
上下文圆环可打开内置 Agent 树，显示包含 root 的 `n running · m failed` 汇总、
每个 Agent 的本轮耗时（`for x min`）和错误详情。
点击弹窗中的 root 或子 Agent，可打开其实时 transcript 标签页。
`diff:left..right` 比较两个本地文件，
两侧均支持绝对或相对路径。用 `[TODO](TODO.md)` 链接本地文件，
裸写的文件名保持纯文本。本地 PDF 链接使用浏览器内置阅读器预览，
并提供新标签页打开和下载入口。运行期间保持服务器 stdin 打开，EOF 会触发关闭。
点击 **New session** 按启动时的 `--profile`、配置默认 `profile`、基础配置的
优先级创建会话。右侧 **+** 按钮（Custom profile）可从同目录的
`*.config.toml` 文件和内嵌 `[profiles.NAME]` 中选择；
选择框中的 **Default** 沿用相同默认选择。会话保存 profile 名称，
重启后按同一名称恢复；已有会话的配置保持不变。增删 profile 后需重启服务器。
当前 Agent 工作时，空输入框从随机位置开始，每约 6 秒顺序轮换一条简短提示，
涵盖命令、界面定制、工作区分享、Agent 状态、transcript 和定时唤醒分隔线等可选功能。
可在 `workspace.html` 的 `updatePromptPlaceholder`
中编辑 `hints` 字符串列表。已有草稿不受影响。
点击齿轮按钮，在 **Customize Agent** 中开启界面定制会话，让 Agent 修改
`~/.pycodex/frontend/` 中的文件；**Share** 可通过链接和密码共享当前工作区。

会话操作、存储和飞书配置见[工作区说明](packages/pycodex-ws/README.md)。
独立分发的 [VS Code 扩展](packages/pycodex-ws/extensions/vscode/README.md)
复用同一后端，支持 Remote SSH 工作区。

## Python API

在 Python 中复用标准模型和工具装配：

```python
from pycodex import AgentsManager, ContextConfig
from pycodex.bootstrap import build_model
from pycodex.model import DEFAULT_CODEX_CONFIG_PATH
from pycodex.utils.async_bridge import run_async

async def main():
    runtime = AgentsManager(
        build_model(), ContextConfig.from_codex_config(DEFAULT_CODEX_CONFIG_PATH),
    ).create_root_runtime()
    await runtime.start()
    try:
        receipt = await runtime.submit_input("用一句话概括这个仓库。")
        print((await receipt.future).output_text)
    finally:
        await runtime.close()

run_async(main())
```

自定义集成时，`Agent` 必须接收 model client、`ToolRegistry`、`ContextConfig`
和 `session_id`。通过 `await agent.run_turn([...])` 异步运行 turn；
`AgentRuntime(agent, agents_manager)` 提供输入队列、会话命令和前端订阅。
`AgentsManager` 统一管理命名任务树，并提供标准构造入口。

当前原型使用 MultiAgentV2 工具：`spawn_agent`、`send_message`、
`followup_task`、`wait_agent`、`interrupt_agent` 和 `list_agents`。
子 Agent 通过 `fork_turns` 继承上下文；会话 `/fork` 命令已移除。
`features.multi_agent_v2.max_concurrent_threads_per_session` 设置总容量，
默认 16（包含根 Agent）。满额时关闭最久未使用且没有待处理或后台工作的
空闲叶子节点；没有可回收节点则报容量不足。已关闭子节点从 Agent 树、
Workspace 和 `list_agents` 中移除，原任务名可重新使用。
已记录会话将 clock 周期保存在 rollout 旁的小型环境快照中；恢复后重新计时，
子 Agent 仍仅保存在内存。
详见[运行时契约](docs/RUNTIME.md)、[上下文装配](docs/CONTEXT.md)和
[新版对齐基线与剩余差异](docs/ALIGNMENT.md)。

## Portable 配置

Portable mode 将配置、指令和 skills 加密打包，供其他机器恢复使用。
先在单独的终端启动存储服务：

```bash
python3 -m pycodex.portable.server --root .tmp/pycodex_storage --host 127.0.0.1 --port 5577
```

上传默认 Codex home，或指定其他目录：

```bash
pycodex --put @127.0.0.1:5577
pycodex --put /path/to/.codex/@127.0.0.1:5577
```

上传并验证下载成功后，命令会打印可复用的 `pycodex --call ...` 启动命令。
在目标机器运行时，使用该机器能够访问的存储服务地址。

## 开发与参考

```bash
uv sync --dev
uv run pycodex
uv run --dev pytest
```

- [测试环境与覆盖范围](tests/TESTS.md)
- [仓库开发约定与阅读导航](AGENTS.md)
- [架构、包布局与运行时契约](docs/RUNTIME.md)
- [提示、配置与上下文](docs/CONTEXT.md)
- [上游对齐](docs/ALIGNMENT.md)
- [Responses 兼容服务](docs/responses_server/README.md)
- [Portable 配置契约](docs/PORTABLE.md)

相对 0.5.0 的主要改动和 Python 接口迁移见
[0.6.0 发布说明](https://github.com/Randomizez/pycodex/blob/v0.6.0/docs/RELEASE_0.6.0.md)，实现细节放在 `docs/`。
