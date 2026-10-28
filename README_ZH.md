# pycodex

[English](README.md)

pycodex 用 Python 实现 [Codex](https://github.com/openai/codex) 的 Agent 主循环，
提供 CLI、可复用的 Agent API 和浏览器工作区。项目以对齐上游行为为目标；
支持范围和刻意保留的差异见[对齐文档](docs/ALIGNMENT.md)。

| 包 | 内容 | Python |
| --- | --- | --- |
| `python-codex` | Agent、Runtime、CLI、工具、Responses 兼容服务和 portable 存储 | 3.6.2+ |
| `pycodex-ws` | 浏览器工作区、Canvas、文件/diff 查看器和可选飞书集成 | 3.7+ |

workspace 包会安装匹配版本的 core。Python 导入路径为 `pycodex`，
命令为 `pycodex` 和 `pycodex-ws`。

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

```bash
pycodex --profile my_profile "概括这个仓库。"
pycodex --config /path/to/config.toml
pycodex doctor
```

默认使用流式 Responses API。接入 Chat Completions 后端时，在对应 provider
配置中设置 `use_chat_completion = true`。其他后端入口包括：

```bash
pycodex --vllm-endpoint http://127.0.0.1:18000
pycodex --profile my_profile --use-messages
```

详见[配置与上下文](docs/CONTEXT.md)和
[Responses 兼容服务](docs/responses_server/README.md)。

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
| `/fork` | 保留当前历史，以新会话 ID 继续 |
| `/queue <message>` | 等当前 turn 完成后再执行 |

运行中提交普通输入，会在下一个安全采样边界插入。
斜杠输入遵循 Codex 的命令规则：`/data/project` 和 `/tmp/` 是普通消息；
`/modle` 或 `/tmp` 这样的未知单层名称会提示命令错误。
在输入开头加一个空格即可按文本发送，包括已知命令，例如 ` /model`。
这些规则同样适用于浏览器工作区和飞书。会话保存在
`CODEX_HOME/sessions`；配置 `model_auto_compact_token_limit` 可启用自动压缩。

Ctrl+D 或 Ctrl+C 发起退出，等待已接收任务完成；关闭等待期间再次 Ctrl+C
立即退出。完整启动参数见 `pycodex --help`。

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
把选中的代码插入可编辑草稿。运行期间保持服务器 stdin 打开，EOF 会触发关闭。
点击齿轮按钮可开启界面定制会话，让 Agent 修改 `~/.pycodex/frontend/` 中的文件。

会话操作、存储和飞书配置见[工作区说明](packages/pycodex-ws/README.md)。
独立分发的 [VS Code 扩展](packages/pycodex-ws/extensions/vscode/README.md)
复用同一后端，支持 Remote SSH 工作区。

## Python API

在 Python 中复用标准模型和工具装配：

```python
from pycodex.bootstrap import build_agent, build_model

agent = build_agent(build_model())
try:
    result = agent.ask("用一句话概括这个仓库。")
    print(result.output_text)
finally:
    agent.shutdown()
```

自定义集成时，`Agent` 接收 model client、`ToolRegistry` 和 `ContextConfig`。
通过 `await agent.run_turn([...])` 异步运行 turn；`AgentRuntime` 提供输入队列、
会话命令和前端订阅。详见[运行时契约与示例](docs/RUNTIME.md)。

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

发布记录和实现细节放在 `docs/`。

保存的会话会记住 `clock` 周期，服务重启或 `/resume` 后自动恢复并重新计时，
让周期任务能继续运行。`clock(null)` 的取消状态同样保存；停机期间错过的
定时触发不会集中补发。
