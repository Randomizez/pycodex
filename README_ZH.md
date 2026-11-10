# pycodex

[English](README.md)

pycodex 用 Python 实现 [Codex](https://github.com/openai/codex) 编程 Agent，
可在终端、浏览器或自己的 Python 程序中使用，支持文件编辑、命令执行、
多 Agent 协作和会话恢复。支持范围及与上游的差异见[对齐文档](docs/ALIGNMENT.md)。

## 安装

| 包 | 内容 | Python |
| --- | --- | --- |
| `python-codex` | CLI 和 Python API | 3.6.2+ |
| `pycodex-ws` | Core、浏览器工作区和可选飞书集成 | 3.7+ |

使用 CLI 或 Python API：

```bash
python3 -m pip install --upgrade python-codex
```

使用浏览器工作区则安装 `pycodex-ws`，它会同时安装匹配版本的 core：

```bash
python3 -m pip install --upgrade pycodex-ws
```

## 配置模型

可从 [config_examples/](config_examples/README.md) 选择 Responses、StepFun、
Chat Completions、Claude Messages 或 Gemini 模板。向 Agent 提供 endpoint 和 key
（StepFun Coding Plan 已内置 endpoint），让它按目录说明安装配置或 profile 并验证连接。

创建 `~/.codex/config.toml`。以 OpenAI 为例：

```toml
model = "gpt-5.4"
model_provider = "openai"
```

API key 通过环境变量或 `~/.codex/.env` 提供：

```bash
export OPENAI_API_KEY="your-api-key"
```

其他 provider 需要配置 `base_url` 和 `env_key`，并在 profile 或 provider 中
用 `compat_api` 选择接口：

| `compat_api` | 接口 |
| --- | --- |
| `"none"`（默认） | Responses |
| `"chat_completions"` | Chat Completions，包括 Step |
| `"messages"` | Claude Messages |
| `"gemini"` | Gemini API |

旧配置 `use_chat_completion = true/false` 仍可用，分别对应 `chat_completions/none`。

用 `--profile NAME` 选择同目录的 `NAME.config.toml` 或内嵌的 `[profiles.NAME]`；
用 `--config PATH` 指定其他配置文件。`pycodex doctor` 可检查配置和连接。
浏览器工作区会为新会话动态读取 profile，修改配置后无需重启。

详见[配置文档](docs/CONTEXT.md#配置读取与-profile)和
[Gemini 配置示例](docs/responses_server/README.md#gemini-api)。
Gemini 4 Argon 内置基于 Antigravity 适配的提示词。
Gemini 工具参数支持流式接收，完整调用收到后再执行工具。

## 使用 CLI

在项目目录中运行：

```bash
pycodex
pycodex "概括这个仓库。"
pycodex --profile my_profile "审查最近的改动。"
pycodex --json "概括这个仓库。"
```

不带问题时进入交互会话，其他命令执行单次任务。
网页 skill 补全是 [optional feature](packages/pycodex-ws/workspace_server/optional_features/skill-completion/README.md)：
通过 Customize 应用 `skill-completion` 后启用 @/$ 名称前缀补全。
Agent 可以修改文件、执行命令，使用当前用户的系统权限。

| 命令 | 用途 |
| --- | --- |
| `/model`、`/effort` | 查看或切换模型及推理强度 |
| `/cd <directory>` | 空闲时切换目录 |
| `/title`、`/history` | 管理标题、查看历史 |
| `/resume` | 列出或恢复已保存会话 |
| `/compact` | 压缩历史，释放上下文 |
| `/queue <message>` | 等当前任务完成后再执行 |

Agent 工作时可直接发送消息调整方向，或用 `/queue` 排队。
会话默认保存在 `~/.codex/sessions`。
Ctrl+C 或 Ctrl+D 会在已发出的操作完成后退出。完整选项见 `pycodex --help`。

接入本地 vLLM 服务：

```bash
pycodex --vllm-endpoint http://127.0.0.1:18000
```

## 使用浏览器工作区

安装 `pycodex-ws` 后，创建 `workspaces.json`：

```json
{
  "workspaces": [
    {"id": "demo", "work_dir": ".", "state_path": "demo", "enable_board": true}
  ]
}
```

启动服务并保持终端打开：

```bash
pycodex-ws --listen 127.0.0.1:6007 --workspace-config workspaces.json
```

访问 `http://127.0.0.1:6007`。工作区沿用 CLI 的模型配置，提供对话、
实时 HTML Canvas 和文件/diff 查看器。点击文件链接查看代码，选中文字后
用 **Insert into chat** 加入下一条消息。

设置菜单支持界面定制和带密码的工作区分享。会话操作、存储和飞书配置见
[工作区指南](packages/pycodex-ws/README.md)。
独立安装的 [VS Code 扩展](packages/pycodex-ws/extensions/vscode/README.md)
可连接同一后端，也支持 Remote SSH。

## 使用 Python API

将以下代码保存为 `example.py`，运行 `python3 example.py`。
它会使用前面的模型配置和内置工具：

```python
from contextlib import closing

from pycodex import AgentsManager, CodexConfig
from pycodex.bootstrap import build_model
from pycodex.utils.async_bridge import run_async


async def main():
    config = CodexConfig.from_env()
    with closing(build_model(config.provider)) as model:
        runtime = AgentsManager(
            model.client, config.context,
        ).create_root_runtime()
        try:
            await runtime.start()
            receipt = await runtime.submit_input("概括这个仓库。")
            print((await receipt.future).output_text)
        finally:
            await runtime.close()


run_async(main())
```

`AgentsManager` 创建 runtime，`submit_input` 提交任务，等待 `receipt.future` 获取结果。
示例结束时会关闭 runtime 和本地兼容服务。
自定义模型、工具与集成方式见[运行时文档](docs/RUNTIME.md#ownership)。

## 更多功能

- [Portable 配置](docs/PORTABLE.md)：将配置、指令和 skills 加密迁移到其他机器。
- [界面定制](packages/pycodex-ws/README.md#customize-the-interface)：通过 Agent 对话或 CSS/JavaScript 修改工作区。
- [Responses 兼容服务](docs/responses_server/README.md)：通过 Responses API 接入支持的模型后端。
- [重试与备用模型](docs/RUNTIME.md#responses-reliability)：配置模型故障后的恢复策略。

## 开发

在仓库根目录运行：

```bash
env -u VIRTUAL_ENV uv sync --dev
env -u VIRTUAL_ENV uv run --dev pytest
```

参考[开发约定](AGENTS.md)、[测试指南](tests/TESTS.md)、
[运行时架构](docs/RUNTIME.md)和[工作区 API](docs/WORKSPACE_API.md)。
