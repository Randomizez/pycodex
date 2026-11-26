# pycodex

[English](README.md) · [0.7.4 发布说明（英文）](https://github.com/Randomizez/pycodex/blob/v0.7.4/docs/RELEASE_0.7.4.md)

用 Python 实现的 [Codex](https://github.com/openai/codex) 编程 Agent，自带浏览器工作区、
实时 HTML Canvas、多模型/多运行时支持，以及完整的 CLI 与 Python SDK。

## 快速开始

```bash
python3 -m pip install --upgrade pycodex-ws
pycodex-ws
```

在浏览器打开 `http://127.0.0.1:6007`，点击 **Create workspace**，后续操作全部在前端完成。

- **已经在用 Codex CLI？** 零配置直接开跑——`pycodex-ws` 会自动继承
  `~/.codex/config.toml`、`~/.codex/.env`、`AGENTS.md`、skills 和 `~/.codex/sessions`。
- **还没有 Codex 配置？** 在 `~/.codex/config.toml` 写入 `model = "gpt-5.4"` 并设置
  `OPENAI_API_KEY`（环境变量或 `~/.codex/.env`）；或直接把
  [config_examples/](config_examples/README.md) 连同你的 endpoint 和 key 交给任意 Agent 帮你配好。
- **偏好纯终端？** 在项目目录下运行 `pycodex` 即可（仅需 CLI / Python API 且兼容 Python 3.6.2+
  时可只装 `python-codex`）。

## 特色功能

- **无痛继承 Codex 生态** — 共用 `~/.codex` 配置、`AGENTS.md`、skills 与会话历史；支持运行中直接
  steer 纠偏、`/queue` 排队、多子 Agent 协作与自动压缩。
- **多模型协议与多运行时** — 支持 OpenAI Responses、Chat Completions（StepFun、vLLM、各类网关）、
  Claude Messages 和 Gemini API，并内置 [`pyagy`](packages/pyagy/README.md)（Antigravity CLI 协议兼容）
  运行时；浏览器内新建会话时可随时切换 profile，改配置无需重启。
- **实时 HTML Canvas 与代码/Diff 查看器** — 对话旁实时渲染 HTML 看板、图表与报告；内置源码、
  Markdown、图片和 Git diff 查看器，选中代码一键 **Insert into chat**，加 `--password` 还可启用内置 VS Code。
- **用 Agent 定制前端（`Customize Agent`）** — 点击 **Settings → Customize Agent**，通过对话
  一键安装可选特性（如 `@/$` skill 补全、文件上传、语音输入）或魔改界面 CSS/JS，随时可用 `?builtin=1` 回退。
- **多机统一 Router 与跨节点借用 Profile** — 多台机器通过 `pycodex-router` 汇聚到同一浏览器目录，
  支持带密码分享工作区、**Peer Profiles**（工具和文件跑在本地，模型请求经 router 代理调用另一台已验证设备的
  profile，API key 不离机），以及网页端 **Settings → Update & reboot** 原地升级重启。
- **External Call、飞书卡片与加密配置迁移** — 支持把远端终端接入 Web 会话（**External Call**）、
  通过飞书卡片 `/link` 在手机上随时交互，以及用 `pycodex --put` / `--call` 加密迁移整套配置与 skills。

## 进阶配置交给 Agent

不需要手写复杂 JSON 或记参数，启动后直接在浏览器对话里让 Agent 帮你完成：

- **接入其他模型 / 新增 Profile（Claude、Gemini、StepFun、`pyagy`、vLLM）**：
  > 请按 `config_examples/README.md` 用我的 endpoint 和 API key 新增一个 profile，并用 `pycodex doctor` 验证可用。
- **定制工作区界面 / 安装扩展功能**：
  点开 **Settings → Customize Agent**，浏览内置功能目录，直接告诉 Agent 你想开启或修改什么。
- **配置多机 Router、密码分享、External Call 或飞书接入**：
  > 请参考 `packages/pycodex-ws/README.md`，帮我配置好 `pycodex-router` / 飞书 `/link` / 工作区分享。

## CLI 与 Python API

```bash
pycodex                       # 进入交互式终端会话
pycodex "概括这个仓库。"        # 执行单次任务
pycodex doctor                # 检查配置与连通性
```

<details>
<summary>最小 Python API 示例</summary>

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

</details>

## 包结构与文档

| 包 | 内容 | Python |
| --- | --- | --- |
| `pycodex-ws` | 浏览器工作区、Router、可选飞书集成，自动包含匹配版本的 `python-codex` 与 `pyagy`（[指南](packages/pycodex-ws/README.md)） | 3.7+ |
| `pyagy` | Core + Antigravity CLI（`agy`）协议兼容 harness（[指南](packages/pyagy/README.md)） | 3.7+ |
| `python-codex` | 核心 CLI、Python API 与 Responses 兼容服务 | 3.6.2+ |

- 配置与使用：[配置模板](config_examples/README.md) · [配置与 profile](docs/CONTEXT.md#配置读取与-profile) · [Responses 兼容服务](docs/responses_server/README.md) · [Portable 迁移](docs/PORTABLE.md)
- 架构与开发：[上游对齐说明](docs/ALIGNMENT.md) · [运行时架构](docs/RUNTIME.md) · [工作区 API](docs/WORKSPACE_API.md) · [开发约定](AGENTS.md) · [测试指南](tests/TESTS.md)

```bash
env -u VIRTUAL_ENV uv sync --dev
env -u VIRTUAL_ENV uv run --dev pytest
```
