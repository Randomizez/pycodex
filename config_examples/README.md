# 配置模板

这些文件可直接作为 `config.toml`，也可放在已有配置旁作为同名 profile。
将本页交给 Agent，提供要用的服务和 endpoint、API key，由 Agent 完成模型核对、
配置安装和连接验证。StepFun Coding Plan 已内置 endpoint，只需提供该服务的 key。

例如：

> 请按 config_examples/README.md 配置 Gemini。endpoint 是 ……，key 已放在
> GEMINI_API_KEY 环境变量中。请新增 gemini profile，验证后给我启动命令。

| 模板 | 接口 | 示例模型 | Key 环境变量 |
| --- | --- | --- | --- |
| [responses.config.toml](responses.config.toml) | 原生 Responses | `gpt-6-astra` | `RESPONSES_API_KEY` |
| [stepfun.config.toml](stepfun.config.toml) | StepFun Coding Plan，Chat Completions | `step-5-preview` | `STEP_API_KEY` |
| [chat-completions.config.toml](chat-completions.config.toml) | 通用 Chat Completions / vLLM | `step-3.5-flash` | `CHAT_API_KEY` |
| [claude.config.toml](claude.config.toml) | Claude Messages，Bearer 网关 | `claude-opus-5-5` | `CLAUDE_API_KEY` |
| [gemini.config.toml](gemini.config.toml) | Gemini 原生 API | `gemini-4-argon-high` | `GEMINI_API_KEY` |

`api.example.invalid` 是需要替换的地址，不会指向真实服务。模型名是已支持模型的
示例，最终须使用目标 endpoint 实际提供的完整 ID，包括 provider 前缀或路由后缀。
StepFun 模板针对 `https://api.stepfun-inc.com/step_plan/v1`，其他 StepFun 产品的
key 和 endpoint 不应混用；使用其他产品时由 Agent 按该服务说明调整。

## Agent 配置流程

1. 读取用户选定的配置文件（默认 `~/.codex/config.toml`）和已有 profile。
   选择上表中与服务协议匹配的模板，不根据模型名猜测协议。模板中的
   `compat_api` 已明确设置；不要改成旧的 `use_messages` / `use_gemini`，
   也不要把 provider 的 `wire_api` 改成 `chat`。
2. 按下表填写 `base_url`，保留服务的自定义路径前缀。使用目标接口的模型目录
   （通常为 `GET <base_url>/models`）核对模型：用户指定的 ID 优先；否则优先选择
   模板中的模型，或目录中同系列的明确对应 ID。多个候选无法判断时再请用户选择。
   若服务不提供模型目录，依据用户提供的服务文档或实际调用验证，不因 `/models`
   不可用而判定生成接口不可用。
3. 将 key 写入配置文件同目录的 `.env`，变量名与 `env_key` 一致；若用户已经通过
   环境变量配置则直接使用。保留其他变量，不覆盖另一服务正在使用的 key；必要时
   为新 profile 选择独立变量名。`.env` 在加载时会覆盖同名环境变量，应检查是否有旧值。
   key 不写进 TOML、仓库、命令行参数或输出；本地 `.env` 使用 `0600` 权限。
4. 新安装时将选好的模板保存为 `~/.codex/config.toml`。已有配置时保存为
   `<name>.config.toml`，保持默认 profile 不变；避免与已有文件或
   `[profiles.<name>]` 重名。profile 名仅用字母、数字、`_`、`-`。
   基础 `config.toml` 必须存在；使用自定义 `--config` 路径时，profile 和 `.env`
   都放在该文件旁。
5. 核对合并后的配置。profile 递归继承基础配置，特别检查模型专用的
   `model_context_window`、`model_auto_compact_token_limit`、推理强度及显式 prompt
   是否仍适用于新模型。用户指定的设置优先；否则根据部署说明和仓库模型元数据
   设置窗口与压缩阈值，不把其他模型的限制直接当作新模型能力。
6. 用下面的命令验证配置和一次真实模型响应，成功后给出 CLI 命令和 Web profile 名。
   仅模型目录可访问不能证明生成或工具调用可用；需要确认工具执行时，再做一次
   简单的只读工具调用及续轮验证。验证失败时报告实际错误，不静默切换接口或模型。

| 接口 | `base_url` 示例 | 生成请求追加路径 | 鉴权 |
| --- | --- | --- | --- |
| Responses | `https://host/prefix/v1` | `/responses` | `Authorization: Bearer <key>` |
| Chat Completions | `https://host/prefix/v1` | `/chat/completions` | `Authorization: Bearer <key>` |
| Claude Messages | `https://host/prefix/v1` | `/messages` | `Authorization: Bearer <key>` |
| Gemini | `https://host/prefix/v1beta` | `/models/{model}:streamGenerateContent?alt=sse` | `x-goog-api-key: <key>` |

不要将 `/responses`、`/chat/completions`、`/messages` 或具体模型的生成路径写入
`base_url`。Claude 模板适用于接受 Bearer 鉴权的 Messages 网关；当前适配器不通过
配置添加 Anthropic 直连所需的 `x-api-key` / `anthropic-version` 头。

模板只保留接入必需项及典型推理设置，不设置未经部署确认的上下文大小。
窗口未配置时读取仓库模型元数据；元数据也没有时窗口未知。
压缩阈值未配置且元数据未提供时，自动压缩关闭。Agent 应在安装时结合目标服务
完成这些设置；字段含义见[上下文配置](../docs/CONTEXT.md#context-用量与压缩)。

## 验证与启动

安装为默认配置后：

```bash
pycodex doctor
pycodex
```

安装为 `stepfun.config.toml` profile 后：

```bash
pycodex doctor --profile stepfun
pycodex --profile stepfun
```

在仓库开发环境中，将 `pycodex` 替换为 `env -u VIRTUAL_ENV uv run pycodex`。
Web 新建会话会自动发现同目录的 profile，无需重启；已有会话保留原配置。
也可用 `pycodex --config /path/to/config.toml` 指定独立配置。

完整的合并规则见[配置读取与 profile](../docs/CONTEXT.md#配置读取与-profile)，
接口适配范围见[Responses Server](../docs/responses_server/README.md)。
