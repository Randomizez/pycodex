# Responses Server

这部分文档只描述独立的 incomming / outcomming HTTP server。

它和现有 `pycodex/` runtime 保持隔离，目标是建模一个面向 Codex 的本地
`/v1/responses` 服务，再把请求翻译到 outcomming backend。

当前 HTTP backend 使用 FastAPI。
当前独立测试里的 fake outcomming chat backend 也使用 FastAPI。

## 目录边界

- incomming server 实现在 `responses_server/`
- 这一组测试放在 `tests/responses_server/`
- 这一组文档放在 `docs/responses_server/`

## 当前范围

当前 server 只覆盖最小闭环：

- incomming `POST /v1/responses`
- incomming `GET /v1/models`
- outcomming `POST /v1/chat/completions`
- 流式 assistant 文本
- vLLM chat-completions `reasoning` / `reasoning_content` -> Responses `reasoning` item 适配
- vLLM 历史 `reasoning` item -> assistant message `reasoning` 字段回放
- 普通 function tools
- custom tools 的 function-wrapper 兼容适配
- mock `web_search` 接口对齐（返回空结果）
- function / custom tool follow-up history 重建

## 当前明确不支持

这些能力当前会被显式拒绝，而不是静默降级：

- 真正的 Responses `web_search` 执行；当前只做空结果 mock
- 结构化 `input_image` tool output
- 非流式 incomming 请求

## Incomming / Outcomming 分层

- incomming：面向 Codex 的 Responses 子集
- outcomming：面向 backend 的 chat completions 子集

当前职责拆分：

- `responses_server/app.py`：FastAPI app 和 CLI 入口
- `responses_server/server.py`：`ResponseServer`，负责持有 `SessionStore` 和 `StreamRouter`
- `responses_server/stream_router.py`：`StreamRouter`，负责 incomming 请求翻译、outcomming chat 请求和流路由；对 `model_provider = "vllm"` 额外适配 chat-level reasoning
- `responses_server/payload_processors.py`：按 `CompatServerConfig.model_provider` 选择 provider-specific payload `post_process`
- `responses_server/tools/`：provider 侧工具适配层；当前放 mock `web_search` 和 custom-tool function wrapper
- `responses_server/session_store.py`：最小隐藏状态存储

## 运行方式

最小启动方式：

```bash
uv run python -m responses_server \
  --outcomming-base-url http://127.0.0.1:8000/v1 \
  --model-provider vllm
```

默认会在本地启动一个 incomming Responses 服务；真正监听地址由 `--host` 和 `--port`
控制。

如果下游 provider 需要对 chat payload 做定制化改写，可以在
`responses_server/payload_processors.py` 里注册对应 `model_provider -> proc_fn`
映射；server 会在真正发出每一条 outcomming `/v1/chat/completions` 请求前，
对 canonical `outcomming_request` 调一次这个 hook，默认按 `vllm` 处理。
当前内置规则里，`vllm` 仍走 chat-completions compat 路径，但会额外保留
reasoning；`stepfun` 会删除所有 `developer` role。

## 验证

当前独立测试：

```bash
uv run pytest tests/responses_server/test_server.py -q
```

这些测试通过一个假的 outcomming chat server 做端到端验证，不依赖现有
`pycodex` runtime。

测试结构现在是：

- 主 incomming server：FastAPI app + `fastapi.testclient.TestClient`
- fake outcomming chat server：FastAPI app + `uvicorn` 临时实例
