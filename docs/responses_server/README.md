# Responses Server

这部分文档只描述独立的 incomming / outcomming HTTP server。

它和现有 `pycodex/` runtime 保持隔离，目标是建模一个面向 Codex 的本地
`/v1/responses` 服务，再把请求翻译到 outcomming backend。

请求中的 `model` 必须透传到下游；不能读取 `/models` 的第一个 ID
并覆盖调用方选择。Provider 差异只在真正发送下游请求及读取 SSE 的边界适配，
`StreamRouter` 内始终保留 canonical payload，避免污染 tool hydration
和 mock `web_search` follow-up。

当前 HTTP backend 直接使用 Starlette 与 Uvicorn，不依赖 FastAPI、
Pydantic 或 `pydantic-core`。独立测试里的 fake outcomming chat backend
也使用 Starlette。Python 3.6 使用 Starlette 0.19.x。
路由显式返回 JSON 或 SSE，保留原有请求校验和错误状态；
不再提供 FastAPI 自动生成的 `/docs`、`/redoc` 和 `/openapi.json`。

compat runtime 的下游 chat-completions / messages 流读取 timeout 默认是
`300` 秒，与 pycodex Responses client 的 `300_000 ms` SSE idle timeout
对齐。

下游 `reasoning` / `reasoning_content` 分片会立即转成
`response.reasoning_text.delta`，同一分片的两个别名内容相同时只转发一次。
`/v1/messages` 的 `thinking` 分片经边界适配后也走这条路径。因此，只要
reasoning 分片持续到达且相邻间隔没有超过读空闲阈值，正文前的思考总时长
可以超过该阈值；真正没有数据的连接仍按原有 timeout 失败。客户端只把
最终 `response.output_item.done` 中的 reasoning 写入历史，增量不会显示为
assistant 正文。

Responses 请求里的 `reasoning.effort` 会转换为下游 chat 请求的
`chat_template_kwargs.reasoning_effort`。这与 vLLM Responses renderer 的
prompt 渲染路径一致，也保留 provider 自定义 effort 值（例如 `max`）。

## 目录边界

- incomming server 实现在 `responses_server/`
- 这一组测试放在 `tests/responses_server/`
- 这一组文档放在 `docs/responses_server/`

## 当前范围

当前 server 只覆盖最小闭环：

- incomming `POST /v1/responses`
- incomming `GET /v1/models`
- outcomming `POST /v1/chat/completions`
- outcomming `POST /v1/messages`（通过边界适配复用同一套 canonical chat request / stream routing）
- 流式 assistant 文本
- chat-completions `reasoning` / `reasoning_content` -> Responses reasoning 增量和最终 `reasoning` item 适配
- vLLM 历史 `reasoning` item -> assistant message `reasoning` 字段回放
- vLLM streaming `usage` -> final `response.completed.response.usage`
- 当环境变量 `PYCODEX_DUMP` 存在时，为每条 outcomming 请求附加 `return_token_ids = true`，并把实际下发的 request body、下游 usage/cache、`prompt_token_ids` / `token_ids` 以 JSONL 追加到 `{PYCODEX_DUMP}/dump.jsonl`
- 下游请求在等待响应头或读流时失败，会返回 `response.failed`，错误直接包含耗时、timeout、失败阶段、响应头/首个 chunk 的时间和 chunk 数量
- 普通 function tools
- 普通 function-call 参数拼接完成后校验 JSON；解析失败返回 `response.failed(type=server_error)`，由客户端按现有策略重试
- custom tools 的 function-wrapper 兼容适配
- mock `web_search` 接口对齐（返回空结果）
- function / custom tool follow-up history 重建
- 结构化 `input_image` content items：user message 里的图片直接转成 chat `image_url` part；tool output 里的图片会在该 `tool` message 之后补一条只带图片的 `user` message（chat backend 普遍会丢弃 `tool` role 里的图片）；`/v1/messages` 侧再转成 Anthropic `image` block

## 当前明确不支持

这些能力当前会被显式拒绝，而不是静默降级：

- 真正的 Responses `web_search` 执行；当前只做空结果 mock
- 非流式 incomming 请求

## Incomming / Outcomming 分层

- incomming：面向 Codex 的 Responses 子集
- outcomming：面向 backend 的 chat-completions / messages 兼容子集

当前职责拆分：

- `responses_server/app.py`：Starlette app 和 CLI 入口
- `responses_server/server.py`：`ResponseServer`，负责持有 `SessionStore` 和 `StreamRouter`
- `responses_server/stream_router.py`：`StreamRouter`，负责 incomming 请求翻译、outcomming request 和流路由；对 `model_provider = "vllm"` 额外适配 chat-level reasoning
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

如果下游不是 `/v1/chat/completions`，而是 Anthropic/Claude 风格的
`/v1/messages`，再额外加：

```bash
uv run python -m responses_server \
  --outcomming-base-url http://127.0.0.1:8000/v1 \
  --outcomming-api messages
```

默认会在本地启动一个 incomming Responses 服务；真正监听地址由 `--host` 和 `--port`
控制。

如果要导出下游 trajectory token id，可以在启动前设置：

```bash
export PYCODEX_DUMP=/tmp/pycodex-dump
```

server 会为每条实际转发到下游的请求附上 `return_token_ids = true`，并把
trajectory 追加到 `${PYCODEX_DUMP}/dump.jsonl`，当前记录格式是：

```json
{"request":{"model":"water18","messages":[],"stream":true,"return_token_ids":true},"usage":{"prompt_tokens":100,"completion_tokens":6,"total_tokens":106,"prompt_tokens_details":{"cached_tokens":64}},"finish_reason":"length","tokens":{"prefill":[1,2,3],"decode":[4,5,6]},"send_timestamp":2222.0}
```

`request` 是 provider post-process 之后真正发出的 JSON body；HTTP headers 和
API key 不会写入 dump。对多轮 tool-call，可以逐条比较
`request.messages` 是否是上一轮的严格前缀扩展，并读取
`usage.prompt_tokens_details.cached_tokens` 核对下游 prefix-cache 命中。
重试和 mock tool follow-up 都会各写一条独立记录。
`finish_reason` 保留实际下游终止原因（如 `stop`、`tool_calls`、`length`）；
usage-only chunk 不会覆盖它，未收到终止 chunk 时为 `null`。
长度截断不会补写 EOS，消费者必须同时校验 usage 和原始 token IDs 的完整性。
vLLM chat-completions 的 `return_token_ids=true` 流中，`prompt_token_ids`
只在首个 chunk 出现，后续 `choices[*].token_ids` 都是 decode delta。
重建 trajectory 时保留首个 prompt，再按顺序拼接所有 decode delta。

如果下游 provider 需要对 chat payload 做定制化改写，可以在
`responses_server/payload_processors.py` 里注册对应 `model_provider -> proc_fn`
映射；server 会在真正发出每一条 outcomming `/v1/chat/completions` 请求前，
对 canonical `outcomming_request` 调一次这个 hook，默认按 `vllm` 处理。
当前内置规则里，`vllm` 仍走 chat-completions compat 路径，但会额外保留
reasoning；`stepfun` 会把 `developer` role 转成 `system`。Step4 chat
template 不消费 `developer` role；使用该模板的 provider 应显式配置为
`stepfun`，否则 base instructions / AGENTS context 会被静默忽略。

如果下游 chat stream 一轮结束时只给了 `reasoning` / `reasoning_content`，
没有 assistant `content` 且没有 tool call，server 会丢弃这次累积的 partial reasoning，
不发对应的 `response.output_item.done`，并用原样 downstream request 静默重试一次。
已转发的 reasoning 增量不进入最终 history。若 retry 后仍是 reasoning-only，才发
`response.failed(type=model_output_invalid)`。这样可以避免 interrupted 或
length-stopped thinking 被持久化成 terminal reasoning-only history，并在下一轮转换成
下游 chat 后端不接受的裸 assistant message。

普通 function-call 的 `arguments` 在拼接完成后由 server 校验 JSON。解析失败时，
server 在发出本次最终 output items 之前返回 `response.failed(type=server_error)`；
错误只包含工具名和解析位置，不包含原始参数。`ResponsesModelClient` 通过通用
`response.failed` 处理按 `stream_max_retries` 重试当前请求，默认最多重试 5 次。
这条路径不增加 server 内部重试，也不把本次失败响应里的工具交给 Agent 执行。
合法 JSON 的内容保持原样；非 object 的参数仍由客户端按既有契约报错。
custom/freeform 工具继续使用原有适配流程，assistant 正文也继续作为文本处理。

`messages` compat 则故意不改这层 canonical request：仍然先构造 chat 风格
`outcomming_request`，只有在真正发请求和读 SSE 时，才在边界把它翻译成
messages request / event。这样 tool hydration、mock `web_search`
follow-up、provider payload post-process 仍然复用同一套主逻辑。

当前 messages 边界还会补一个兼容性细节：下游如果像 vLLM `0.19.0` 一样要求
`max_tokens`，则优先透传上游请求里的 `max_output_tokens` / `max_tokens`；
如果上游没给，当前默认补 `32000`，避免直接被下游 `400` 拒绝。

下游失败无需开启调试模式。正常错误里的 `phase=waiting_response_headers`
表示尚未收到响应头，`waiting_first_chunk` 表示响应头已到但尚无完整 chunk，
`reading_stream` 表示流已开始；`elapsed` 是本次请求耗时，`idle` 是距离
最近一个完整 chunk 的时间。SSE 心跳不计入 chunk 数。

## 验证

当前独立测试：

```bash
uv run pytest tests/responses_server/test_server.py -q
```

这些测试通过一个假的 outcomming chat server 做端到端验证，不依赖现有
`pycodex` runtime。

`tests/test_model.py` 另有真实 HTTP client → compat server → fake backend 的
读空闲超时回归：每 `0.1` 秒发送 reasoning 分片，正文延迟超过客户端的
`1` 秒空闲阈值，验证请求仍能成功且没有重试。覆盖两个 chat reasoning
字段和 messages thinking 流：

```bash
uv run pytest tests/test_model.py -k long_compat_reasoning -q
```

测试结构现在是：

- 主 incomming server：Starlette app + `starlette.testclient.TestClient`
- fake outcomming chat server：Starlette app + `uvicorn` 临时实例
