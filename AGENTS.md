# pycodex 开发约定

## 目标与包边界

- 尽量准确复现 [upstream Codex](https://github.com/openai/codex)；
  对齐以实际请求和行为为依据，保留下文及文档明确说明的本地差异。
- 只维护两个普通 Python 包：`python-codex`（core，Python 3.6.2+）和
  精确依赖同版本 core 的 `pycodex-ws`（Python 3.7+）。
  core 的目录、导入及 CLI 名仍是 `pycodex`，直接位于仓库根目录。
  Portable 和 `responses_server` 属于 core；Web、router、飞书适配属于 ws。
- 飞书 SDK `lark-oapi` 为手动安装的可选依赖；core 不依赖前端或 SDK。
  VS Code 扩展独立构建 VSIX，不进入 Python wheel/sdist。
  双包发布必须先成功发布 core，再发布 ws。

## 关键约束

- 后端保持 `AgentRuntime -> Agent` 两层。CLI/Web/飞书通过
  `AgentRuntime.submit_input` 接入；命令、队列、问答、订阅和关闭由 Runtime
  负责。前端做输入与展示，Agent 不反向持有 Runtime/队列。
- `Agent(model_client, tool_registry, context_config)` 三项必传；
  每个 Agent 独占 context manager、工具环境和历史，外层 Runtime 可选。
  使用显式协议和普通属性，不为缺字段添加静默兜底或机械 getter/setter。
- Agent 串行执行普通 async `run_turn`，不持有 turn Task、不提供取消 API。
  steer 在模型请求和已发工具完成后的安全边界协作式停止；
  正常关闭停止接纳并排空已接受工作，不用超时取消 turn。
  Follow-up/tool handoff 自然决定结束，不加固定迭代上限。
- 上下文装配统一在 `context.py`；固定指令、AGENTS、环境不写回会话历史。
  Rollout recorder 由 Agent 内部管理：默认仅内存，前端显式传记录路径，
  子 Agent 保持仅内存。恢复、fork 和压缩遵循文档中的一致性约束。
- 工具继承 `BaseTool` 并注册实例；schema 由真实建模层序列化。
  除明确扩展 `clock` 外，逐一对齐上游工具契约。
  不恢复 collaboration mode；`request_user_input` 固定返回 Default mode
  下不可用，通用 Runtime 问答/权限服务仍保留。

## 开发与验证

```bash
env -u VIRTUAL_ENV uv sync --dev
env -u VIRTUAL_ENV uv run --dev pytest
```

- 运行与测试依赖使用仓库的 uv 环境。core 及 import 阶段执行的共享测试
  辅助代码保持 Python 3.6.2 语法兼容。接口不使用 `*` 定义 keyword-only 参数。
- 修改 workspace 前端源码后，用其 `frontend/` 目录的 `npm ci` /
  `npm run build` 重建资源，不手改生成的 JS/CSS/许可证文件。
- 按改动范围验证；测试分层、隔离 HOME/CODEX_HOME、Node 版本和发布检查见
  [测试文档](tests/TESTS.md)。独立 Node/VS Code 测试手动运行，不加入 CI；
  Python 测试已有的 Node fixture 继续随原测试运行。

## 按修改范围阅读

修改模块前阅读对应章节；只加载本次任务涉及的内容，跨模块时补读相关章节。
下列文档中的实现契约与本文件共同约束开发，不因移出根文件而失效。

| 修改范围 | 阅读入口 |
| --- | --- |
| Agent、Runtime、队列、关闭 | [Ownership](docs/RUNTIME.md#ownership)、[Frontend boundary](docs/RUNTIME.md#frontend-boundary) |
| 历史、记录、resume、fork、compact | [Model and history](docs/RUNTIME.md#model-and-history-contracts)、[Compaction](docs/RUNTIME.md#compaction) |
| CLI、事件展示、飞书 | [Observation](docs/RUNTIME.md#observation)、[Feishu lifecycle](docs/RUNTIME.md#feishu-connection-lifecycle) |
| Prompt、模型配置、token 用量、重试 | [Context](docs/CONTEXT.md)、[Responses reliability](docs/RUNTIME.md#responses-reliability) |
| 工具、子 Agent、exec、clock | [Tools](docs/RUNTIME.md#tools) |
| Workspace 状态、API、Canvas、查看器、定制 | [Session API](docs/RUNTIME.md#workspace-session-api)、[Frontend customization](docs/RUNTIME.md#frontend-customization)、[Content views](docs/RUNTIME.md#workspace-content-views) |
| Responses compat、下游适配 | [Responses Server](docs/responses_server/README.md) |
| Portable 上传、下载、配置恢复 | [Portable](docs/PORTABLE.md) |
| 测试、打包、发布、上游比较 | [Package layout](docs/RUNTIME.md#package-layout)、[Tests](tests/TESTS.md)、[Alignment](docs/ALIGNMENT.md) |

## 文档维护

- 根文件只放跨模块关键约束、开发命令和阅读入口；模块契约维护在对应文档，
  一次性排障和验证经过留在 worklog，避免不断追加长段落。
- 实现、使用方式或对齐结论变化时，同步更新 README 和对应文档；
  历史验证结论注明适用版本，不将旧抓包等同于当前整包一致。
