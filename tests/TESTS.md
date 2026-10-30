# Tests

这个文件记录当前 `pycodex` 的测试面，以及每类测试的预期行为。

## 按包运行

测试客户端 `ScriptedModelClient` 只放在 `tests/fakes.py`，不进入运行时包。
core 及测试收集/import 时执行的共享辅助代码必须能由 Python 3.6.2 解析；
运行时不会进入的分支也不能含 walrus、`match` 等新语法。

- `tests/` 覆盖 core：Agent、Runtime、CLI、工具、Responses compat 和整套
  portable，支持 Python 3.6.2+。
- `packages/pycodex-ws/tests/` 覆盖 Web、飞书及跨前端集成，
  支持 Python 3.7+。混合测试按行为归属拆开，纯 Runtime 用例仍留在 core。
- 根目录 `conftest.py` 为两组测试隔离 `CODEX_HOME`；workspace 的
  `conftest.py` 另隔离 `HOME`，避免飞书测试读取开发者的真实凭据。
  两组共用 `tests/fakes.py`。

在仓库根目录运行全部测试或仅 workspace 测试：

```bash
env -u VIRTUAL_ENV uv sync --dev
env -u VIRTUAL_ENV uv run --dev pytest
env -u VIRTUAL_ENV uv run --dev pytest packages/pycodex-ws/tests
```

core 使用独立的 `test` 依赖组。在单独环境验证，可保留日常开发环境中的
workspace 安装：

```bash
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=.tmp/core-tests uv sync --group test
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=.tmp/core-tests uv run --group test pytest tests
```

`test` 不安装 workspace 或飞书。Pillow 用于图片工具的生成、解码及
有/无 Pillow 两种行为测试；PyCryptodomex 用于加密参考结果。两者只在测试
依赖组中，不是 core 的运行时依赖。`dev` 包含 `test`、workspace 和开发工具，
保留 `uv sync --dev` 一次安装两包的方式。pytest 默认收集这两个测试目录。
两组都不自动安装可选的 `lark-oapi`，飞书 SDK 行为通过 fake 验证。

飞书卡片生产路径会读取 `~/.codex/.feishu_refresh_token`。上述 workspace
fixture 会隔离 HOME；脱离这组 fixture 的测试必须自行隔离 HOME。相关回归：

```bash
env -u VIRTUAL_ENV uv run --dev pytest packages/pycodex-ws/tests/test_feishu_card.py packages/pycodex-ws/tests/test_feishu_link.py
```

## CI 环境

- `test.yml` 和 `publish.yml` 分别运行 core 和 workspace job。
  core 在 Python 3.7、3.8、3.10 使用 `uv sync --group test`，并在运行
  `pytest tests` 前确认 `workspace_server`、`pycodex_ws`、`lark_oapi` 和
  `IPython` 均不可导入。workspace 在 Python 3.7、3.8、3.10 使用
  `uv sync --dev`，只运行 `packages/pycodex-ws/tests/`。
- core 的 Python 3.6 兼容性需要用真实解释器运行；现代 Python 下的语法检查不覆盖旧
  asyncio 和依赖库的执行行为。异步测试显式标记 `pytest.mark.asyncio`，
  避免 pytest-asyncio 0.16 在没有 auto mode 时跳过这些用例。
- `test.yml` 和 `publish.yml` 的 Python 3.6 job 使用
  `python:3.6.15-slim-bullseye`，系统依赖仅从
  `https://archive.debian.org/debian` 的 `bullseye main` 归档安装。
  不使用镜像内旧的 security / updates 源，也不假定 `bullseye-security`
  归档存在。这个冻结环境仅用于旧版本兼容性测试，不提供持续的安全更新。
- Python 3.6 job 只从源码测试 core，使用
  `PYTHONPATH="$PWD" python -m pytest -c /dev/null tests`。路径以容器内的
  `$PWD` 为准，不包含 workspace 源码，也不安装 workspace 或 SDK。
  根目录的语法检查只扫描 core、Responses compat、core 测试和共享 fixture。
- workspace 线程清理覆盖兼容层的 `asyncio.all_tasks()`，确保启动/关闭失败后
  仍关闭事件循环。workspace 脚本测试由 fixture 显式提供 `fetch`、
  `AbortController` 等浏览器接口；模拟请求可回传迟到结果，以覆盖切换后的
  视图隔离。这些测试随 workspace job 运行。
- 归档源使用 `check-valid-until=no` 允许读取已过有效期的固定索引，
  但继续校验仓库签名；`APT::Update::Error-Mode=any` 保证索引更新失败时
  立即停止，不会继续拿旧索引安装已经不存在的包。
- 如果已推送的 tag 在 CI 初始化阶段失败，修复 workflow 后应从包含修复的
  branch 手动运行 `publish`，并选择正确的 `repository`。直接重跑旧的
  tag workflow 仍使用旧提交；不要为了 CI 修复强制移动已公开的 tag。
- 两个包都构建普通 `py3-none-any` wheel。AES-GCM 使用内嵌纯 Python 实现，
  不增加编译器、原生库构建或平台 wheel 矩阵。
- `smoke_core_wheel.py` 在独立安装环境中检查 NIST 向量、认证失败、
  1 MiB portable round-trip、模型元数据和提示资源读取、CLI/doctor/
  `pycodex.portable.server` 模块入口，以及 core 不安装 Crypto、Cryptodome、
  Pillow、FastAPI、Pydantic、pydantic-core 或 workspace 模块。必须离开源码目录执行：
  `python /path/to/pycodex/tests/smoke_core_wheel.py`。
- `publish.yml` 构建 core 和 workspace 的普通 wheel/sdist。
  安装 core 后再安装同批 workspace，检查两个 console script、
  workspace 模块入口和依赖，卸载 workspace 后检查 core 仍可用；另从 sdist
  重新构建并验证 portable。全部成功后才上传发布产物，发布顺序保持 core 在前。
  PyPI 和 TestPyPI 均先上传 core，再上传 workspace；core 上传失败时
  workspace 发布 job 不执行。workspace wheel smoke 还从源码目录外运行
  `python -m pycodex_ws.feishu_oauth --help`，确认授权入口随包分发，
  并验证安装 workspace 不会带入 `lark_oapi`。
  core 的 `test_packaging.py` 覆盖模块入口的帮助、doctor 分发及参数错误退出码；
  workspace 的同名测试检查双包版本、依赖、模块和命令的归属。

## 自动化测试

Workspace 重连诊断折叠的浏览器回归手动运行，不加入 CI。
使用 Node 18+，在独立目录安装 Playwright 及 Chromium 后运行：

```bash
npm install --prefix /tmp/pycodex-browser-check playwright
/tmp/pycodex-browser-check/node_modules/.bin/playwright install chromium
NODE_PATH=/tmp/pycodex-browser-check/node_modules node packages/pycodex-ws/tests/workspace_retry_groups.cjs
```

覆盖有限/无限重试、1000 条记录的折叠和滚动、增量更新保持展开状态、
其他诊断及最终错误可见、会话隔离、重新渲染、键盘、手机触摸和 HTML 转义。
该脚本使用合成诊断与真实页面渲染函数，不请求模型。

会话缓存的完整页面浏览器回归使用同一 Playwright 环境，手动运行：

```bash
NODE_PATH=/tmp/pycodex-browser-check/node_modules node packages/pycodex-ws/tests/workspace_session_cache.cjs
```

覆盖首次直接加载、已删除的保存会话、服务错误、切换即时复用节点、
内容更新、展开状态、草稿、手机滚动、新建/关闭会话及切走后的 Mermaid
异步完成。所有请求使用离线 fixture，不访问真实会话或 CDN。

### HTTP 框架依赖

- `packages/pycodex-ws/tests/test_workspace_sharing.py` 覆盖管理员/访问者登录、
  单/多 workspace 的 HTTP/WS 访问范围、管理接口拒绝、密码更换/撤销后旧连接
  关闭，以及重启失效和配置不持久化。使用合成模型，不访问真实模型服务。
- `test_session_api.py` 覆盖单/多 workspace、普通/独立线程会话的资源查询、
  命令回执、结果、鉴权、分页和事件恢复；core 的 `test_history_page.py`
  验证压缩前历史与工具结果分页。扩展 host 的独立 Node 测试手动运行：
  `node packages/pycodex-ws/tests/workspace_custom_host.mjs`。
- 两处 HTTP 服务及 fake 下游均直接使用 Starlette。Python 3.6 CI 安装
  Starlette 0.19.x；安装和运行不需要 FastAPI、Pydantic 或 pydantic-core。
- `tests/responses_server/test_server.py` 覆盖路由别名、JSON 响应和 HTTP 错误，
  并通过真实本地下游验证 SSE、tool follow-up 和模型请求转换。
- `packages/pycodex-ws/tests/test_workspace_server.py` 覆盖单/多 workspace 的 JSON 对象校验、
  非 JSON Content-Type 拒绝、JSON HTTP 错误、登录和 WebSocket。
  `test_workspace_views.py` 还在文件打开操作阻塞时并发轮询会话，
  验证线程池边界；原有静态资源、图片目录边界和生命周期回归继续保留。

### Portable 加密

- `test_crypto.py` 对内嵌纯 Python AES-GCM 后端验证
  AES-256-GCM NIST 向量、空输入、非整块长度、1 MiB 输入，以及与 PyCryptodome
  的密文和标签互操作；验证 key/nonce/tag 长度、计数器耗尽保护、篡改拒绝、
  先验证再解密和旧 `PCX1` 密文。
  PyCryptodome 仅作为开发测试参考依赖。
- `test_portable.py` 继续覆盖打包、加解密、服务上传/下载和错误处理。

### Provider 配置

- `ResponsesModelClient` 默认 `stream_max_retries = 5`。CLI/REPL 测试若要求
  “先报错，再由下一句 `go on` 继续”，fixture 的 provider 配置必须设
  `stream_max_retries = 0`，否则自动重试会延后预期错误并使交互测试等待。
- `test_model.py` 覆盖省略 `model_provider` / provider 表时默认使用内置
  `openai`、显式选择内置 provider、profile 选择及 `openai_base_url` 覆盖、
  保留显式 provider 表语义，以及未知 provider 的明确错误。
- `test_doctor.py` 用本地 Responses fixture 验证省略 provider 的配置可完成
  配置、API key、连通性和模型请求检查，不访问真实模型服务。

### VS Code 前端

- `packages/pycodex-ws/extensions/vscode` 属于 workspace 前端，独立构建 VSIX。它使用 Node 22+，运行
  `npm ci --omit=optional --ignore-scripts && npm test`；与 Python 3.6
  测试环境独立，不给 Python runtime 增加 Node 依赖。按项目约定手动运行，
  不加入 CI。
- 前端测试覆盖项目后端启动、固定配置文件权限、退出/启动失败后配置保留、stdin 关闭、
  停用等待（含已移除目录的后端）、
  远程工作目录、venv 选择/校验、远端文件选择、项目级设置、
  profile 参数/重启生效及重新加载提示、
  context meter 快照字段、每个 chat 新建 session、重开侧边栏恢复已有对话及过滤空会话、
  显式 session id、POST 不自动重试、旧轮询、多 session 隔离及 detach 不关闭后端。
- 后端准备测试覆盖复用兼容版本、从 console script 定位原 Python 环境、
  空 venv 的独立安装与启动、旧版升级、通过 workspace 依赖升级 core、安装后版本复查、并发安装复用
  （含同一环境的 `python` / `python3` 别名），
  以及安装失败/缺少 Python/无效自定义路径报错与凭据脱敏。
  进程 fixture 在 POSIX 上执行；Windows launcher 解析使用固定样本验证。
  安装耗时不占服务启动超时，安装期间关闭会等待准备结束且不再启动后端。
- 编辑器文本块覆盖未保存缓冲区、1-based 光标、反向选区、截断和捕获后编辑/关闭文件，
  以及只读 Git diff、Diff 标签页聚焦、Explorer/SCM 单选和多选资源。
- 新会话无需源文件且默认没有代码上下文；可主动将片段插入草稿后随普通 prompt
  发送，续问不会自动附加。片段与文字穿插，发送按原顺序展开。
  命令、空输入和问题回答保持原样发送。
- WebviewView 测试覆盖活动栏入口、多项目选择、文件读取期间切换会话、
  上下文插入的 ready 握手和目标会话、折叠会话状态更新、
  `/model` 原样发送、关闭等待/失败与其他会话隔离，
  以及隐藏/恢复握手、CSP、过期消息和发送确认。VS Code API 使用 mock，
  桌面 UI 联调需运行 Extension Development Host；聊天通过稳定 API 在左侧栏打开。
- `uv run --dev pytest packages/pycodex-ws/tests/test_owned_workspace_server.py packages/pycodex-ws/tests/test_workspace_server.py`
  覆盖 UTF-8 workspace 配置文件、新格式及旧 `board: false` 的归一化，以及
  不传额外选项时默认启用 EOF 关闭、真实 CLI 使用关闭的 stdin 启动后正常退出，
  stdin 在启动前/启动后关闭时仍执行完整 lifespan shutdown，
  以及 stdin 保持打开时 SIGINT/SIGTERM 关闭不会导致解释器退出崩溃。

### Context 抓包审计（2026-09-23）

基线为本机 `codex-cli 0.153.4`；脚本使用隔离的 localhost fixture，
不读取用户账号配置、不调用真实模型：

```bash
env -u VIRTUAL_ENV uv sync --dev
env -u VIRTUAL_ENV uv run --dev python -m tests.compare_context_requests
env -u VIRTUAL_ENV uv run --dev python -m tests.compare_context_requests \
  --model gpt-6-astra:caishi-azure --personality default \
  --reasoning-effort xhigh --scenario plain --scenario resume
```

- 首轮、文件 resume、reasoning/tool follow-up 走生产装配和真实 HTTP。
- 默认创建新的临时目录并打印 `comparison.json`；`--root` 必须使用新的目录。
- 只对客户端动态 UUID、Chunk ID、Wall time 做动态值规范化。
  raw context 有差异时仍退出 `1`，不能把 shared context 一致当成整包一致。
- 报告同时保留 shared-context exclusions、工具声明、其他 body 和 header 差异。
- `test_context.py` 覆盖最长前缀/namespace、skills alias/顺序/YAML policy/
  model guidance 和 filesystem XML；`test_model.py` 覆盖 suffix wire model、
  lite prefix UUIDv5、assistant phase/分段及 tool 原始参数的 resume/compact 回放。
- `test_fake_responses_server.py` 验证审计投影不吞 provider ID/phase 差异，
  不会把 user 正文中的协作标签当作可排除的 developer 指令。
- 完整边界及已知差异见 `docs/CONTEXT.md`。

### 测试分层与去重

- 小幅文案、布局、样式改动做页面检查，不搭建专用 Node/DOM 模拟测试；
  状态或队列语义只保留必要的行为回归，按变更范围运行相关用例。
- 纯后端的权限、问题取消、fork 和关闭契约集中在 `tests/test_session_backend.py`；
  CLI/Web/飞书的会话命令与订阅集成集中在
  `packages/pycodex-ws/tests/test_session_backend.py`，不在每个前端重复测试后端命令。
- 两组 `test_async_bridge.py` 分别覆盖 core 与 workspace 在构造和执行使用
  不同事件循环时的行为；core 测试不导入 workspace。
- Agent/Runtime 保持单向控制：Agent 无队列引用，裸 `stop_asap` 不取消已发出的
  工具且保持 call/result 配对；Runtime 结算停止回执并启动下一批输入。
  compaction/overflow 期间停止、steer 合并、关闭排空及子 Agent 接入在
  `test_agent_state.py`、`test_runtime_state.py` 和 `test_agent.py` 覆盖。
- pre-turn、mid-turn 和 overflow compact 中收到停止请求时，不再发后续采样，
  不增加未发出请求的 iteration；溢出恢复只重试一次，不重跑已完成工具。
- `test_runtime_state.py` 覆盖 exec/clock 唤醒后的 steer：正常中断不报通知失败，
  真实模型异常仍上报，待处理用户输入由 Runtime worker 继续执行。
- 子 Agent 状态等待超时后仍能收到后续完成通知；Python 3.6 下也必须等
  condition waiter 取消并重新获得锁后再离开临界区，不能留下卡死的通知等待。
- `test_cli.py` 保留入口、装配、portable round-trip、steer/queue 反馈和流式展示；
  验证普通/JSON 回执不创建额外任务，移除逐个 argparse 字段、
  相同 buffer 状态和重复恢复流程的微测试。
- `test_workspace_server.py` 以路由安全、工作区 CRUD、tab 持久化、线程隔离为主；
  `state_path` 用例验证旧 HTML/无 Canvas 配置恢复原 rollout 和文件视图、
  Canvas 开关切换保持会话与视图状态、单/多 workspace 的 Canvas 路由关闭、
  前缀含点时的历史隔离，以及跨开关的重复前缀拒绝；
  损坏的会话索引直接报错并保留原文件，不跳过记录或创建空会话覆盖它；
  fork 重启覆盖普通/线程会话、连续未落盘 fork、标题更新、续写与源文件隔离，
  并覆盖 compact/模型失败后的恢复路径、关闭排空、空标签页和无效源路径。
  验证启动/清理失败仍回收线程，取消关闭调用方不打断当前 turn；
  `/history` 整条命令只产生一条控制消息，不逐行挤占聊天记录；
  执行前端实际切换/渲染代码，验证关闭 tab 后恢复已访问 tab 的内容、草稿和
  滚动位置，以及旧轮询迟到或报错时不能覆盖当前 tab；
  不再把 CSS 颜色、函数名等实现细节当成独立回归。
- `test_workspace_views.py` 覆盖模型只输出链接时不打开标签、点击不增加模型请求/会话消息、
  空工具集仍可打开文件、普通路径与 Git 风格 URI、百分号编码及非法目标拒绝、
  完整路径/Git 标题、有序比较与 Git 实际版本身份、并发打开去重、路径别名复用和 reveal 持久化、
  文件/diff/Git 行号定位、左右侧与数字 Git 文件名解析、无效行号拒绝、定位复用及原子保存、
  文件和图片自动更新、原子替换、未变文件不重读、删除/读取失败后的展示及恢复、
  持久化来源与旧快照保留、无 shell 执行的链接解析、文件/存储预算、
  Git 对象固定及工作区文件更新、打开/读取/关闭 API 的认证及跨 workspace 隔离，
  以及标签顺序和关闭/恢复后的更新。
  同文件通过 Node 执行 `workspace_content_views.mjs`，验证点击才切换、其他页面打开不抢焦点、
  新页面默认 Canvas、恢复选中状态、Canvas 关闭时的文件视图恢复/空状态与 URI 拒绝、
  连续点击与失败、手动切换/关闭/停止使旧打开失效、
  关闭后的旧轮询、迟到内容成功/失败、关闭期间切换标签及停止/恢复轮询；
  内容 revision 更新、后台缓存失效、更新时保留可见内容及编辑器、刷新失败重试，
  用户重开时切换已有标签、同标签重新展示、切换只处理一次及刷新期间的展示保留，
  定位使用打开响应、等待目标内容版本、普通刷新/手动切换不重放定位、
  Markdown/HTML 无行号预览与带行号源码的双向切换、模式及各自阅读状态保留、
  编辑器创建/内容更新/diff 等待期间的新定位或切换使旧请求失效，
  编辑器加载/创建期间切换标签、迟到实例销毁、阅读状态恢复、加载错误可见，
  HTML 预览更新时复用实例、切换时释放、原生资源引用保留，
  滚动消息按 frame 和文档版本校验，以及关闭后释放编辑器和 Markdown/图片缓存。
  另用页面实际的 ES module import 检查随包分发的 JS 导出，防止 HTML 与旧 bundle
  不一致导致整个聊天脚本无法启动，并限制首屏控制层 JS/CSS 大小，避免重新打入 Monaco。
  静态资源测试覆盖 gzip/identity 协商、拒绝的编码、HEAD、按表示区分的 ETag/304、
  文件更新后的缓存失效、每个应用实例的缓存隔离及 API 不进入压缩路径。
  `test_workspace_pdf.py` 覆盖本地 PDF 原文件响应、GET/HEAD 和下载、
  管理员/访问者鉴权与 workspace 隔离、超过 2 MiB 的文件、轻量持久化、
  文件更新/删除/恢复、无效文件头和行号拒绝。
  生命周期测试直接读取前端源码，无需安装 npm 依赖；发布前用真实浏览器检查
  聊天/Markdown/Canvas 链接、普通外链与锚点、危险协议过滤、中文和特殊字符路径，
  HTML 预览的样式/脚本、与 Canvas 一致的 iframe 同源行为、滚动恢复及普通链接返回预览，
  带登录时浏览器自动携带 cookie 加载相对图片、`<base>` 和 `srcset`，
  以及图片失败时 HTML 仍正常呈现，
  Monaco 的只读输入/粘贴/复制、手机触摸滚动与输入焦点（代码区原生只读、查找框可输入）、
  查找/折叠、双栏对齐、字符级 diff、容器响应式切换、整行定位及折叠行展开、
  宽屏定位左侧后缩窄保持可见、手机左右侧定位及切换恢复、
  大文件滚动恢复、文件更新后的阅读/查找状态、多个页面同步、
  标签栏鼠标滚轮/触控板滚动、文件名优先与同名目录区分、Git 次要说明、
  标签长标题省略及完整悬停信息、查看器完整路径标题的无滚动条横向滚动与固定操作按钮、
  编辑器/model/observer 释放和 worker 运行，并检查构建后的同源资源和 wheel 内容。
  按需加载需在真实浏览器中确认 Canvas/Markdown/HTML/PDF 预览不请求编辑器 JS/CSS，
  源码和 diff 首次打开才加载、等待样式完成后创建，后续切换复用资源；
  样式加载失败可见并可重试，裁剪后的 JSON/常用语言高亮、查找、折叠与字符级
  diff 保持可用，缩窄 diff 后返回 Canvas 不产生已销毁菜单的延迟回调错误。
- 恢复失败按缺失/空文件/损坏/无 session id 选代表状态，不再对不改变执行路径的
  persisted/closed 标志做完整笛卡尔积。数据落盘原子性、路径隔离、SSE 协议、
  截断、clock 和自然关闭等独立风险回归保留。
- `test_agent_state.py` 验证 `replace_history` 已移除，fork/resume 装配 recorder
  失败不改变原会话；延迟写入、恢复续写和无参 resume 的状态保留继续覆盖。
- 恢复时跳过未配对的 function/custom tool call，保留完整兄弟工具及后续对话；
  验证续写后再次恢复仍保留新历史，原文件前缀不被改写。
- 小测试文件并入所属组件：终端 API 兼容检查并入 `test_cli.py`，
  provider 迟到回调检查并入 `test_model.py`。
- 事件字段/冻结约束、纯文本和有状态展示在 `test_events.py` 验证，身份补全不修改原事件由
  workspace 的 `test_session_backend.py` 覆盖；`test_workspace_server.py` 固定 Web 序列化形状。

### `tests/test_events.py`

- 事件 `visualize()` 的命令、问答、权限、provider web search 和 compact 文案。
- 无状态格式化 helpers 位于 `utils/event_helpers.py`，继续在这里覆盖其文本/颜色输出和事件调用。
- 工具成功/失败/空输出、后台 exec session、Python heredoc、plan 进度及子 Agent 摘要；
  不在各前端重复遍历这些文本规则。
- `EventDisplay` 的订阅端状态隔离、着色、流缓冲/重试/fatal 顺序、steer 反馈、
  昵称替换、输入提示、快照恢复和非事件输出；不依赖终端或 SDK。
- compact summary 从计数派生，手动 compact 的结果只由命令事件输出一次，
  自动 compact 不结束 turn；ANSI 只出现在展示输出，不进入事件数据和纯文本视图。
- CLI 仅验证事件委派与实际 log/status/prompt 执行，不再重复遍历流重试规则。

### `packages/pycodex-ws/tests/test_feishu_card.py`

- 飞书复用独立的无色 `EventDisplay`；未知事件只需实现 `render` 即能展示，
  无需增加飞书分支。测试验证回调到卡片字段的适配、展示长度上限和流式区域。
- 灰框只保留最近完成的回复，下一轮开始时标记 `(*last turn)`，完成后替换；
  工具活动、错误和待答问题单独展示，不能混入回复正文。
- 覆盖快照恢复不重复 active turn、重试移除 partial、fatal 保留 partial、
  问答输入提示和关闭/detach；保留 Markdown 降级及请求传输回归。
- 包内的 autouse fixture 隔离 HOME，避免测试读取真实飞书 refresh token。
- refresh token 只从专用文件读取和轮换，旧环境变量不生效；覆盖多个实例及
  已缓存 access token 的实例在过期后接力、写入失败不缓存新 access token。

### `packages/pycodex-ws/tests/test_feishu_oauth.py`

- 首次授权只写 `~/.codex/.feishu_refresh_token`，权限为 `0600`，不创建或
  更新 `.env`，不打印凭证；`pycodex_ws.feishu_oauth` 从 `.env` 读取应用配置
  并写入共享 token 文件。
- 子进程在源码目录外运行模块的 `--help`，不需要应用凭据。

### `tests/test_session_backend.py`

- 权限接纳与显式 scope、空答/超时/detach/close 取消及过期 request id。
- fork 同步 Agent/provider identity，保留 history/usage；首次续写才创建 rollout。
- 未启动/重复关闭、工具清理错误和 worker 错误传播，不留下悬挂的关闭等待。
- 子进程验证 core 后端装配不导入 CLI、Web 或飞书。

### `packages/pycodex-ws/tests/test_session_backend.py`

- CLI/Web/飞书的全部会话命令使用相同后端、不落为模型 user message；
  compact 只调用摘要请求，fork 同步 Agent/provider identity、保留 history/usage；
  未续写即关闭不建空文件，续写只在新文件记录 metadata 和继承历史。
- 独立 attach/detach、多前端模型/标题同步、迟加入前端恢复 active stream；
  飞书 busy 时接受 steer，显式 queue 顺序不变。
- `/link` 在 SDK 不可导入时只报告不可用，不创建连接或调用模型；
  普通会话仍可使用。SDK 可导入时验证连接及关闭清理。
- 多题、选项标签、Other、权限 scope、结构化回答、空答/超时/detach/close
  取消及过期 request id。
- 单 worker、未启动/并发/重复关闭、工具关闭钩子只执行一次；
  关闭等待已开始命令，调用方取消不打断 worker，清理错误通过 worker 统一回传。
- 停止接纳不额外入队，拒绝 exec/clock 新唤醒；工具在已有请求排空后清理，
  包括排队中的请求重新设置 clock 的情况，清理期间禁止恢复会话。
- `test_runtime_state.py` 验证子 Agent 清理报错后仍关闭其余子 Agent，
  首个错误向调用方传播，其余错误向事件循环报告。
- 验证 Web 不导入 CLI。

### `tests/test_agent.py`

- `test_agent_loop_runs_tool_then_returns_final_message`
  - 预期：`AgentLoop` 能完成一轮 `ToolCall -> ToolResult -> 下一轮模型回复` 闭环。
- `test_parallel_tools_share_one_model_round`
  - 预期：支持并行工具批执行，不会把两个可并行工具串行跑掉。
- `test_runtime_submission_loop_processes_turn_and_shutdown`
  - 预期：队列通过 `start()` / `close()` 处理用户提交并自然关闭，不另建 worker。

### `tests/test_builtin_tools.py`

覆盖当前已接入的本地工具实现：

- `shell`
  - 预期：按 argv 执行命令；能返回工作目录、退出码、stdout/stderr；超时能正确报出。
- `shell_command`
  - 预期：按 shell script 字符串执行命令。
- `exec_command`
  - 预期：长命令会返回运行中的 `session_id`。
- `write_stdin`
  - 预期：能复用已有 session，向 stdin 写入并拿到后续输出。
- `exec`
  - 预期：能运行 raw JavaScript；支持返回 completed 或 running cell 状态。
- `wait`
  - 预期：能等待 `exec` cell 的后续输出或完成态。
- `web_search`
  - 预期：按 provider-native tool 形式序列化到 Responses API payload 中，而不是走本地 handler round-trip。
- `update_plan`
  - 预期：更新 `PlanStore`，并返回固定确认文本 `Plan updated`。
- `request_user_input`
  - 预期：保留工具声明，调用固定返回 `request_user_input is unavailable in Default mode`；无论是否注册 handler 都不收集答案。CLI/Web/飞书集成测试验证不会产生 `input_requested`，模型下一轮能收到不可用结果。通用输入服务独立验证回答传输；`test_model.py` 继续覆盖本地 `success` 不进入请求及旧问答 rollout 的恢复。
- `request_permissions`
  - 预期：把权限请求转发给交互层，并返回带 `scope` 的权限响应。
- `apply_patch`
  - 预期：支持 add/delete/update；整体验证失败时不留下部分写入。
- `grep_files`
  - 预期：只返回匹配文件路径，不把不匹配文件混进结果。
- `read_file`
  - 预期：按 1-based 行号切片读取，并带行号前缀。
- `list_dir`
  - 预期：列出绝对路径目录树片段。
- `view_image`
  - 预期：返回 data URL，并把工具结果序列化成 `input_image` content item。
  - 已安装 Pillow 时默认缩图；缺少 Pillow 时，省略或显式 `high` 均保留原图并返回
    `original`。显式 `original` 始终保留原图；验证字节、尺寸和序列化后的 detail。
- `spawn_agent`
  - 预期：happy path 下能创建子 agent 并通过 `wait_agent` / `close_agent` 正常回收；当前还额外校验了空参数时要返回固定校验错误 `Provide one of: message or items`。
- `send_input`
  - 预期：能给已有子 agent 继续发消息。
- `resume_agent`
  - 预期：关闭后的子 agent 能被恢复并继续接收输入。
- `wait_agent`
  - 预期：能等待子 agent 到达完成态。
- `close_agent`
  - 预期：能关闭子 agent，并返回关闭前状态。

### `tests/test_context.py`

- `ContextManager` 的 base instructions 优先级：
  - 预期：`override > config.base_instructions > model_instructions_file > 默认 prompt`。
- `ContextManager` 的 model instructions 解析：
  - 预期：能从 vendored `models.json` 按 `model + personality` 解析出与上游一致的 base instructions。
- `ContextManager` 的 prompt 拼接：
  - 预期：developer 上下文单独作为一条 `developer` message，且内部保留多个 `content` item；
  - 预期：`AGENTS.md` 指令和 `<environment_context>` 合并为一条 contextual user message，且内部保留两个 `content` item；
  - 预期：`~/.codex/AGENTS.md`、repo 根 `AGENTS.md`、当前目录 `AGENTS.override.md` 能按顺序合并；
  - 预期：permissions / skills prompt 的顺序和 shape 接近上游 Codex。
- `AgentLoop` + `ContextManager` 集成：
  - 预期：上下文只注入到本轮 `Prompt.input`，不会污染持久 history。

### `tests/test_model.py`

- `ResponsesProviderConfig` 能读 `~/.codex/config.toml` 风格配置。
- `ResponsesModelClient` 构造的 payload 要正确包含：
  - `developer` / `user` message 形式的上下文注入
  - `prompt_cache_key`
  - Codex 风格请求 headers（如 request id / turn metadata / originator）
  - 普通 function tools
  - freeform/custom tools
  - 不在 request-visible function tool payload 里序列化 `output_schema`
  - `custom_tool_call` / `custom_tool_call_output`
- 其中有一个 hardcoded reference 测试会直接固定一份示例 request body JSON，用来防止 payload format 被悄悄改坏。
- SSE 解析要能恢复 assistant message、function tool call、custom tool call。
- 如果 Responses 流里出现 `reasoning` item，协议层也要保留它，并允许下一轮请求把它原样回传。
- 真实 HTTP compat 回归覆盖持续 reasoning 超过客户端读空闲阈值后仍成功、无重试；两个 chat reasoning 字段和 messages thinking 流都不混入 assistant 正文，最终 reasoning 只保留一次。
- 普通 function-call 的 JSON 解析失败由 compat server 返回 `response.failed(type=server_error)`；客户端通用重试保持请求和 session 不变，已完成工具不重复执行，失败响应的工具不写入 history。直接收到坏参数时客户端不再自行归类为 retryable。
- 本地 compat 下游在响应头到达前断连时，返回 `response.failed`；模型客户端用相同请求重试并完成回复，不让网络异常逃出 ASGI。
- 下游失败的正常错误包含阶段、耗时、timeout 和接收进度；验证响应头前断连、首个 chunk 前失败、读流中断及 HTTP 错误。

### `tests/test_cli.py`

- 覆盖参数和 stdin/TTY 选择、单次调用和 JSON 输出、命令不进模型、失败退出码。
- 装配验证 chat/messages/vLLM/managed transport、模型选择、`codex-tui` originator、
  context 注入、工具选择和 schema 边界。
- 用本地 storage server 验证 `--put`/`--call` round-trip、默认 GBK locale 下读取 UTF-8，
  以及 `.env` 加载时过滤 `CODEX_` 前缀变量。
- 终端验证 steer/queue 反馈、重试不重复输出、fatal 保留 partial、context 和后台状态；
  清理失败也解除视图订阅；终端库参数兼容也在本文件中验证。
- 子进程内使用真实 prompt_toolkit 输入验证单次 Ctrl+C、Ctrl+D 和 POSIX SIGINT：
  空闲时正常退出，有 active/queued 请求时先排空再清理一次，不取消请求或泄漏输入任务异常。
  覆盖 3.0.36 的按键绑定接口及其退出输入后重置 SIGINT 的行为。
  pipe-input fixture 显式传入输入/输出，避免依赖 Python 3.6 回调不支持的 app-session 继承。
  `[closing]` 提示及不提前 flush partial 输出的规则由 `test_events.py` 覆盖。

### `tests/fake_responses_server.py`

- 这是 prompt/context 对齐时使用的本地 fake Responses API server。
- 代码放在 `tests/` 下，临时抓包输出默认仍然写到 `.tmp/prompt_capture`。
- 预期：同一台 fake server 可以同时接 upstream Codex 和 `pycodex` 的请求，然后把 `/models` 和 `/responses` 的交互稳定记录下来。
- 现在也支持 proxy 模式：传 `--proxy-base-url <upstream>` 后，会把请求真实转发到上游，同时把上游响应一起落盘。

### `tests/test_fake_responses_server.py`

- 本地起一个“上游小服务器”和一个 proxy capture server，验证 proxy 模式会真实转发 `/responses`，并把 response status / headers / body 记进 capture 文件。

### `tests/compare_tool_schemas.py`

- 用 proxy 模式分别抓 upstream Codex 和 `pycodex` 的真实首轮 `/responses` 请求。
- 工具顺序直接从本文件里的真实 smoke table 读取，逐个比较 request-visible tool schema。
- 当前比较的是被抓到的那条具体 context path；如果某个工具在这条路径下根本没有暴露出来，会明确记成 `not exposed`，而不是误判为 schema 相等。
- 真正的 `round-trip same` 仍然要靠单独的手工抓包确认；如果某个工具的默认参数显式与否有随机性，需要用更强约束的 prompt 固定参数，再比较 `tool_call` / `tool_result`。

### `tests/compare_steer_request_bodies.py`

- 用 deterministic fake origin + proxy capture 比较 upstream Codex 和 `pycodex` 在 steer 流程下的两轮 `/responses` request body。
- upstream 和 `pycodex` 都通过 tmux 驱动真实交互会话：先发第一条 prompt，再在第一条 request 仍未返回时发第二条 steer prompt。
- 这个场景里不能用 proxy capture 文件出现的时刻当作“第一条 request 已到达”的信号，因为 proxy handler 只有在整条 upstream response 读完后才会落盘；脚本现在改为同步等待 fake origin 收到第 1 条 POST，再注入 steer。
- 为了比较“默认 steer”而不是本机用户配置带来的 fast-mode 噪音，脚本会给 upstream 和 `pycodex` 都生成临时 config，并显式去掉顶层 `service_tier`。
- 所需 tmux/capture helper 直接放在本脚本内。完整请求比较会保留 pycodex 移除协作指令、修改问答工具可用性描述产生的有意差异，不再承诺整包相同。

## 真实模型 smoke tests

下面这些测试是用真实 `pycodex` + `~/.codex/config.toml` 跑过的 prompt 级 smoke，用来确认模型在真实对话里会主动选对工具。

说明：

- 文件类工具测试时使用仓库内临时目录 `.tmp_tool_smoke`。
- `spawn_agent` 相关工具有时会出现额外一轮自检或重复清理；只要目标工具被调用且最终回复符合预期，就记为通过。

| tool name | test prompt | expected behavior |
|---|---|---|
| `shell` | 必须且只需调用 `shell`，运行 `['bash','-lc','printf SHELL_OK']`，最后只回复 `SHELL_OK` | 应调用 `shell`，最终回复 `SHELL_OK` |
| `shell_command` | 必须且只需调用 `shell_command`，运行 `printf SHELL_COMMAND_OK`，最后只回复 `SHELL_COMMAND_OK` | 应调用 `shell_command`，最终回复 `SHELL_COMMAND_OK` |
| `exec_command` | 必须且只需调用 `exec_command`，运行 `printf EXEC_COMMAND_OK`，最后只回复 `EXEC_COMMAND_OK` | 应调用 `exec_command`，最终回复 `EXEC_COMMAND_OK` |
| `write_stdin` | 先 `exec_command` 跑 `read line; printf "$line"`，再 `write_stdin` 发送 `WRITE_STDIN_OK\n` | 应依次调用 `exec_command`、`write_stdin`，最终回复 `WRITE_STDIN_OK` |
| `exec` | 只调用 `exec`，运行 `text('EXEC_SMOKE_OK')`，最后只回复 `EXEC_SMOKE_OK` | 应调用 `exec`，最终回复 `EXEC_SMOKE_OK` |
| `wait` | 先 `exec` 运行一次会先 `yield_control()` 再输出 `WAIT_OK` 的脚本，再 `wait` 拿到后续输出 | 应依次调用 `exec`、`wait`，最终回复 `WAIT_OK` |
| `web_search` | 只调用 `web_search` 搜索一个明确问题，再基于搜索结果简短作答 | 应调用 `web_search`，并基于搜索结果回复 |
| `update_plan` | 先调用 `update_plan` 设两步计划，再只回复 `TOOL_OK` | 应调用 `update_plan`，最终回复 `TOOL_OK` |
| `request_user_input` | 调用一次 `request_user_input` | 返回 `request_user_input is unavailable in Default mode`，不弹出问题、不等待答案，Responses input 中没有 `success` |
| `request_permissions` | 只调用 `request_permissions`，请求 `network.enabled=true`；CLI 侧输入 `t` | 应调用 `request_permissions`，最终回复 `REQUEST_PERMISSIONS_OK` |
| `apply_patch` | 只调用 `apply_patch`，把目标文件里的 `before` 改成 `APPLY_PATCH_OK` | 应调用 `apply_patch`，最终回复 `APPLY_PATCH_OK` |
| `grep_files` | 只调用 `grep_files` 搜索 `NEEDLE_123` | 应调用 `grep_files`，最终回复 `grep_target.txt` |
| `read_file` | 只调用 `read_file` 读取目标文件 | 应调用 `read_file`，最终回复 `READ_FILE_OK` |
| `list_dir` | 只调用 `list_dir` 列出目标目录 | 应调用 `list_dir`，最终回复 `child` |
| `view_image` | 只调用 `view_image` 查看目标图片 | 应调用 `view_image`，最终回复 `VIEW_IMAGE_OK` |
| `spawn_agent` | `spawn_agent` 创建子 agent，再 `wait_agent` / `close_agent` 清理 | 至少应调用 `spawn_agent`、`wait_agent`、`close_agent`，最终回复 `SPAWN_AGENT_OK` |
| `send_input` | `spawn_agent` 后再 `send_input` 发第二条消息 | 至少应调用 `spawn_agent`、`send_input`、`wait_agent`、`close_agent`，最终回复 `SEND_INPUT_OK` |
| `resume_agent` | 子 agent 完成并关闭后再 `resume_agent`，然后继续 `send_input` | 至少应调用 `spawn_agent`、`wait_agent`、`close_agent`、`resume_agent`、`send_input`，最终回复 `RESUME_AGENT_OK` |
| `wait_agent` | 创建子 agent 后显式等待 | 至少应调用 `spawn_agent`、`wait_agent`、`close_agent`，最终回复 `WAIT_AGENT_OK` |
| `close_agent` | 创建子 agent 后显式关闭 | 至少应调用 `spawn_agent`、`wait_agent`、`close_agent`，最终回复 `CLOSE_AGENT_OK` |

## 当前通过情况

- 本地验证使用隔离 HOME/CODEX_HOME 的 `uv run --dev pytest`；数量以实际 collection 为准，
  不把历史通过数量当成当前覆盖范围。
- 已对 `exec` 做过真实模型 smoke，当前通过。
- 已对 `pycodex "请只回复当前目录的 basename，不要解释。"` 做过真实配置 smoke；预期回复应等于当前 checkout 目录名。
- collaboration mode 状态和模式专用对齐脚本已删除；`request_user_input` 的默认不可用行为保留，通过工具及 CLI/Web/飞书接入测试验证。通用输入服务的回答与取消测试继续独立运行。
- 已手工补抓 `write_stdin` 的 round-trip；在固定参数的 prompt 下，upstream Codex 和 `pycodex` 的 `function_call` / `function_call_output` 外层 schema 一致，tool result 文本包装也一致。
- `exec_command` / `write_stdin` 的本地 unified-exec 默认截断也已补齐：省略 `max_output_tokens` 时默认走 `10_000` token 预算；长时间未轮询的未读输出缓冲会保留 upstream 同款 `1 MiB` head/tail。
- 已补抓 `apply_patch` 的 `custom_tool_call_output`；当前 `pycodex` 已对齐 upstream 的 `Exit code` / `Wall time` / `Output` 文本包装。
- 已补抓 `view_image` 的 round-trip；当前 upstream Codex 和 `pycodex` 都会把结果回传成 `function_call_output.output = [input_image]`，其中 `image_url` data URL 一致。
- 已补抓 `spawn_agent` 的最小 validation-path；当前 upstream Codex 和 `pycodex` 在缺少 `message/items` 时都会回传 `Provide one of: message or items`。
- 已补抓 sub-agent happy path：`spawn_agent` / `send_input` / `wait_agent` / `close_agent` 当前在真实链路里的 `function_call_output` 外层 shape 已和 upstream 对齐；`pycodex` 也已补齐默认昵称和 `<subagent_notification>` 回灌。
- 已补抓 `resume_agent` 的真实 happy path：当前 `pycodex` 已对齐 upstream 的 `{"status":"pending_init"}` 返回值；同一条链路里的 sub-agent request 也已收敛到 upstream 那 6 个工具。
- sub-agent request body 里的 `prompt_cache_key` 现在也已对齐：parent thread 用自己的稳定 session id，sub-agent thread 改为使用 `agent_id`。
- sub-agent request 的 `x-openai-subagent: collab_spawn` header，以及后续 turn 不再携带 `workspaces` 的 metadata 细节，也已对齐 upstream。
- 已补抓默认 CLI 主线程的两轮无工具对话：当前 upstream Codex 和 `pycodex` 的首轮/次轮 request body 与 header shape 一致；第二轮都会省略 `workspaces`。
- sub-agent 那 6 个工具的 schema 现在来自类内 `BaseTool` spec，并由 CLI serialization 测试覆盖；`pycodex/prompts/subagent_tools.json` 已删除。
- 其余文件/agent/交互类工具 smoke 见上表；`web_search` 当前主要通过 payload/事件层测试验证接入，是否在真实模型里被主动选中还受具体 prompt 与 provider 行为影响。

## 后续新增工具时的记录要求

新增 tool 后，至少补两类记录：

1. 一个 `pytest` 级别的实现测试，验证本地 handler 的确定性行为。
2. 一个真实 `pycodex` prompt 级 smoke，验证模型在真实对话里会选到该工具，并记录 prompt 与预期最终回复。
