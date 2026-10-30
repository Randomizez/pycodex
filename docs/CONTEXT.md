# Context

## 基线与结论

当前实现是 2026-10-29 新版协议原型，源码参照 upstream
`9cc14e79882ff8885cd5d33b4adadae014045aef`；HTTP 审计参照 CLI
**0.165.0**（release commit `360319d8dee62d87e2d59fc0c21805d79519cfed`）。
本页描述当前实现契约；抓包的覆盖范围和保留差异见
[Alignment](ALIGNMENT.md#current-prototype-baseline-2026-10-29)。

旧版 0.115/0.138/0.153.4 的一致结论不适用于当前整包。
共享 context 的投影比较会排除部分字段，也不能代替原始 wire-format 比较。

## 构造边界

- `Agent` 的 history 只维护真实 user/agent/assistant/reasoning/tool items。
- `ContextManager` 每次请求注入 developer、AGENTS 和 environment，
  不将它们写回真实 history。
- Agent 使用 client 的原始 model 名称。元数据先取最长 slug 前缀；
  无匹配时仅允许剥掉一个由 ASCII 字母、数字、`_`、`-` 构成的
  provider namespace，例如 `custom/gpt-5.4`。请求里的 model 不改名。
- Base instructions 的优先级仍是显式 override、config
  `base_instructions`、`model_instructions_file`、模型 prompt、通用默认。
- 模型专用 prompt 在 `pycodex/prompts/models.json`，不通过字符串补丁或
  另一套 serialized-tools 通道覆盖请求。
  本地 `step-3.5-flash`、`step-3.5-flash-2603`、`step-3.6` 均通过该表接入；
  新增模型也应沿用这个入口。
- `AgentsManager` 构造根和子 Runtime，Agent 仍由 Runtime 单向控制；
  低层 `Agent` 构造必须显式传入 `session_id`。不恢复 collaboration mode
  切换、turn cancellation 或固定 iteration cap。
- `agent_id` 和 canonical `agent_path` 标识 Agent；resume 改变的是 rollout
  `session_id`。`ContextManager.agent_path` 由 Manager 在注册时设置。
- Agent 通过 `Prompt.session_id` 传入请求身份，普通采样及所有压缩路径一致；
  retry/fallback 继续使用同一身份。Context 的冻结配置通过 `config` 公开，
  模型和 cwd 的运行时修改仍走维护缓存的操作接口。

`x-codex-turn-metadata.workspaces` 按逻辑 turn 区分：首个 turn 的
steer/follow-up 请求继续携带，后续新 turn 才省略，不是整个 session
仅第一条请求携带。历史对比及其测试隔离边界见
[Steer semantics](ALIGNMENT.md#steer-semantics)。

## 配置读取与 profile

启动入口使用 `CodexConfig.from_env(config_path, profile, use_default_profile=True)`。
它沿用既有 dotenv 加载规则，读取一次选中的 TOML，返回 `path`、实际 `profile`
和冻结的 `provider` / `context` 配置。`from_env()` 默认仍读取
`~/.codex/config.toml`；显式路径及其相邻 `.env` 不变。API key 只保存环境变量名，
每次请求由 `api_key()` 动态读取，不在加载时冻结凭据。

`build_model(config.provider)` 只构造模型及所需服务，不再接收路径或重复加载；
`AgentsManager(client, config.context)` 接收上下文配置。覆盖参数直接使用
`dataclasses.replace(config.context, cwd=work_dir)` 或
`replace(config.provider, reasoning_effort=None)`，省略的字段保留原值，
显式 `None` 清除原值；不使用特殊的 config `with_overrides`。
客户端的 `with_overrides` 用于创建子客户端，其 `None` 参数仍表示继承。
`CompatServerConfig` 同样通过 `replace(config, port=0)` 修改。

单独读取组件时仍可使用 `from_codex_config`；它们和统一入口共用
`from_mapping` 解码，后者接收已合并的配置。Context 解码仍基于原配置路径
解析相对文件和读取 home AGENTS，dotenv 只在启动加载边界处理。

`ResponsesProviderConfig` 和 `ContextConfig` 共用
`pycodex.utils.codex_config.read_codex_config`。先读基础配置，再将同目录的
`<name>.config.toml` 或基础配置内的 `[profiles.<name>]` 作为覆盖层；显式 `--config`
指定其他文件时，profile 仍相对于该文件的目录。`.env`、AGENTS 和相对
`model_instructions_file` 也保持以该目录为根。

表递归合并，标量、数组及不同类型的值整体替换；provider 从合并后的
`model_providers` 选择，因此 profile 可以修改现有 provider 的部分字段，
也可以定义新的 provider。profile 名仅允许 ASCII 字母、数字、`_`、`-`。

选择顺序为显式 `--profile name`、基础配置顶层的 `profile = "name"`、
基础配置本身。两种格式共用递归合并语义；旧内嵌表可继续使用，无需迁移。
profile 覆盖层不能再通过 `profile` 选择其他 profile。

同名的独立文件与内嵌表并存时，选择该名称或枚举 profile 列表会明确报错，
需只保留一处定义；未选择的同名冲突不影响仅加载基础配置或其他 profile。
`profiles` 和内嵌条目必须是表。选中的 profile 在两种来源中均不存在、
名称无效或 TOML 无效时直接报错，不回退到基础模型。

读取结果的 `profile` 字段记录实际选择，基础配置为 `None`。
Workspace 在启动时解析默认选择并保存每个会话的实际名称；恢复显式保存的
基础配置时传 `use_default_profile=False`，避免配置默认值变化导致换模型。
CLI、doctor、provider/context 的普通读取默认启用顶层 selector。

文件分层和合并行为参照 CLI 0.166.0。本地额外保留内嵌表和顶层 selector
兼容，并要求选中的 profile 存在，以维持 Workspace 保存会话恢复时的
provider 一致性。此处不代表实现了上游所有系统、项目或托管配置层。

## Context 用量与压缩

修改用量显示时，区分 CLI 与 workspace 的分母：

- CLI 展示剩余 context 百分比。窗口优先采用 config/profile 的
  `model_context_window`，其次是模型元数据 `context_window`；按模型的
  `effective_context_window_percent` 缩放，未指定时为 `95%`。
  显示归一化使用 `BASELINE_TOKENS=12000`：从 effective window 和
  `usage.total_tokens` 中扣除这个基线，用量最低为零。模型窗口已知时，
  第一次 usage 到来前显示 `100%`。
- Workspace 优先以自动压缩阈值为分母，未配置阈值时才使用最大窗口。
  compact/resume 后 usage 未知，当前界面先显示 `100%`，随后由新的
  provider usage 更新；该值不表示压缩后的提示为空。
  浮层首行为 `To compact: N/compact_limit`，`N` 是当前 usage；
  尚未知时显示 `—`，无压缩阈值时显示 `Off`。下方 Agent 树的文字取最新
  assistant 消息，工具调用期间保留该消息，运行状态由圆点和呼吸动画表示。
- 正常统计来自最近一次 `response.completed.response.usage.total_tokens`，
  不是把各轮 usage 累加。固定指令和工具定义在下一次请求中仍占 context；
  压缩只替换真实历史，不删除这些注入项。

阈值、溢出重试、摘要保留用户语言、部分摘要及 append-only checkpoint
的实现约束统一见 [Compaction](RUNTIME.md#compaction)，不在本页维护第二份。

## 普通 Responses

```python
{
    "model": original_model_name,
    "tools": [tool.serialize() for tool in tools],
    "input": [
        {
            "type": "message",
            "id": stable_base_instruction_id,
            "role": "developer",
            "content": [{"type": "input_text", "text": base_instructions}],
        },
        {
            "type": "message",
            "role": "developer",
            "content": [
                {"type": "input_text", "text": custom_developer_instructions},
                {"type": "input_text", "text": skills_instructions},
                {"type": "input_text", "text": permissions_instructions},
            ],
        },
        *multi_agent_developer_messages,
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": agents_instructions},
                {"type": "input_text", "text": environment_context},
                *extra_contextual_content,
            ],
        },
        *real_history,
    ],
}
```

普通 Responses 的基础指令作为首个 developer input item，省略顶层
`instructions`；ID 与下述 lite 基础指令使用同一 UUIDv5 算法。
固定 context 随后进入 `input`，不重复放入历史。模型目录若提供
`model_messages.collaboration_modes.default`，其文本以
`<collaboration_mode>…</collaboration_mode>` 成为普通 developer message
的一个 content part。这只是模型指定的 Default 指令文本，不提供模式切换。

没有内容的可选 section 省略，不能为了保持示例长度发送空 section。
AGENTS 继续按全局指令、repo root、逐级到 cwd 合并；
每个目录优先 `AGENTS.override.md`。

Skills 的目录别名按发现的 root 顺序分配：用户 root 在 system root 前；
展示按 scope 排序，system skills 在 user skills 前。例：

```text
### Skill roots
- `r0` = `/path/to/.codex/skills`
- `r1` = `/path/to/.codex/skills/.system`
### Available skills
- system-skill: Description. (file: r1/system-skill/SKILL.md)
- user-skill: Description. (file: r0/user-skill/SKILL.md)
```

只存在 system skills 时其别名是 `r0`。
SKILL frontmatter 和 `agents/openai.yaml` 使用安全 YAML 解析；
`policy.allow_implicit_invocation: false` 的 skill 不出现在自动 catalog。
本地仍没有实现上游完整的显式 skill 调用/动态选择器。
模型 metadata 的 `include_skills_usage_instructions=false` 会省略
`### How to use skills`，但不移除 catalog。

默认 full-access environment：

```xml
<environment_context>
  <cwd>/workspace</cwd>
  <shell>bash</shell>
  <current_date>2026-09-23</current_date>
  <timezone>Asia/Hong_Kong</timezone>
  <filesystem><workspace_roots><root>/workspace</root></workspace_roots><permission_profile type="disabled"><file_system type="unrestricted" /></permission_profile></filesystem>
</environment_context>
```

路径进行 XML 转义。默认 read-only/workspace-write 的 filesystem XML
使用 managed/restricted entries；这只是请求描述，不新增实际沙箱执行层。
它们的 permissions 文本尚有旧差异：network 默认描述和 workspace-write
的 writable-roots 句子未全面对齐，因此不能套用 full-access 的一致结论。

## MultiAgentV2 developer context

普通 developer message 后依次放两个独立 developer items：Agent 角色指令、
多 Agent mode hint，然后才是 AGENTS/environment 等 contextual user items。
裸 `ContextManager` 未设置 `agent_path` 时不注入这两个 items。

- 角色按 `/root` 或子路径选择。配置覆盖优先，其次模型
  `model_messages.multi_agent.role.root|subagent`，最后是
  `pycodex/prompts/developer_messages.json`。该资源记录上游源码 commit。
- 模型提供的角色文本用 `<multi_agent_role>` 包裹；通用 fallback 沿用其
  原始文本形状。角色指令附加共享目录说明、并发数，以及仅在相关工具
  可见时加入的 wait/model-override 提示。
- Mode hint 优先采用配置，再取模型 `multi_agent.mode.hint_text`；
  否则 effort 为 `ultra` 时取 `proactive`，其他取 `explicit`，缺项再取
  bundled 默认值。最终用 `<multi_agent_mode>` 包裹。显式空字符串省略该项。
- `features.multi_agent_v2` 表支持 `root_agent_usage_hint_text`、
  `subagent_usage_hint_text`、`multi_agent_mode_hint_text` 和
  `max_concurrent_threads_per_session`。默认总容量为 16，包含根 Agent；
  旧 `agents.max_concurrent_threads_per_session` 按子节点数量加 1。
  同一数字同时用于提示和 Manager 的创建容量检查。满额时按最近使用顺序
  关闭没有存活子节点、待处理输入、命令或后台任务的空闲子 Agent；
  没有可回收节点则明确报容量不足。关闭后的子节点从 Manager、`list_agents`
  和 Workspace 树中注销，释放任务路径；同名重建会获得新 ID，不自动重载。

## 子 Agent 的 fork context

`spawn_agent(task_name, message, fork_turns="all")` 默认继承当前父历史。
`"none"` 使用空历史和子 Agent 自己的固定 context。子节点继承父节点
当前 cwd、工具集合及模型配置；工具环境、历史和 ContextManager 各自独占。

全历史 fork 使用 `ForkContext` 同时复制真实 history 和以 history 位置为锚点
的固定 context blocks。父请求的 prefix 保持在原位置，末尾再加入本地
handoff、子角色、必要的 mode 更新以及 `NEW_TASK`。handoff 明确父身份、
旧任务和操作授权属于祖先，历史中的工具进程句柄不会变成子节点的活动句柄。
这些说明属于 context，不写入真实 history。

后续采样保留锚点；子 Agent 修改 cwd、effort 或其他 context 后，把新增
固定 blocks 放在当前历史末尾，不重写继承的 prefix。嵌套 fork 复制这一
完整布局。Context-overflow 重试若裁剪历史，锚点跟随仍存活的 items，
不会从快照复活已裁掉的工具记录。成功 compact 或 resume 替换历史后，
清除继承 context 快照，按当前 Agent 重新装配。

fork 的请求投影保留已完成 call/result；未配对的 function/custom call
补一个 `output="aborted"` 的结果，孤立结果省略。补出的结果只进入请求，
不修改父历史，也不执行继承的调用。存在 provider call item ID 时，结果
ID 用固定 namespace `90d38d3e-6a5b-4d52-bfe2-2f1e634bfac4` 和
`fco:<call-item-id>` / `ctco:<call-item-id>` 做 UUIDv5，保持重试稳定。

该 fork 是创建子 Agent 的 context 继承；会话 `/fork` 及相应 Agent/Runtime
入口已移除。子 Agent 仅保存在内存，重启根 Workspace 不恢复整棵树。

## AgentMessage 与 compat

`AgentMessage` 是输入历史 item，序列化为：

```json
{
  "type": "agent_message",
  "id": "amsg_<uuid>",
  "author": "/root",
  "recipient": "/root/review",
  "content": [{"type": "input_text", "text": "Message Type: NEW_TASK\nTask name: /root/review\nSender: /root\nPayload:\n检查实现"}]
}
```

`AgentsManager` 从 spawn/follow-up、send-message 参数或子 Agent 完成/报错
事件构造消息，分别使用 `NEW_TASK`、`MESSAGE`、`FINAL_ANSWER`。
Runtime 先排入 mailbox/steer，在模型输入安全边界写入接收方历史。
消息本身不启动 idle Agent；`followup_task`/spawn 另行 trigger。
Rollout 以 `response_item` 记录这种输入，字段名不表示它由模型直接输出。

Function call 的 `encrypted_function_args` 原样保存和回放。当前判定规则是：
显式 `[]` 表示明文；其他值（包括缺字段的 `None`）按加密消息处理。
加密分支使用明文 header 和独立 `encrypted_content` 正文，不解密、不将其
转成 user 文本。`author`、`recipient`、`id`、content 及可选的
`internal_chat_message_metadata_passthrough` 在 resume/compact 回放中保留。

Chat-completions/messages compat 把明文 AgentMessage 转成 user 消息，
加 `Agent message from <author> to <recipient>:` 前缀；messages 后端再将
正文装入 text blocks。compat 输出的 function call 显式带
`encrypted_function_args: []`，因此本地协作消息保持明文。包含
`encrypted_content` 的输入明确返回 HTTP 501，不做解密或静默降级。
原生 provider 返回明文参数却省略该标记时不符合这一协议约定。

## Responses-lite

模型 metadata 启用 `use_responses_lite` 时，不发送顶层 `instructions` /
`tools`，而在 `input` 前放：

1. `additional_tools(role="developer", tools=...)`
2. 基础指令的 developer message（非空时）
3. 普通 context 和真实 history

这些 prefix 的 ID 按上游算法生成：以 thread/session id 建立
`UUIDv5(NAMESPACE_OID, session_id)` namespace，再对工具 JSON 的紧凑 UTF-8
字节和基础指令文本分别计算 UUIDv5，前缀是 `at_` / `msg_`。
内容不变时 retry/resume 的 ID 稳定，内容变化时 ID 改变；不另建缓存层。
工具 JSON key 顺序参与哈希，不能自行排序。

`parallel_tool_calls=false`，reasoning 非空时加 `context="all_turns"`；
图像 detail 按原有 lite 路径处理。原始带后缀 model 名称原样发送。

## 历史回放

- Assistant 保留 provider `id`、`phase` 和 output-text 分段；
  不把 annotations 等上游不回放的字段当正文。
- Function/custom calls 保留 `id`、`namespace`，custom call 也保留 `status`；function arguments
  执行时解析为对象，回放仍使用收到的原始 JSON 字符串及加密参数标记。
- Reasoning 按上游 optional-field 语义序列化：缺 content 时为 `null`；
  非空 content 只有含 `reasoning_text` 才保留，encrypted content 保留。
- Rollout resume 和 compact replacement-history 继续保留这些字段；
  用户事件读取仍沿用既有恢复路径，不新增一套 session history。
- `exec_command` / `write_stdin` 结果从 `Chunk ID:` 开始，不再额外加入
  `Command:`。原始命令仍在 tool call 中。

## 比较边界

`comparison.json` 同时提供：

- `context_differences`：真实 instructions/input，保留字段有无和顺序；
  仅把客户端动态 UUID、Chunk ID 和 Wall time 规范化。
- `shared_context_differences` 和 `shared_context_exclusions`：单独移出工具
  声明，保留旧版投影的排除规则：客户端生成的 message/result ID，以及 developer
  `collaboration_mode`、`multi_agent_role`、`multi_agent_mode`。
  不排除 provider assistant/tool-call ID、phase、arguments 或一般正文。
- `tool_declaration_differences`：比较普通顶层 tools 或 lite additional_tools。
- `non_context_differences`：其他 body 字段（不比较 session prompt-cache-key
  的随机值）；新 `client_metadata` 的缺失保持可见。
- `header_differences`：原始 header 差异，包括身份、大小、大小写、遥测和动态值。

上述比较字段来自旧版诊断脚本；其中排除协作/多 Agent 指令的结果仅供历史
对照。当前原型必须另行检查这些 developer blocks 和原生 AgentMessage，
不能以排除后的相等证明新版协议一致。
当前原始请求仍有 user/context/tool-result ID、工具 namespace/目录、
client metadata 和请求头等差异，详见 Alignment。
不为满足字节比较而把 context 塞回 history、扩展生命周期或模拟整套上游遥测。
完整工具目录、interactive steer、模型切换、外部/plugin skills、custom
permission profiles 和 remote compact 不在此次通过的矩阵内。

## 复现

从 repo root，先 `env -u VIRTUAL_ENV uv sync --dev`，然后：

```bash
env -u VIRTUAL_ENV uv run --dev python -m tests.compare_context_requests
env -u VIRTUAL_ENV uv run --dev python -m tests.compare_context_requests \
  --model gpt-6-astra:caishi-azure --personality default \
  --reasoning-effort xhigh --scenario plain --scenario resume
```

需要本机可执行的 `codex`。默认创建并保留独立临时目录，打印报告路径；
可用 `--codex-bin /path/to/codex` 指定要审计的版本，报告记录实际版本与路径；
也可指定一个新的 `--root`（相对路径在启动子进程前解析为绝对路径）。
配置、skills、AGENTS、历史和请求均来自 fixture，
不读取真实账号配置，不调用外部模型。`--sandbox` 可用于诊断其他默认 profile。
新版 fixture 显式开启 MultiAgentV2，关闭 code mode 和 WebSocket transport，
确保比较走可抓取的 HTTP 与直接工具调用路径。
比较发现差异时退出 `1` 是审计结果，不是抓包失败；不能只看退出码宣称已对齐。

本地协议回归可独立运行，不需要 upstream 可执行文件：

```bash
env -u VIRTUAL_ENV uv run --dev pytest \
  tests/test_multi_agents.py tests/test_multi_agent_context.py \
  tests/test_runtime_state.py tests/test_builtin_tools.py \
  tests/responses_server/test_server.py \
  packages/pycodex-ws/tests/test_workspace_agents.py
```

同步 prompt data：

```bash
env -u VIRTUAL_ENV uv run --dev python tools/sync_model_prompts.py \
  /path/to/codex/codex-rs/models-manager/models.json
```

只同步共有 slug 的 `base_instructions` / `model_messages`，
保留本地 Step 模型及其他 runtime metadata。同步后必须重跑测试和抓包，
不能把更新 prompt 等同于更新整个 Codex 协议。
