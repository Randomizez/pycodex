# Context

## 基线与结论

2026-09-23 重新抓取本机 **Codex CLI 0.153.4** 的真实 HTTP 请求，
并核对 tag `rust-v0.153.4`、commit
`3d2ee51ca2d5db578f328aa75e20aa22c0197c9a`。
旧版 0.115/0.138 的抓包不能证明当前版本整包一致。

使用隔离的 HOME/CODEX_HOME、相同配置、AGENTS/skills fixture、相同 SSE
输出，比较 Codex CLI 和 `build_model → build_agent → build_runtime` 的
真实 `/v1/responses` 请求：

| 配置 | 场景 | 共享 context |
| --- | --- | --- |
| `gpt-5.4`, pragmatic, high | 首轮 | 一致 |
| 同上 | 保存后重新创建 runtime 并 resume | 一致 |
| 同上 | reasoning + exec_command + tool result 后续请求 | 一致 |
| `gpt-6-astra:caishi-azure`, 默认 personality, xhigh | 首轮、resume | 一致，显式排除协作/多 Agent 指令和工具声明 |

**共享 context 一致不等于原始请求严格相同。** 下文列出所有排除项，
脚本也保留原始差异，发现原始 context 差异仍退出 `1`。

## 构造边界

- `Agent` 的 history 只维护真实 user/assistant/reasoning/tool items。
- `ContextManager` 每次请求注入 developer、AGENTS 和 environment，
  不将它们写回真实 history。
- Agent 使用 client 的原始 model 名称。元数据先取最长 slug 前缀；
  无匹配时仅允许剥掉一个由 ASCII 字母、数字、`_`、`-` 构成的
  provider namespace，例如 `custom/gpt-5.4`。请求里的 model 不改名。
- Base instructions 的优先级仍是显式 override、config
  `base_instructions`、`model_instructions_file`、模型 prompt、通用默认。
- 模型专用 prompt 在 `pycodex/prompts/models.json`，不通过字符串补丁或
  另一套 serialized-tools 通道覆盖请求。
- 这次对齐不改变 `AgentRuntime → Agent` 两层结构，也不恢复 collaboration
  mode、turn cancellation 或固定 iteration cap。

## 普通 Responses

```python
{
    "model": original_model_name,
    "instructions": base_instructions,
    "tools": [tool.serialize() for tool in tools],
    "input": [
        {
            "type": "message",
            "role": "developer",
            "content": [
                {"type": "input_text", "text": custom_developer_instructions},
                {"type": "input_text", "text": skills_instructions},
                {"type": "input_text", "text": permissions_instructions},
            ],
        },
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
  执行时解析为对象，回放仍使用收到的原始 JSON 字符串。
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
  声明，排除客户端生成的 message/result ID，以及明确不实现的 developer
  `collaboration_mode`、`multi_agent_role`、`multi_agent_mode`。
  不排除 provider assistant/tool-call ID、phase、arguments 或一般正文。
- `tool_declaration_differences`：比较普通顶层 tools 或 lite additional_tools。
- `non_context_differences`：其他 body 字段（不比较 session prompt-cache-key
  的随机值）；新 `client_metadata` 的缺失保持可见。
- `header_differences`：原始 header 差异，包括身份、大小、大小写、遥测和动态值。

当前普通模型的原始 context 差异主要是未合成本地 user/context/tool-result
UUIDv7 ID。实际 Astra 模型另外有上游协作/多 Agent 指令与工具列表差异。
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
也可指定一个新的 `--root`。配置、skills、AGENTS、历史和请求均来自 fixture，
不读取真实账号配置，不调用外部模型。`--sandbox` 可用于诊断其他默认 profile。
比较发现差异时退出 `1` 是审计结果，不是抓包失败；不能只看退出码宣称已对齐。

同步 prompt data：

```bash
env -u VIRTUAL_ENV uv run --dev python tools/sync_model_prompts.py \
  /path/to/codex/codex-rs/models-manager/models.json
```

只同步共有 slug 的 `base_instructions` / `model_messages`，
保留本地 Step 模型及其他 runtime metadata。同步后必须重跑测试和抓包，
不能把更新 prompt 等同于更新整个 Codex 协议。
