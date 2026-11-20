# 可选前端功能

点击功能名称可查看详情。在对话中发送功能名称或 ID（例如“添加 exec-status”），或直接描述你想要的界面改动即可启动定制 Agent。

| ID | 功能 | 简介 | 验证情况 |
| --- | --- | --- | --- |
| clock-ticks | [定时唤醒时间分隔线](clock-ticks/README.md) | 将周期唤醒消息渲染为低调的时间分隔线，可展开查看触发详情。 | Chromium 历史/实时更新、原文保留、会话切换、键盘/移动端与清理检查；合成预览 |
| consecutive-tool-groups | [连续工具调用折叠](consecutive-tool-groups/README.md) | 将连续多条工具调用折叠为单行可展开项，并显示运行与失败计数。 | Chromium 样例、实时更新、刷新、键盘/移动端与清理检查；需前端接入适配 |
| exec-status | [后台命令状态](exec-status/README.md) | 在输入框上方展示运行中及已退出的后台命令会话。 | Chromium 样例、生命周期与桌面/移动端检查；未测试真实进程 |
| file-attachments | [本地文件附件](file-attachments/README.md) | 选择、粘贴或拖拽本地文件，支持预览、单文件重试与读取期间发送保护。 | Node 队列/编码/容量/会话隔离检查；Chromium 输入框样例、预览清理与移动端布局；未调用真实模型 |
| reasoning-display | [已记录的推理摘要](reasoning-display/README.md) | 展示对话历史中可读的模型推理摘要与文本。 | Chromium 样例、加密/空白过滤与桌面/移动端检查；未测试真实模型 |
| session-time-sort | [会话按时间排序](session-time-sort/README.md) | 按最新用户消息时间重排会话栏并保持窄屏悬浮会话点原位，最多展示最近 20 个并保留当前会话。 | Node 用户输入排序/圆点稳定/上限/旧版 ID 回退检查；Chromium 桌面与移动端、重命名焦点、新建/切换与清理检查 |
| session-tips | [会话私有便笺](session-tips/README.md) | 在对话右上角提供仅保存在当前浏览器的单会话便笺卡片。 | Chromium 样例、会话/rollout 隔离、刷新、存储失败与桌面/移动端检查；合成会话 |
| skill-completion | [Skill 快捷补全](skill-completion/README.md) | 输入 @/$ 时弹出前缀过滤的 Skill 候选菜单，支持键盘选择与自定义目录。 | Chromium 默认关闭、键盘/点击/触控、撤销/上下文块、过期响应、自定义目录与清理检查；Node 前缀检查 |
| tool-calls | [工具调用详情](tool-calls/README.md) | 配对展示近期函数与自定义工具调用的输入参数和返回结果。 | Chromium 样例、生命周期与桌面/移动端检查；未测试真实工具 |
| voice-input | [Step Plan 语音输入](voice-input/README.md) | 浏览器录音并经 Step Plan 转写整理后插入当前草稿。 | Chromium 模拟麦克风、模拟 ASR/润色、密钥流程、撤销/上下文块、取消、移动端焦点/布局与清理检查 |
| xiaoai-notifications | [小爱音箱播报通知](xiaoai-notifications/README.md) | 在选定的小爱音箱上播报简短的最终结果，并在对话中保留播报记录；需显式适配后端。 | 设备/队列隔离测试、真实 Runtime 完成/打断/重启检查，以及 Chromium 桌面/移动端记录展示检查 |

界面改动对该服务用户的全部工作区生效，刷新页面即可加载。访问工作区时附带 `?builtin=1` 可临时恢复内置界面。

Agent 运行状态已内置在上下文圆环弹窗中；点击弹窗内的根 Agent 或子 Agent，即可在右侧查看区打开其实时 transcript。

所提供的状态面板在无匹配内容时保持隐藏，内容消失后自动收起；未选中会话和初始加载阶段不占用空间。请求错误保持可见；推理记录仅展示包含可读文本的条目。

表格中每个已发布功能均链接到 `<feature-id>/README.md`。目录仅列出当前代码库中已包含源码与文档的功能；提案和未合并 PR 不计入可用功能。

如需贡献功能，可在对话中告诉 Agent：“将我的本地改动发布为功能 PR。”GitHub PR 的目录结构、README 元数据与提交格式见[贡献指南](CONTRIBUTING.md)。
