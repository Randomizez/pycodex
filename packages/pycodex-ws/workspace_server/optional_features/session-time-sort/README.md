---
title: "会话按时间排序"
summary: "按最新用户消息时间重排工作区会话列表，保持窄屏悬浮会话点原位，最多展示最近 20 个并保留当前会话。"
tags: [sessions, sidebar, sorting]
---

# 会话按时间排序

## Behavior

内置页面默认保持会话的创建顺序，并在每个会话标题下方按分钟展示距最后活动的时间（如 `<1m ago`、`1h 12m ago`）。应用本功能后，会话侧栏与抽屉列表（`#tabs`）按最新一条用户消息时间（`last_user_at`）从新到旧排列，时间相同时按会话 ID 倒序排列；模型回复只更新活动时间文案而不改变会话顺序，窄屏悬浮会话点（`.mobile-session-dots`）保持原有槽位不跳动。例如，当用户在较早创建的会话 A 中发送新消息后，A 会自动排到侧栏顶部，无需手动寻找。

当工作区会话超过 20 个时，列表默认只展示最近活跃的 20 个会话；若当前选中的会话不在前 20 名内，它会占据第 20 个可见位置，被隐藏的旧会话不会被关闭或删除。排序通过 CSS Flex `order` 原地调整视觉顺序，不会在每两秒轮询时搬移 DOM 节点或打断会话重命名输入框的焦点。

## Source

- [session-time-sort.mjs](source/session-time-sort.mjs)：提供纯函数 `sessionActivity`、`sortSessionsByActivity`、`keepMobileSessionOrder` 与内置页面挂载控制器 `mountSessionTimeSort`。
- [custom.js](source/custom.js)：面向 `~/.pycodex/frontend/custom.js` 的 `mount(host)` 适配入口。
- [custom.css](source/custom.css)：隐藏超出上限会话项及展示请求错误提示的样式。
- [verify.mjs](source/verify.mjs)：可重复运行的排序、上限、旧版 UUIDv7 回退、错误提示与卸载清理检查。

## Integration

先阅读已安装的 `WORKSPACE_API.md` 并检查现有定制文件。将 `session-time-sort.mjs` 复制到用户的 `~/.pycodex/frontend/` 目录，在现有 `custom.js` 的 `mount(host)` 中调用 `mountSessionTimeSort(host)`（默认 `{limit: 20, pollMs: 2000}`，传 `limit: null` 可保留全部会话），并在 `custom.css` 中合并本功能的样式规则。保留无关的既有定制并维持单一 `mount(host)` 入口。

适配器仅依赖内置宿主契约（`getActiveSessionId`、`onSessionChanged`、`request`）及页面已有的 `#tabs`、`.mobile-session-dots` Flex 容器，每两秒通过轻量级 `GET api/sessions` 读取各会话的 `last_user_at`（旧版后端缺失该字段时回退到时间有序的 `last_final_id`）。`#tabs` 更新视觉顺序与 20 条上限，`.mobile-session-dots` 仅同步 20 条可见上限而保持原有槽位。页面隐藏时自动暂停轮询，重新可见或切换会话时恢复；若请求失败，在侧栏顶部展示可见错误并保留上一次排序直到恢复。

若自定义前端自行接管了会话轮询，也可直接请求后端的显式参数 `GET api/session?recent=true`（或带 `session_id`），由服务端直接返回按 `(last_user_at, id)` 倒序截取的前 20 个摘要及全局 `workspace_running`。

撤销方式：调用 `mountSessionTimeSort` 返回的清理函数并移除合并进 `custom.js` / `custom.css` 的代码，或用 `?builtin=1` 临时恢复内置页面。

在仓库根目录执行验证：

```bash
node packages/pycodex-ws/workspace_server/optional_features/session-time-sort/source/verify.mjs
```

## Validation

Node 验证（`source/verify.mjs`）覆盖了 `last_user_at` 优先排序（模型新回复 `last_message_at` 不改变顺序）、UUIDv7 `last_final_id` 回退、25 个会话截断到 20 个且保留当前活跃旧会话、`keepMobileSessionOrder` 与 `.mobile-session-dots` 槽位稳定、`limit: null` 全量保留、请求错误可见性、新插入 DOM 标签同步隐藏，以及清理后恢复原始 `style.order`、属性与事件监听。

真实 Chromium + `pycodex-ws` 检查覆盖了桌面端会话栏按用户消息置顶、390px 移动端悬浮点槽位稳定、会话重命名输入框在轮询期间保持焦点、24 个会话下的 20 条上限切换，以及 `?builtin=1` 恢复默认创建顺序且保留回复时间显示。

## Preview

未附带静态截图。启用后外观与内置会话栏一致：每个会话保持原有的标题、状态圆点与 `<1m ago` / `1h 12m ago` 活动时间副行，仅侧栏上下顺序随最新用户消息时间动态置顶。

## Limitations

本目录是可选参考源码，需合并到用户前端后生效，不会自动改动内置默认页面。内置挂载方式通过 Flex `order` 调整视觉顺序以兼容内置轮询和重命名焦点，键盘 Tab 顺序仍跟随底层 DOM 顺序；完全接管会话列表渲染的自定义模板可直接用 `sortSessionsByActivity` 与 `keepMobileSessionOrder` 输出的顺序渲染 DOM 节点。
