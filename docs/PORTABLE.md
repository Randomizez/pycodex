# Portable configuration

Portable 属于 `python-codex` core。入口保持简单：

```bash
python -m pycodex.portable.server --host 127.0.0.1 --port 5577
pycodex --put @127.0.0.1:5577
```

`--put` 成功输出的最后一行是可在目标机器执行的
`pycodex --call SECRET-CALLID@<host:port>`；这里的 token 是格式占位符。
真实 call token 包含解密秘密，不能写进文档、日志、Git 或工单。

## Startup boundary

- 远端 Codex home 的处理只发生在 `pycodex/cli.py` 的启动前阶段。
  `--put` 上传本地 bundle；`--call` 下载、解密并落成本地 `CODEX_HOME`，
  然后重写 `args.config`。
- `model`、`context` 和 `runtime` 继续按 `config_path.parent` 读取 `.env`、
  `AGENTS.md`、skills 和模型指令，不感知远端存储模式。
- `config.toml`、prompts、AGENTS、skills、dotenv、session history 的文本读取
  显式传 `encoding="utf-8"`；用户编写的指令和历史使用 `errors="replace"`。
  不能依赖 Windows GBK 等进程默认编码。

## Transfer contract

| 请求 | 行为 |
| --- | --- |
| `GET /healthz` | 上传前检查服务可达，成功后才扫描和压缩目录 |
| `POST /v1/storage/put` | 上传客户端已加密的 bundle，返回与 call token 配套的 call ID |
| `GET /v1/storage/call/<call_id>` | 只下载密文，解密和解压均在客户端 |

Call token 保持 `SECRET-CALLID@<host:port>` 的形状。加密格式及纯 Python
实现见 [Runtime package boundary](RUNTIME.md#frontend-boundary) 和
[vendored source notes](../pycodex/utils/_aes_gcm/README.md)。

打包使用白名单：

- `config.toml`、`.env`、`AGENTS.md`、`AGENTS.override.md`；
- `skills/**`；
- config 中相对引用的 `model_instructions_file`。

`sessions/` 等运行态目录不在白名单中。不要恢复依赖黑名单排除运行态文件
的扫描策略。

## Output and verification

`--put` 先做 health preflight，再扫描和打包；打包时打印文件清单，上传前
打印目标。上传后执行真实 `--call` 下载/解包 round-trip，最后打印成功结果
及一键启动命令。
上传成功后若 round-trip 失败，整个 `--put` 仍返回失败；不能仅凭上传结果
报告成功。最后一行保留可执行的 call 命令，便于复用。

排查“卡住”时，先核实目标 `host:port` 上的 storage server 和 `/healthz`。
端到端检查以 put/call 两个 endpoint、token 格式和客户端恢复为准。
测试覆盖入口见 [Portable 加密](../tests/TESTS.md#portable-加密)，
真实凭据和 call token 不进入测试报告。
