# cf_share_poc——共享服务端能否跑在 Cloudflare 上的一次性验收装置

回答「共享知识库能否上 Cloudflare」那份评估（在维护者的本地计划里，不随公开快照发布）里
A 路线——整个共享服务搬进 Python Worker + Durable Object——唯一的未知数：
**现有的 `KnowledgeStore` 能不能不改一行地跑在 Durable Object 的 SQLite 上**——即那条路线是
「改造」还是「重写」。**2026-09-09 实测：能。**

这不是产品代码，也不随主线维护；它是那份评估的证据。真要走 A 路线时，`entry.py` 里的
`DoConnection` 就是适配器的雏形。

## 结果（2026-09-09，wrangler 4.130 / pywrangler 1.17.2 / Pyodide → Python 3.13）

```json
{"sqlite3": "real",
 "steps": [{"step": "init_schema", "tables": 17, "has_node_versions": true, "schema_version": "5"},
           {"step": "transactionSync commit", "rev": 1, "node_read_back": "原神", "begin_commit_pairs": [2, 2]},
           {"step": "transactionSync rollback", "raised": "RuntimeError: deliberate",
            "doomed_node_present": false, "rev_after": 1, "rollbacks_seen_by_adapter": 1}],
 "statements": 36}
```

读出来的五件事：

1. **`sqlite3` 在 Python Workers 里是真模块**（`"real"`，`entry.py` 里的 stub 分支没被用到）——
   `store.py` / `schema.py` 顶层那两个 import 不必动。
2. **17 张表的 DDL 与迁移阶梯原样通过 `ctx.storage.sql.exec()`**，`schema_version` 落到 5。
3. **`store.begin()` 的 `with` 形态可以保留**：`BEGIN`/`COMMIT` 在适配器里中和，原子性由外层
   `ctx.storage.transactionSync()` 提供。评估稿里担心的「contextmanager 得改写成回调」不必做。
4. **回滚是真的**：事务内抛异常 → 节点不存在、`current_rev` 没有前进、异常照常向上传播。
5. `model.py` / `schema.py` / `store.py` **一个字都没改**（PoC 按下面的命令复制它们）。

对照组 `control.js`（同形状的 JS DO）另外钉死两条平台事实：`sql.exec("BEGIN IMMEDIATE")` 被明确
拒绝并**指名要用 `transactionSync`**；`SELECT last_insert_rowid()` 是拿 `lastrowid` 的可行替代
（DO 的 cursor 没有这个字段）。

## 怎么跑

```bash
mkdir -p /c/cfpoc/src && cp entry.py wrangler.jsonc pyproject.toml /c/cfpoc/ 2>/dev/null
cp entry.py /c/cfpoc/src/entry.py
cp -r src/finesub/llm/knowledge/node/{model,schema,store}.py /c/cfpoc/src/knode/   # 加一个空的 __init__.py
cd /c/cfpoc && uv venv .venv --python 3.13 && uv pip install --python .venv "uv>=0.12.3" workers-py workers-runtime-sdk
PATH="/c/cfpoc/.venv/Scripts:$PATH" ./.venv/Scripts/pywrangler dev --port 8803 --ip 127.0.0.1
curl -s http://127.0.0.1:8803/
```

⚠ **三个坑，每个都花了一轮**：

- **路径不能长**。放在深目录（Claude 的 scratchpad 就是）时，本地 miniflare 的 SQLite 起不来，
  一切 DO 调用都返回没有任何细节的 `internal error`——**连官方最小示例和 JS 对照组都一样**。
  换到 `C:\cfpoc` 这种短路径立刻全通。
- **`pywrangler` 要 uv ≥ 0.12.3**。别去动全局 uv：装一个到本地 venv，把它的 `Scripts` 放进
  `PATH`（Git Bash 里要写 POSIX 形式 `/c/...`，写成 `C:/...` 不生效）。
- **`workers` 模块要靠 `workers-runtime-sdk` 依赖装进来**，所以入口是 `pywrangler dev` 而不是
  `wrangler dev`；用后者会 `ModuleNotFoundError: No module named 'workers'`。

`knode/` 是那三个文件的**复制**，故意不入库：入库就会与源漂移，而这个装置的全部意义正是
「跑的是没改过的那份」。
