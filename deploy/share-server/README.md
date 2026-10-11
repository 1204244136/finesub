# 共享知识库服务端部署（plan §6 服务端形态）

服务本体是 stdlib-only 的单进程 HTTP（`python -m finesub.llm.knowledge.share.server`），
**只绑定 127.0.0.1**；TLS、域名与对外暴露一律交给前面的 Caddy（终态）或 Cloudflare
Tunnel（过渡），服务自身永远不直接见公网。服务器上没有任何 LLM API key——审核在
维护者本机跑（`share review`），服务器只存队列与语料。

## 终态：轻量 Linux VPS

1. 装 Python ≥ 3.10 与本包（只需要 `[harness]` 之下更少的东西——实际上
   `finesub.llm.knowledge.share.server` 是 stdlib-only，`pip install finesub` 即可，或
   直接把仓库 `src/` 放上去用 `PYTHONPATH` 跑）。要签快照（`--signing-key-file`）再加
   `cryptography`。
2. 建数据目录（仓库外）：`/var/lib/finesub-share`。
3. 生成 maintainer token，写进只有 root 可读的文件（同一串存进密码管理器）：

   ```bash
   sudo install -d -m 700 /etc/finesub-share
   python3 -c "import secrets; print(secrets.token_urlsafe(32))" | sudo tee /etc/finesub-share/maintainer-token >/dev/null
   sudo chmod 600 /etc/finesub-share/maintainer-token
   ```

   服务只从 `--maintainer-token-file` 或 `FINESUB_SHARE_MAINTAINER_TOKEN` 读它，**不收命令行参数**：
   命令行（`/proc/<pid>/cmdline`）对同机所有用户可读。
4. 装 systemd 单元（`finesub-share.service` 用 `LoadCredential` 把上面的文件交给服务）：

   ```bash
   sudo cp finesub-share.service /etc/systemd/system/
   sudo systemctl enable --now finesub-share
   ```

5. Caddy 反代 + 自动 TLS（`Caddyfile`，改成你的域名）。⚠ 服务只接受以 localhost 相称的请求
   （防 DNS 重绑定），模板里的 `header_up Host {upstream_hostport}` 就是为此，别删：

   ```bash
   sudo cp Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy
   ```

6. litestream 把 SQLite 连续复制到对象存储（`litestream.yml`，填桶与密钥）：

   ```bash
   sudo cp litestream.yml /etc/litestream.yml
   sudo systemctl enable --now litestream
   ```

## 过渡：本机 + Cloudflare Tunnel（与终态同构）

服务照常本机起（同一条命令、同一个 `--root`）；对外用：

```bash
cloudflared tunnel --url http://127.0.0.1:8787 --http-host-header 127.0.0.1:8787
```

（`--http-host-header` 与 Caddyfile 的 `header_up Host` 同理：服务只接受以 localhost 相称的请求。）

免费、无需公网 IP；迁移到 VPS = 拷 `--root` 目录 + 同一份 systemd 单元。
（或 Oracle/GCP 免费档 VM 上直接走终态步骤。）

## 换地址与搬家

服务端在 `--root` 里铸一个 **`corpus_id`** 当库的身份（`GET /identity` 返回它，哈希链以它为种子）。
客户端的 token、防回滚锚点、三方合并基线与两本台账**全都以它为键，不以 URL 为键**，所以：

- 换域名、从 `*.workers.dev` 搬到 VPS、加一个只读镜像，客户端的本地状态照旧认得出是同一个库；
  只是**第一次指向新地址要加 `--accept-move`**（`share pull --remote <新地址> --accept-move`），
  之后照常。
- ⚠ 那道闸门不是形式：`corpus_id` 写在每份公开 snapshot 里，谁都抄得到，所以**知道它不等于有权
  代表它**。没有 `--accept-move` 这一步，一个抄了 id 的镜像就能拿到真服务端发给贡献者的 token。
  所以**换地址时要主动告知用户**，而不是指望他们照旧敲命令。
- 同一个地址若开始报另一个 `corpus_id`，客户端**同样停下来要 `--accept-move`**：用户并没有要求
  这次更换，照单全收等于让下一次 pull 把陌生人的库并进本地。授权之后它按**另一个库**处理，原来
  那个库的 token 不会跟着地址走。

⚠ 前提是**搬走的是 `--root` 里那个库文件本身**（连同 `corpus_id` 与整条链）。拿 `GET /snapshot`
重建一个新库 = 新身份 + 从 1 开始的 rev，所有客户端都会按防回滚拒绝拉取。

## 运维要点

- **滥用防护分两层**：应用层内建硬配额（每贡献者 pending ≤5、全局 pending ≤200、注册
  每天 ≤20、总数 ≤500、注册 7 天后仍没有待审或已批准投稿的 token 回收、pending 30 天自动过期——
  常量在 `share/server.py` 顶部，改了重启生效）；代理层 IP 限速见 Caddyfile（需 caddy-ratelimit
  插件）。⚠ **公开部署前代理层限速是必做**：应用层配额是全局的，一个人每天先把 20 个注册名额
  用完，别人当天就注册不了，只有按 IP 限速挡得住。
- **维护者端点**（`/queue`、`/verdict`、`/digest` 等）同样经公网暴露、只靠口令保护，因为日常
  `share review --remote` 就是从维护者本机连过来的。想收窄：在 Caddyfile 里把这些路径限到你自己的
  IP，或只经 SSH 隧道访问。
- `GET /snapshot` 以链头为 ETag，客户端带 `If-None-Match` 时库没变就回 304；同一链头的快照只构建一次。

- **备份/恢复**：真相只有 `--root` 下的 `share-server.sqlite`（含队列、贡献者、哈希链）。
  litestream 之外，凡本地手动备份直接拷文件即可（服务停下或用 sqlite3 `.backup`）。
  恢复 = 放回文件重启。哈希链带防回滚：**不要**用旧备份覆盖新库后继续服务——客户端
  会因 server_rev 倒退拒绝拉取；确需回滚要同时通知所有客户端清锚点（meta 里
  `share:<remote>:server_rev`/`chain_hash` 两键）。
- **token**：contributor token 匿名自助（`POST /register`），库里只存它的 SHA-256（贡献者的身份就是
  这个哈希），拿到库文件或备份的人也冒充不了贡献者；maintainer token 泄露 = 队列可被清空，换 token
  只需改口令文件重启（已有队列数据不受影响）。
- **GitHub snapshot 镜像**（可选，未内建）：定期把 `GET /snapshot` 的 JSON 推到公开
  仓库当匿名拉取源与异地备份；客户端验的是哈希链，不信任镜像本身。
- 升级：换包重启即可；store schema 迁移在进程启动时就地执行。
