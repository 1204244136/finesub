// 共享知识库的「信箱 + 快照 CDN」（方案 B，取舍见同目录 README.md）。
//
// 这个 Worker 刻意**不理解 bundle 语义**：它不清洗、不校验 anchors、不判门槛、
// 不 parse 快照。它只做三件事——收 push 存进信箱、按 token 回答状态、把维护者
// 发布的快照原样发出去。审核、apply、发链仍然在维护者本机（key 从不上服务器）。
//
// 客户端本来就把服务端当**不可信**（拉取侧逐项验哈希链再走结构准入，见
// share/exchange.py 的 snapshot_content_problems），把服务端做薄只是缩小攻击面。
// 发布权则不在这里：快照由维护者本机签名（share/signing.py），拿到 MAINTAINER_TOKEN
// 的人能发布，却发不出客户端会收的快照。
//
// 贡献者 token 只以 SHA-256 进 Durable Object：哈希在下面的 route() 里算，对象本身
// 从不见到明文，所以库、/inbox 与本机 relay 都只拿得到「这是谁」，冒充不了谁。
//
// 对应的本机一侧是 `python -m finesub.llm.knowledge.share relay`：它把信箱里的条目
// 喂给本机 ShareService.push()，于是清洗、配额、幂等内容校验、§6.3 门槛全部在本机
// 原样发生。
import { DurableObject } from "cloudflare:workers";

// 与 share/server.py 顶部同源的滥用边界。⚠ 改这里要同时改那边：本机 relay 会把
// 同一批内容再过一次本机配额，两边不一致时以更严的那边为准，症状是信箱收下了、
// 本机却拒绝。
const MAX_CONTRIBUTORS = 500;
// ⚠ 总额 500 是防表膨胀，**不是**防刷：地址随包发布，人人有，一条 for 循环就能把它
// 占满，之后新人永远只看到 429（reviewer 2026-09-09）。所以另加两道——每天新注册的
// 窗口，以及回收「从没用过、且发出去有些天了」的 token。真正的速率限制仍应加在
// Cloudflare 边缘；这两道是应用层的底。
const MAX_REGISTRATIONS_PER_DAY = 20;
// 全局日额挡不住「一个人每天先把 20 个名额用完」，所以同一 IP 另有自己的日额。IP 只以
// 哈希存，一天后删。⚠ 这一道**只在信箱这边有**，share/server.py 没有、也不该照抄：自托管
// 服务端常跑在 Tunnel 或反代后面，看到的来源地址是那一跳（所有人同一个），而转发头又能伪造。
// 那边按 IP 限的是反代（deploy/share-server/Caddyfile 的 rate_limit）。这里的 CF-Connecting-IP
// 是 Cloudflare 边缘写的，可信。
const MAX_REGISTRATIONS_PER_IP_PER_DAY = 3;
const UNUSED_TOKEN_DAYS = 7;
const MAX_PENDING_PER_CONTRIBUTOR = 5;
const MAX_PENDING_TOTAL = 200;
const PENDING_EXPIRY_DAYS = 30;
// 被拒 / 过期的信箱行留这么久（贡献者查得到结论），之后整行删掉。已批准的行不删：
// 贡献者的 `share status` 要从它取 handle→canonical 映射。
const PURGE_SETTLED_DAYS = 90;

// body 上限，按**字节**量（readText 直接量 arrayBuffer 的 byteLength）。⚠ 别用 `text.length`：那是 UTF-16 码元数，
// 而这个知识库的正文以中日文为主，一个字符 3 字节——量错会让信箱收下本机
// (`_Handler.MAX_BODY_BYTES = 2_000_000`) 会拒的东西，贡献者要等到维护者中继之后才收到
// 413，而不是投稿当场被拒。
//
// bundle 压到 1.5MB 而不是与本机同数：投稿整条存在 inbox 的一行里，而 DO 的 SQLite
// 单行上限是 2MB——超了 INSERT 抛的是没有线索的 500，不是 413。快照走切片所以不受这条限制。
const MAX_PUSH_BYTES = 1_500_000;
const MAX_SNAPSHOT_BYTES = 24 * 1024 * 1024;
// 一行 SQLite 有 2MB 上限，快照因此切片存；取值留足余量。
const SNAPSHOT_CHUNK = 512 * 1024;

/** Where to cut `text` so the chunk ends on a whole character.
 *
 * `slice` indexes UTF-16 code units, so a boundary can land between the two
 * halves of a surrogate pair — and CJK Extension B (U+20000+), which is where
 * rarer Japanese and Chinese name characters live, is exactly that. A lone
 * surrogate cannot be encoded as UTF-8: SQLite's TEXT side either throws or
 * substitutes U+FFFD, and the substitution is worse — the reassembled snapshot
 * no longer matches its content digest, so `verify_snapshot` fails for every
 * client at once with nothing pointing at the chunking (reviewer, 2026-09-09).
 */
function chunkEnd(text, start, size) {
  const end = Math.min(start + size, text.length);
  if (end < text.length) {
    const code = text.charCodeAt(end - 1);
    // Backing off one unit must still make progress — the caller assigns
    // `i = end` inside transactionSync, so returning `start` would spin until
    // the CPU limit kills the request. Unreachable at any sane chunk size;
    // the guard costs nothing and the failure mode does not.
    if (code >= 0xd800 && code <= 0xdbff && end - 1 > start) return end - 1;
  }
  return end;
}

const DDL = [
  `CREATE TABLE IF NOT EXISTS contributors(
     token TEXT PRIMARY KEY,
     created_at TEXT NOT NULL)`,
  `CREATE TABLE IF NOT EXISTS inbox(
     inbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
     idempotency_key TEXT NOT NULL UNIQUE,
     contributor TEXT NOT NULL,
     bundle TEXT NOT NULL,
     received_at TEXT NOT NULL,
     status TEXT NOT NULL DEFAULT 'pending',
     verdict TEXT NOT NULL DEFAULT '',
     assigned TEXT NOT NULL DEFAULT '{}',
     queue_id INTEGER,
     relayed_at TEXT NOT NULL DEFAULT '')`,
  `CREATE TABLE IF NOT EXISTS snapshot_meta(k TEXT PRIMARY KEY, v TEXT NOT NULL)`,
  `CREATE TABLE IF NOT EXISTS snapshot_chunks(idx INTEGER PRIMARY KEY, body TEXT NOT NULL)`,
  `CREATE TABLE IF NOT EXISTS registrations(ip TEXT NOT NULL, at TEXT NOT NULL)`,
  `CREATE TABLE IF NOT EXISTS migrations(name TEXT PRIMARY KEY)`,
];

const daysAgoIso = (days) =>
  new Date(Date.now() - days * 86400 * 1000).toISOString().replace(/\.\d+Z$/, "Z");

async function sha256Hex(text) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Constant-time: `===` on a secret leaks how long the matching prefix is. */
function sameSecret(given, expected) {
  if (!given || !expected) return false;
  const a = new TextEncoder().encode(given);
  const b = new TextEncoder().encode(expected);
  return a.byteLength === b.byteLength && crypto.subtle.timingSafeEqual(a, b);
}

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

const nowIso = () => new Date().toISOString().replace(/\.\d+Z$/, "Z");

function randomToken() {
  const bytes = crypto.getRandomValues(new Uint8Array(24));
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export class ShareInbox extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    this.sql = ctx.storage.sql;
    for (const statement of DDL) this.sql.exec(statement);
    ctx.blockConcurrencyWhile(() => this.hashStoredTokens());
  }

  /** Tokens stored before contributor ids were hashes: hash them (and every
   * inbox row's reference) once, before the first request is served. */
  async hashStoredTokens() {
    const done = this.sql.exec("SELECT 1 FROM migrations WHERE name='contributor_ids'").toArray();
    if (done.length) return;
    const tokens = this.sql.exec("SELECT token FROM contributors").toArray().map((r) => r.token);
    const referenced = this.sql
      .exec("SELECT DISTINCT contributor FROM inbox")
      .toArray()
      .map((r) => r.contributor);
    const hashed = new Map();
    for (const token of new Set([...tokens, ...referenced])) hashed.set(token, await sha256Hex(token));
    this.ctx.storage.transactionSync(() => {
      for (const [token, id] of hashed) {
        this.sql.exec("UPDATE contributors SET token=? WHERE token=?", id, token);
        this.sql.exec("UPDATE inbox SET contributor=? WHERE contributor=?", id, token);
      }
      this.sql.exec("INSERT INTO migrations(name) VALUES ('contributor_ids')");
    });
  }

  /** The only entry point the Worker calls.
   *
   * Custom error classes do not survive the RPC boundary — a thrown HttpError
   * would reach the Worker as a generic Error with its status stripped, and
   * every refusal would turn into a 500. So failures come back as data.
   */
  call(name, args) {
    try {
      return { ok: true, value: this[name](...args) };
    } catch (error) {
      return { ok: false, status: Number(error?.status) || 500, error: String(error?.message || error) };
    }
  }

  // ---- contributors -------------------------------------------------

  /** Drop tokens that were handed out and never got anything worth keeping.
   *
   * A pending or approved push holds the seat; rejected and expired ones do
   * not — otherwise one well-formed junk push would make a token permanent,
   * and 20 a day for 25 days would fill every slot (the failure an earlier
   * `last_used` stamp had with malformed pushes, reviewer 2026-09-09, came back
   * through that door). Never touches a token younger than UNUSED_TOKEN_DAYS.
   */
  reclaimUnusedTokens() {
    this.sql.exec(
      "DELETE FROM contributors WHERE created_at < ?" +
        " AND token NOT IN (SELECT contributor FROM inbox WHERE status IN ('pending', 'approved'))",
      daysAgoIso(UNUSED_TOKEN_DAYS),
    );
  }

  /** `id` is the new token's hash (route() mints the token), `ip` the
   * caller's hashed address. */
  register(id, ip) {
    this.reclaimUnusedTokens();
    const since = daysAgoIso(1);
    this.sql.exec("DELETE FROM registrations WHERE at < ?", since);
    const fromHere = this.sql
      .exec("SELECT COUNT(*) AS n FROM registrations WHERE ip=?", ip)
      .one().n;
    if (fromHere >= MAX_REGISTRATIONS_PER_IP_PER_DAY) {
      throw new HttpError(429, "too many registrations from this address today; try again tomorrow");
    }
    const today = this.sql
      .exec("SELECT COUNT(*) AS n FROM contributors WHERE created_at >= ?", since)
      .one().n;
    if (today >= MAX_REGISTRATIONS_PER_DAY) {
      throw new HttpError(429, "too many registrations today; try again tomorrow");
    }
    const { n } = this.sql.exec("SELECT COUNT(*) AS n FROM contributors").one();
    if (n >= MAX_CONTRIBUTORS) {
      throw new HttpError(429, "contributor registrations are full; contact the maintainer");
    }
    this.sql.exec("INSERT INTO contributors(token, created_at) VALUES (?, ?)", id, nowIso());
    this.sql.exec("INSERT INTO registrations(ip, at) VALUES (?, ?)", ip, nowIso());
    return { registered: true };
  }

  /** `id` is the presented token's hash ("" when none was presented). */
  requireContributor(id) {
    if (!id) throw new HttpError(401, "missing contributor token");
    const rows = this.sql.exec("SELECT token FROM contributors WHERE token=?", id).toArray();
    if (rows.length === 0) throw new HttpError(401, "unknown contributor token");
    return id;
  }

  // ---- push ----------------------------------------------------------

  /** Age out submissions nobody ever collected.
   *
   * `relayed_at=''` is the whole condition: once an item has been handed to
   * the maintainer's machine its fate belongs to the local queue, and the two
   * sides count from different moments (here: arrival; there: relaying). An
   * item expired on this side while alive on that one drops out of `tracked()`
   * forever, so a verdict reached in the gap — including an approval whose
   * content already entered the corpus — can never be carried back: the
   * contributor sees "expired", the handle→canonical map is lost, and their
   * next pull rebuilds the entry as if it were new (reviewer, 2026-09-09).
   */
  expireStale() {
    const cutoff = new Date(Date.now() - PENDING_EXPIRY_DAYS * 86400 * 1000)
      .toISOString()
      .replace(/\.\d+Z$/, "Z");
    this.sql.exec(
      "UPDATE inbox SET status='expired'" +
        " WHERE status='pending' AND relayed_at='' AND received_at < ?",
      cutoff,
    );
  }

  checkQuotas(contributor) {
    const mine = this.sql
      .exec("SELECT COUNT(*) AS n FROM inbox WHERE contributor=? AND status='pending'", contributor)
      .one().n;
    if (mine >= MAX_PENDING_PER_CONTRIBUTOR) {
      throw new HttpError(429, "you already have the maximum number of pending pushes");
    }
    const total = this.sql.exec("SELECT COUNT(*) AS n FROM inbox WHERE status='pending'").one().n;
    if (total >= MAX_PENDING_TOTAL) {
      throw new HttpError(429, "the review queue is full; try again later");
    }
  }

  push(id, bundleText) {
    const contributor = this.requireContributor(id);
    let key = "";
    try {
      key = String(JSON.parse(bundleText)?.idempotency_key ?? "");
    } catch {
      throw new HttpError(400, "bundle is not valid JSON");
    }
    // The one field the mailbox reads. Everything else about the bundle —
    // shape, tags, anchors, thresholds — is the maintainer's machine to judge.
    if (!key || key.length > 200) throw new HttpError(400, "missing or oversized idempotency_key");
    this.expireStale();
    const existing = this.sql
      .exec("SELECT inbox_id, contributor, status FROM inbox WHERE idempotency_key=?", key)
      .toArray();
    if (existing.length > 0) {
      const row = existing[0];
      if (row.contributor !== contributor) {
        throw new HttpError(409, "idempotency key already used by a different contributor");
      }
      // Same key, same contributor: the first body wins and is never
      // overwritten. A client whose content really changed mints a new key
      // (cli._cmd_push looks the key up by content digest), so "same key,
      // different content" is either a bug or an attempt to swap a queued
      // bundle out from under the maintainer.
      if (row.status === "expired") {
        this.checkQuotas(contributor);
        this.sql.exec(
          "UPDATE inbox SET status='pending', received_at=?, relayed_at='' WHERE inbox_id=?",
          nowIso(),
          row.inbox_id,
        );
        return { queue_id: row.inbox_id, status: "pending", revived: true };
      }
      return { queue_id: row.inbox_id, status: row.status, duplicate: true };
    }
    this.checkQuotas(contributor);
    this.sql.exec(
      "INSERT INTO inbox(idempotency_key, contributor, bundle, received_at) VALUES (?, ?, ?, ?)",
      key,
      contributor,
      bundleText,
      nowIso(),
    );
    const inboxId = this.sql.exec("SELECT last_insert_rowid() AS id").one().id;
    return { queue_id: inboxId, status: "pending", duplicate: false };
  }

  pushStatus(inboxId, id, maintainerOk) {
    const rows = this.sql
      .exec(
        "SELECT inbox_id, status, verdict, assigned, contributor FROM inbox WHERE inbox_id=?",
        inboxId,
      )
      .toArray();
    if (rows.length === 0) throw new HttpError(404, `unknown queue item ${inboxId}`);
    const row = rows[0];
    // Ids are guessable; the verdict and the handle→canonical map belong to
    // whoever pushed it (server.py:push_status keeps the same rule).
    if (!maintainerOk && (!id || id !== row.contributor)) {
      throw new HttpError(403, "this queue item belongs to a different contributor");
    }
    return {
      queue_id: row.inbox_id,
      status: row.status,
      verdict: row.verdict,
      assigned: JSON.parse(row.assigned || "{}"),
    };
  }

  // ---- maintainer ----------------------------------------------------

  inbox(after, limit) {
    const rows = this.sql
      .exec(
        "SELECT inbox_id, contributor, bundle, received_at FROM inbox" +
          " WHERE status='pending' AND relayed_at='' AND inbox_id > ?" +
          " ORDER BY inbox_id LIMIT ?",
        after,
        limit,
      )
      .toArray();
    return {
      items: rows.map((row) => ({
        inbox_id: row.inbox_id,
        contributor: row.contributor,
        received_at: row.received_at,
        bundle: JSON.parse(row.bundle),
      })),
    };
  }

  /** Items already handed to the maintainer's machine whose fate is still
   * open. The relay re-asks the local queue about each one, so an approval
   * that happens hours later still reaches the contributor's `share status`.
   *
   * Paged from `after`: these come back oldest-first, so without a cursor a
   * long-undecided head of the queue would hide everything behind it forever.
   */
  tracked(after, limit) {
    const rows = this.sql
      .exec(
        "SELECT inbox_id, queue_id FROM inbox" +
          " WHERE queue_id IS NOT NULL AND status='pending' AND inbox_id > ?" +
          " ORDER BY inbox_id LIMIT ?",
        after,
        limit,
      )
      .toArray();
    return { items: rows.map((row) => ({ inbox_id: row.inbox_id, queue_id: row.queue_id })) };
  }

  outbox(results) {
    let updated = 0;
    for (const result of results) {
      const inboxId = Number(result.inbox_id);
      if (!Number.isInteger(inboxId)) throw new HttpError(400, "outbox entry without inbox_id");
      const status = String(result.status || "pending");
      // A decided item's body has a copy on the maintainer's machine and no
      // reader here (only an expired one can be revived, and it keeps its
      // body), so it stops taking up to 1.5 MB a row.
      this.sql.exec(
        "UPDATE inbox SET status=?, verdict=?, assigned=?, queue_id=?, relayed_at=?," +
          " bundle=CASE WHEN ? IN ('approved', 'rejected') THEN '' ELSE bundle END" +
          " WHERE inbox_id=?",
        status,
        String(result.verdict || ""),
        JSON.stringify(result.assigned || {}),
        result.queue_id == null ? null : Number(result.queue_id),
        nowIso(),
        status,
        inboxId,
      );
      updated += 1;
    }
    this.purgeSettled();
    return { updated };
  }

  /** Rejected and expired rows go PURGE_SETTLED_DAYS after they were
   * decided; approved rows stay (their assignment map is the contributor's
   * to fetch). A rejection is dated by `relayed_at`, which the outbox stamps
   * when it carries the verdict back -- dating it by arrival would delete a
   * slowly reviewed item in the very call that records its rejection, before
   * the contributor could read why. An expired row never left, so it is
   * dated by arrival. */
  purgeSettled() {
    const cutoff = daysAgoIso(PURGE_SETTLED_DAYS);
    this.sql.exec(
      "DELETE FROM inbox WHERE (status='rejected' AND relayed_at <> '' AND relayed_at < ?)" +
        " OR (status='expired' AND received_at < ?)",
      cutoff,
      cutoff,
    );
  }

  publish(meta, snapshotText) {
    // The snapshot is stored verbatim and never parsed here: the client
    // verifies the hash chain, and this Worker has no opinion to add.
    this.ctx.storage.transactionSync(() => {
      this.sql.exec("DELETE FROM snapshot_chunks");
      for (let i = 0, idx = 0; i < snapshotText.length; idx += 1) {
        const end = chunkEnd(snapshotText, i, SNAPSHOT_CHUNK);
        this.sql.exec(
          "INSERT INTO snapshot_chunks(idx, body) VALUES (?, ?)",
          idx,
          snapshotText.slice(i, end),
        );
        i = end;
      }
      for (const [k, v] of Object.entries(meta)) {
        this.sql.exec("INSERT OR REPLACE INTO snapshot_meta(k, v) VALUES (?, ?)", k, String(v));
      }
      this.sql.exec(
        "INSERT OR REPLACE INTO snapshot_meta(k, v) VALUES ('published_at', ?)",
        nowIso(),
      );
    });
    return { published: true };
  }

  // ---- public reads ---------------------------------------------------

  identity() {
    const rows = this.sql.exec("SELECT k, v FROM snapshot_meta").toArray();
    const meta = Object.fromEntries(rows.map((row) => [row.k, row.v]));
    if (!meta.corpus_id) {
      throw new HttpError(503, "no snapshot published yet; the maintainer has not relayed one");
    }
    return {
      corpus_id: meta.corpus_id,
      server_rev: Number(meta.server_rev || 0),
      chain_hash: meta.chain_hash || "",
    };
  }

  snapshotText() {
    const rows = this.sql.exec("SELECT body FROM snapshot_chunks ORDER BY idx").toArray();
    if (rows.length === 0) {
      throw new HttpError(503, "no snapshot published yet; the maintainer has not relayed one");
    }
    return rows.map((row) => row.body).join("");
  }
}

const json = (status, body) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });

async function readText(request, limit) {
  const declared = Number(request.headers.get("Content-Length") || 0);
  if (declared > limit) throw new HttpError(413, `body exceeds ${limit} bytes`);
  // Measure the bytes we already have rather than re-encoding the decoded
  // string: /publish carries megabytes, and re-encoding would hold a second
  // copy of all of them at peak. Bytes, not code units — see MAX_PUSH_BYTES.
  const bytes = await request.arrayBuffer();
  if (bytes.byteLength > limit) throw new HttpError(413, `body exceeds ${limit} bytes`);
  return new TextDecoder("utf-8").decode(bytes);
}

async function route(request, env, stub) {
  const call = async (name, ...args) => {
    const result = await stub.call(name, args);
    if (!result.ok) throw new HttpError(result.status, result.error);
    return result.value;
  };
  const url = new URL(request.url);
  const path = url.pathname.replace(/\/+$/, "") || "/";
  const method = request.method;
  const shareToken = request.headers.get("X-Share-Token") || "";
  // The object only ever sees the token's hash (see the header comment).
  const contributorId = shareToken ? await sha256Hex(shareToken) : "";
  const maintainer = request.headers.get("X-Maintainer-Token") || "";
  const maintainerOk = sameSecret(maintainer, env.MAINTAINER_TOKEN || "");
  const requireMaintainer = () => {
    if (!maintainerOk) throw new HttpError(403, "maintainer token required");
  };

  if (method === "GET" && path === "/identity") return json(200, await call("identity"));
  if (method === "GET" && path === "/snapshot") {
    // The chain head IS a strong ETag: it commits to the whole content. So a
    // conditional request costs one small DO read instead of reassembling and
    // shipping the entire corpus — this is the one anonymous, unlimited,
    // size-proportional endpoint here.
    //
    // ⚠ This is conditional-request support, not edge caching: a Worker's
    // response is not cached at the edge unless it goes through the Cache API,
    // which this does not do. Reach for that if traffic ever justifies it.
    const head = await call("identity");
    const etag = `"${head.chain_hash}"`;
    const headers = {
      "Content-Type": "application/json; charset=utf-8",
      ETag: etag,
      // no-cache, NOT max-age: the client refuses a snapshot whose server_rev
      // went backwards (`sync.check_anchor`), so letting any intermediary hand
      // out a stale copy turns a normal pull into a hard error — two pulls in
      // the minute after a publish would do it. "Revalidate every time, but
      // trade an ETag for a 304" gives the whole benefit and permits nobody to
      // serve the past (reviewer, 2026-09-09).
      "Cache-Control": "no-cache",
    };
    if ((request.headers.get("If-None-Match") || "") === etag) {
      return new Response(null, { status: 304, headers });
    }
    return new Response(await call("snapshotText"), { headers });
  }
  if (method === "POST" && path === "/register") {
    const token = randomToken();
    const ip = await sha256Hex(`registration:${request.headers.get("CF-Connecting-IP") || ""}`);
    await call("register", await sha256Hex(token), ip);
    return json(200, { token });
  }
  if (method === "POST" && path === "/push") {
    return json(200, await call("push", contributorId, await readText(request, MAX_PUSH_BYTES)));
  }
  const pushStatus = path.match(/^\/push\/(\d+)$/);
  if (method === "GET" && pushStatus) {
    return json(200, await call("pushStatus", Number(pushStatus[1]), contributorId, maintainerOk));
  }
  if (method === "GET" && path === "/inbox") {
    requireMaintainer();
    const after = Number(url.searchParams.get("after") || 0);
    const limit = Math.min(Number(url.searchParams.get("limit") || 20), 50);
    return json(200, await call("inbox", after, limit));
  }
  if (method === "GET" && path === "/tracked") {
    requireMaintainer();
    return json(
      200,
      await call(
        "tracked",
        Number(url.searchParams.get("after") || 0),
        Math.min(Number(url.searchParams.get("limit") || 50), 200),
      ),
    );
  }
  if (method === "POST" && path === "/outbox") {
    requireMaintainer();
    const body = JSON.parse(await readText(request, MAX_PUSH_BYTES));
    return json(200, await call("outbox", body.results || []));
  }
  if (method === "POST" && path === "/publish") {
    requireMaintainer();
    // Metadata rides in the query string so the body stays the snapshot bytes
    // and this Worker never has to parse them.
    const meta = {
      corpus_id: url.searchParams.get("corpus_id") || "",
      server_rev: url.searchParams.get("server_rev") || "0",
      chain_hash: url.searchParams.get("chain_hash") || "",
    };
    if (!meta.corpus_id) throw new HttpError(400, "publish needs corpus_id");
    return json(200, await call("publish", meta, await readText(request, MAX_SNAPSHOT_BYTES)));
  }
  throw new HttpError(404, `no route ${method} ${path}`);
}

export default {
  async fetch(request, env) {
    const id = env.INBOX.idFromName("share");
    const stub = env.INBOX.get(id);
    try {
      return await route(request, env, stub);
    } catch (error) {
      const status = Number(error?.status) || 500;
      return json(status, { error: String(error?.message || error) });
    }
  },
};
