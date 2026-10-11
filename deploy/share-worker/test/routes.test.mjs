// Contract tests for the Worker's routes, driven end to end through `fetch`.

import assert from "node:assert/strict";
import { test } from "node:test";

import { MAINTAINER_TOKEN, sha256Hex, startWorker } from "./harness.mjs";

test("a push travels register -> inbox -> relay -> verdict -> status", async () => {
  const w = startWorker();
  const author = await w.register();
  const stranger = await w.register();
  assert.match(author, /^[A-Za-z0-9_-]{32}$/);
  const bundle = { idempotency_key: "k1", entries: [{ text: "東京" }] };

  const pushed = await w.call("POST", "/push", { token: author, body: bundle });
  assert.deepEqual(pushed, { status: 200, body: { queue_id: 1, status: "pending", duplicate: false } });
  // Same key, same author: the first body stands and the id comes back.
  const again = await w.call("POST", "/push", { token: author, body: { ...bundle, swapped: true } });
  assert.deepEqual(again.body, { queue_id: 1, status: "pending", duplicate: true });
  assert.equal((await w.call("POST", "/push", { token: stranger, body: bundle })).status, 409);

  // Only the author and the maintainer may read an item.
  assert.equal((await w.call("GET", "/push/1", { token: author })).body.status, "pending");
  assert.equal((await w.call("GET", "/push/1", { token: stranger })).status, 403);
  assert.equal((await w.call("GET", "/push/1")).status, 403);
  assert.equal((await w.call("GET", "/push/1", w.asMaintainer())).status, 200);
  assert.equal((await w.call("GET", "/push/99", { token: author })).status, 404);

  // The relay sees the bundle as pushed and the author only as a hash.
  const inbox = (await w.call("GET", "/inbox", w.asMaintainer())).body.items;
  assert.equal(inbox.length, 1);
  assert.equal(inbox[0].contributor, await sha256Hex(author));
  assert.deepEqual(inbox[0].bundle, bundle);

  // Handed to the maintainer's queue: off /inbox, onto /tracked.
  const relay = (result) => w.call("POST", "/outbox", w.asMaintainer({ body: { results: [result] } }));
  assert.deepEqual((await relay({ inbox_id: 1, status: "pending", queue_id: 9 })).body, { updated: 1 });
  assert.deepEqual((await w.call("GET", "/inbox", w.asMaintainer())).body.items, []);
  assert.deepEqual((await w.call("GET", "/tracked", w.asMaintainer())).body.items, [{ inbox_id: 1, queue_id: 9 }]);

  await relay({ inbox_id: 1, status: "approved", verdict: "looks right", queue_id: 9, assigned: { h1: "kb-42" } });
  assert.deepEqual((await w.call("GET", "/push/1", { token: author })).body, {
    queue_id: 1,
    status: "approved",
    verdict: "looks right",
    assigned: { h1: "kb-42" },
  });
  assert.deepEqual((await w.call("GET", "/tracked", w.asMaintainer())).body.items, []);
});

test("a contributor holds at most five pending pushes", async () => {
  const w = startWorker();
  const token = await w.register();
  for (let i = 0; i < 5; i += 1) {
    assert.equal((await w.call("POST", "/push", { token, body: { idempotency_key: `k${i}` } })).status, 200);
  }
  const sixth = await w.call("POST", "/push", { token, body: { idempotency_key: "k5" } });
  assert.equal(sixth.status, 429);
});

test("a published snapshot comes back byte for byte, across chunks", async () => {
  const w = startWorker();
  assert.equal((await w.call("GET", "/identity")).status, 503);
  assert.equal((await w.call("GET", "/snapshot")).status, 503);

  const publish = (query, body, options = w.asMaintainer()) =>
    w.call("POST", `/publish?${query}`, { ...options, body });
  assert.equal((await publish("server_rev=1", "{}")).status, 400); // no corpus_id
  assert.equal((await publish("corpus_id=c", "{}", {})).status, 403);

  // Several chunks of mixed widths: ASCII, CJK, astral CJK, emoji, accents.
  const big = JSON.stringify({ entries: Array.from({ length: 100_000 }, (_, i) => `${i} 東京𠀋😀é`) });
  assert.equal((await publish("corpus_id=corp&server_rev=7&chain_hash=head7", big)).status, 200);
  const chunks = w.inbox.database("share").prepare("SELECT COUNT(*) AS n FROM snapshot_chunks").get().n;
  assert.ok(chunks >= 3, `expected several chunks, got ${chunks}`);
  assert.deepEqual((await w.call("GET", "/identity")).body, { corpus_id: "corp", server_rev: 7, chain_hash: "head7" });
  const snapshot = await w.send("GET", "/snapshot");
  assert.equal(snapshot.headers.get("ETag"), '"head7"');
  assert.ok(Buffer.from(await snapshot.arrayBuffer()).equals(Buffer.from(big, "utf8")));

  // A smaller republish leaves none of the old chunks behind.
  assert.equal((await publish("corpus_id=corp&server_rev=8&chain_hash=head8", '{"entries":[]}')).status, 200);
  assert.equal(await (await w.send("GET", "/snapshot")).text(), '{"entries":[]}');
});

test("maintainer routes refuse anyone without the maintainer token", async () => {
  const w = startWorker();
  const contributor = await w.register();
  const sameLength = "x".repeat(MAINTAINER_TOKEN.length);
  const routes = [
    ["GET", "/inbox"],
    ["GET", "/tracked"],
    ["POST", "/outbox", { results: [] }],
    ["POST", "/publish?corpus_id=c", "{}"],
  ];
  for (const [method, path, body] of routes) {
    for (const maintainer of [undefined, "", sameLength, contributor]) {
      const response = await w.call(method, path, { maintainer, body });
      assert.equal(response.status, 403, `${method} ${path} with ${JSON.stringify(maintainer)}`);
    }
  }
  assert.equal((await w.call("POST", "/push", { body: { idempotency_key: "k" } })).status, 401);
  assert.equal((await w.call("POST", "/push", { token: "made-up", body: { idempotency_key: "k" } })).status, 401);
  assert.equal((await w.call("GET", "/nowhere")).status, 404);

  // An unset secret is not an empty password.
  const unset = startWorker({ maintainerToken: null });
  assert.equal((await unset.call("GET", "/inbox", { maintainer: "" })).status, 403);
});

test("plaintext tokens from an older deployment are hashed before the first request", async () => {
  // One-way (README: a rollback invalidates every registered token), and it
  // must finish before any call is served or the first push would see 401.
  const w = startWorker();
  await w.call("GET", "/identity"); // constructs the object, which migrates an empty table
  const db = w.inbox.database("share");
  const stamp = new Date().toISOString();
  db.exec("DELETE FROM migrations");
  db.prepare("INSERT INTO contributors(token, created_at) VALUES (?, ?)").run("legacy-token", stamp);
  db.prepare("INSERT INTO inbox(idempotency_key, contributor, bundle, received_at) VALUES (?, ?, ?, ?)").run(
    "old",
    "legacy-token",
    "{}",
    stamp,
  );
  w.inbox.restart();

  assert.equal((await w.call("GET", "/push/1", { token: "legacy-token" })).body.status, "pending");
  assert.equal((await w.call("POST", "/push", { token: "legacy-token", body: { idempotency_key: "new" } })).status, 200);
  const stored = db.prepare("SELECT token FROM contributors").all().map((row) => row.token);
  assert.deepEqual(stored, [await sha256Hex("legacy-token")]);
});
