// Regression tests for the six defects the 2026-09-09 review found in
// src/index.js (fixed in d32a7e22 and 9ac1a874, tightened in 90e7a974). Each
// test was checked to go red with its defect put back.

import assert from "node:assert/strict";
import { test } from "node:test";

import { startWorker, useClock } from "./harness.mjs";

const MAX_PUSH_BYTES = 1_500_000; // mirrors src/index.js

test("the push limit counts UTF-8 bytes, not UTF-16 code units", async () => {
  // A CJK character is one code unit and three bytes, so a body measured by
  // `text.length` passed at three times the limit -- and the maintainer's
  // machine (2 MB) then refused it at relay time instead of at push time.
  const w = startWorker();
  const token = await w.register();
  const bundle = (key, chars) => JSON.stringify({ idempotency_key: key, text: "知".repeat(chars) });

  const over = bundle("over", 510_000);
  assert.ok(over.length < MAX_PUSH_BYTES && Buffer.byteLength(over) > MAX_PUSH_BYTES);
  // Without Content-Length (a chunked upload) the bytes are the only thing to measure.
  for (const contentLength of [false, true]) {
    const response = await w.call("POST", "/push", { token, body: over, contentLength });
    assert.equal(response.status, 413, `contentLength=${contentLength}`);
  }

  const under = bundle("under", 490_000);
  assert.ok(Buffer.byteLength(under) < MAX_PUSH_BYTES);
  const accepted = await w.call("POST", "/push", { token, body: under, contentLength: false });
  assert.equal(accepted.status, 200);
});

test("snapshot chunking never splits a surrogate pair", async () => {
  // CJK Extension B lives above U+FFFF. A chunk cut between the two halves of
  // a pair stores a lone surrogate, which cannot be UTF-8: the snapshot read
  // back differs from the one published and every client's verify fails.
  // Both parities, so a boundary lands on a pair whatever the chunk size.
  const w = startWorker();
  for (const text of ["a" + "𠀋".repeat(600_000), "𠀋".repeat(600_000)]) {
    const published = await w.send("POST", "/publish?corpus_id=c&server_rev=1&chain_hash=h", w.asMaintainer({ body: text }));
    assert.equal(published.status, 200);
    const chunks = w.inbox.database("share").prepare("SELECT COUNT(*) AS n FROM snapshot_chunks").get().n;
    assert.ok(chunks > 1, "the snapshot must span a chunk boundary for this test to mean anything");

    const response = await w.send("GET", "/snapshot");
    assert.equal(response.status, 200);
    assert.ok(Buffer.from(await response.arrayBuffer()).equals(Buffer.from(text, "utf8")));
  }
});

test("expiry leaves items alone once they are relayed", async (t) => {
  // The mailbox counts 30 days from arrival, the maintainer's queue from
  // relaying. An item expired here while alive there drops out of /tracked,
  // so a verdict reached in the gap could never be carried back.
  const advanceDays = useClock(t);
  const w = startWorker();
  const author = await w.register();
  const other = await w.register();
  const relayed = (await w.call("POST", "/push", { token: author, body: { idempotency_key: "relayed" } })).body.queue_id;
  const neverRelayed = (await w.call("POST", "/push", { token: author, body: { idempotency_key: "left" } })).body
    .queue_id;
  advanceDays(1);
  const handedOver = await w.call(
    "POST",
    "/outbox",
    w.asMaintainer({ body: { results: [{ inbox_id: relayed, status: "pending", queue_id: 70 }] } }),
  );
  assert.equal(handedOver.status, 200);

  advanceDays(30);
  // Expiry runs on push.
  assert.equal((await w.call("POST", "/push", { token: other, body: { idempotency_key: "trigger" } })).status, 200);

  const status = async (id) => (await w.call("GET", `/push/${id}`, { token: author })).body.status;
  assert.equal(await status(neverRelayed), "expired");
  assert.equal(await status(relayed), "pending");
  assert.deepEqual((await w.call("GET", "/tracked", w.asMaintainer())).body.items, [
    { inbox_id: relayed, queue_id: 70 },
  ]);

  const verdict = { inbox_id: relayed, status: "approved", verdict: "ok", queue_id: 70, assigned: { h1: "c1" } };
  await w.call("POST", "/outbox", w.asMaintainer({ body: { results: [verdict] } }));
  const final = (await w.call("GET", `/push/${relayed}`, { token: author })).body;
  assert.equal(final.status, "approved");
  assert.deepEqual(final.assigned, { h1: "c1" });
});

test("one loop cannot use up registration", async (t) => {
  // The address ships with the package, so a `for` loop used to take all 500
  // seats for good. Now: 3 a day per address, 20 a day in all.
  const advanceDays = useClock(t);
  const w = startWorker();
  const fromHere = () => w.call("POST", "/register", { ip: "192.0.2.1" });

  for (let i = 0; i < 3; i += 1) assert.equal((await fromHere()).status, 200);
  const sameAddress = await fromHere();
  assert.equal(sameAddress.status, 429);
  assert.match(sameAddress.body.error, /this address/);

  for (let i = 0; i < 17; i += 1) await w.register();
  const daily = await w.call("POST", "/register", { ip: "192.0.2.99" });
  assert.equal(daily.status, 429);
  assert.match(daily.body.error, /today/);

  advanceDays(1.05);
  assert.equal((await fromHere()).status, 200);
});

test("a malformed or rejected push does not keep a token's seat", async (t) => {
  // An earlier fix stamped `last_used` before the bundle was parsed, so one
  // garbage push made a token permanent; the next one counted any inbox row,
  // so one well-formed junk push did. Only a pending or approved push holds a
  // seat past UNUSED_TOKEN_DAYS (7).
  const advanceDays = useClock(t);
  const w = startWorker();
  const garbage = await w.register();
  const rejected = await w.register();
  const pending = await w.register();

  assert.equal((await w.call("POST", "/push", { token: garbage, body: "not json" })).status, 400);
  assert.equal((await w.call("POST", "/push", { token: garbage, body: {} })).status, 400);
  const junk = (await w.call("POST", "/push", { token: rejected, body: { idempotency_key: "junk" } })).body.queue_id;
  await w.call(
    "POST",
    "/outbox",
    w.asMaintainer({ body: { results: [{ inbox_id: junk, status: "rejected", verdict: "spam" }] } }),
  );
  assert.equal((await w.call("POST", "/push", { token: pending, body: { idempotency_key: "real" } })).status, 200);

  advanceDays(8);
  await w.register(); // reclaiming runs on registration

  const push = (token, key) => w.call("POST", "/push", { token, body: { idempotency_key: key } });
  assert.equal((await push(garbage, "g2")).status, 401);
  assert.equal((await push(rejected, "r2")).status, 401);
  assert.equal((await push(pending, "p2")).status, 200);
});

test("a conditional snapshot request costs one small RPC and is answered 304", async () => {
  // The ETag branch first shipped with no client sending If-None-Match, so it
  // only added an RPC to every pull. The client now sends its anti-rollback
  // anchor as `"<chain_hash>"` (client.fetch_snapshot; the format is pinned on
  // the Python side by test_llm_knowledge_share_worker.py), and a match must
  // never reassemble the corpus.
  const w = startWorker();
  await w.send("POST", "/publish?corpus_id=c&server_rev=3&chain_hash=abc123", w.asMaintainer({ body: '{"x":1}' }));

  w.inbox.calls.length = 0;
  const unchanged = await w.send("GET", "/snapshot", { headers: { "If-None-Match": '"abc123"' } });
  assert.equal(unchanged.status, 304);
  assert.equal(await unchanged.text(), "");
  assert.equal(unchanged.headers.get("ETag"), '"abc123"');
  // no-cache, not max-age: an intermediary serving a stale copy would trip the
  // client's server_rev rollback check.
  assert.equal(unchanged.headers.get("Cache-Control"), "no-cache");
  assert.deepEqual(w.rpcNames(), ["identity"]);

  for (const headers of [{}, { "If-None-Match": '"older"' }]) {
    w.inbox.calls.length = 0;
    const full = await w.send("GET", "/snapshot", { headers });
    assert.equal(full.status, 200);
    assert.equal(await full.text(), '{"x":1}');
    assert.deepEqual(w.rpcNames(), ["identity", "snapshotText"]);
  }
});

