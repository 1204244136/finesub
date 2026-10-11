// Stand-in for the slice of workerd that src/index.js touches, so the Worker
// runs unchanged under `node --test`. hooks.mjs maps the `cloudflare:workers`
// specifier to this file; harness.mjs builds the env from it.
//
// A wrong shim hides bugs silently, so this is deliberately small and says
// exactly what it stands for.
//
// Reproduced (index.js relies on each):
// - `ctx.storage.sql.exec(query, ...bindings)` -> a single-pass cursor with
//   `toArray()`, `one()` and iteration. Bindings take workerd's types: string
//   -> TEXT, number -> REAL (an integral REAL stored into an INTEGER column
//   reads back as an integer, as in any SQLite), null -> NULL; anything else is
//   refused here rather than guessed at. Rows are plain objects; INTEGER
//   columns read back as JS numbers.
// - `one()` throws unless exactly one row remains.
// - A string that cannot be encoded as UTF-8 (a lone surrogate) is stored with
//   U+FFFD in its place. That is the silent one of workerd's two possible
//   outcomes (the other is throwing); either way the bytes read back differ
//   from the bytes written, which is the failure a test needs to see.
// - A bound value over MAX_VALUE_BYTES is refused with SQLite's TOOBIG message,
//   standing in for Durable Object SQLite's 2 MB string / BLOB / row limit
//   (node's SQLite allows ~1 GB). Only bindings are measured, not whole rows.
// - `ctx.storage.transactionSync(fn)`: one transaction, rolled back if fn
//   throws. BEGIN / COMMIT / SAVEPOINT through exec() are refused, as workerd
//   refuses them.
// - `ctx.blockConcurrencyWhile(fn)`: no RPC reaches the object before fn
//   settles. (If fn fails, later calls fail; workerd would reset the object.)
// - RPC: `env.INBOX.idFromName(name)` / `get(id)` give one object per name for
//   the namespace's life, and `restart()` evicts it (the next call constructs a
//   fresh instance over the same storage). A stub call is async; arguments and
//   results are structured-cloned, so a custom Error class would not survive --
//   the reason ShareInbox.call returns failures as data -- and an error thrown
//   by the method reaches the caller as a plain Error with only its message.
//   Every call is logged on `namespace.calls`, so a test can count round trips.
// - `crypto.subtle.timingSafeEqual`, a workerd extension Node lacks (installed
//   by harness.mjs).
//
// NOT reproduced -- a green test here says nothing about these; only
// e2e_check.py against `wrangler dev` runs real workerd:
// - Input / output gates and write coalescing. Each exec() commits at once; in
//   workerd one event's writes commit together and no reply leaves before they
//   are durable. Every ShareInbox method is synchronous, so within one RPC
//   there is nothing to interleave either way.
// - Lazy cursors: rows are materialised at exec(); workerd streams them.
// - Multi-statement exec(): refused here (node's SQLite would silently run only
//   the first statement; workerd runs them all).
// - Alarms, hibernation, the KV storage API, eviction on its own schedule, and
//   every limit except value size (CPU time, request size, statement length,
//   bound-parameter count). Error messages are not workerd's.
// - The edge in front of the Worker: CF-Connecting-IP is whatever a test sets,
//   there is no caching, and no User-Agent filtering.

import { DatabaseSync } from "node:sqlite";

/** The stricter reading of "2 MB". */
export const MAX_VALUE_BYTES = 2_000_000;

export class DurableObject {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
  }
}

class SqlStorageCursor {
  #rows;
  constructor(rows) {
    this.#rows = rows;
  }

  next() {
    return this.#rows.length ? { done: false, value: this.#rows.shift() } : { done: true, value: undefined };
  }

  [Symbol.iterator]() {
    return this;
  }

  toArray() {
    return this.#rows.splice(0);
  }

  one() {
    const rows = this.toArray();
    if (rows.length !== 1) {
      throw new Error(`Expected exactly one result from SQL query, but got ${rows.length}.`);
    }
    return rows[0];
  }
}

function checkBinding(value) {
  if (value === null || typeof value === "number") return;
  if (typeof value !== "string") {
    throw new TypeError(`workerd shim: cannot bind a ${typeof value}; extend workerd.mjs deliberately`);
  }
  if (Buffer.byteLength(value, "utf8") > MAX_VALUE_BYTES) throw new Error("string or blob too big: SQLITE_TOOBIG");
}

export class SqlStorage {
  #db;
  constructor(db) {
    this.#db = db;
  }

  exec(query, ...bindings) {
    const statement = query.trim().replace(/;\s*$/, "");
    if (statement.includes(";")) {
      throw new Error("workerd shim: one statement per exec() (node would silently drop the rest)");
    }
    if (/^(BEGIN|COMMIT|END|ROLLBACK|SAVEPOINT|RELEASE)\b/i.test(statement)) {
      throw new Error("To execute a transaction, please use the state.storage.transactionSync() API");
    }
    bindings.forEach(checkBinding);
    const rows = this.#db.prepare(statement).all(...bindings);
    return new SqlStorageCursor(rows.map((row) => ({ ...row })));
  }
}

export class DurableObjectState {
  #db;
  #gate = Promise.resolve();
  constructor(db) {
    this.#db = db;
    this.storage = {
      sql: new SqlStorage(db),
      transactionSync: (fn) => this.#transactionSync(fn),
    };
  }

  #transactionSync(fn) {
    this.#db.exec("SAVEPOINT workerd_shim");
    try {
      const value = fn();
      if (typeof value?.then === "function") throw new TypeError("transactionSync callback must be synchronous");
      this.#db.exec("RELEASE workerd_shim");
      return value;
    } catch (error) {
      this.#db.exec("ROLLBACK TO workerd_shim");
      this.#db.exec("RELEASE workerd_shim");
      throw error;
    }
  }

  blockConcurrencyWhile(fn) {
    // Started at once, as workerd does: it runs up to its first await before
    // the constructor returns.
    const run = (async () => fn())();
    this.#gate = this.#gate.then(() => run);
    return this.#gate;
  }

  /** What an incoming call waits on. A failed blocking callback fails every
   * later call on this instance (workerd would reset the object instead). */
  get gate() {
    return this.#gate;
  }
}

export class DurableObjectNamespace {
  #cls;
  #env;
  #objects = new Map();
  /** Every RPC, in order: `{ name, method, args }`. */
  calls = [];

  constructor(cls, env) {
    this.#cls = cls;
    this.#env = env;
  }

  idFromName(name) {
    return Object.freeze({ name: String(name), toString: () => `id:${name}` });
  }

  /** The storage behind one object, for tests that seed or inspect it. */
  database(name) {
    return this.#entry(name).db;
  }

  /** Evict every live instance; the next call constructs anew over the same storage. */
  restart() {
    for (const entry of this.#objects.values()) entry.instance = null;
  }

  #entry(name) {
    let entry = this.#objects.get(name);
    if (!entry) {
      entry = { db: new DatabaseSync(":memory:"), instance: null, state: null };
      this.#objects.set(name, entry);
    }
    return entry;
  }

  #instance(entry) {
    if (!entry.instance) {
      entry.state = new DurableObjectState(entry.db);
      entry.instance = new this.#cls(entry.state, this.#env);
    }
    return entry.instance;
  }

  get(id) {
    const entry = this.#entry(id.name);
    return new Proxy(
      {},
      {
        get: (_target, method) => {
          if (typeof method !== "string" || method === "then") return undefined;
          return async (...args) => {
            this.calls.push({ name: id.name, method, args: structuredClone(args) });
            const instance = this.#instance(entry);
            await entry.state.gate;
            let result;
            try {
              result = instance[method](...structuredClone(args));
            } catch (error) {
              throw new Error(String(error?.message || error));
            }
            return structuredClone(await result);
          };
        },
      },
    );
  }
}
