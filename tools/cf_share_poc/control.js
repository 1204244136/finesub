// Control for the Python PoC: the same Durable Object shape in JavaScript.
// If this works and the Python one does not, the blocker is Python-DO-specific
// rather than "Durable Objects do not run on this machine".
import { DurableObject } from "cloudflare:workers";

export class StorePoc extends DurableObject {
  async probe() {
    const sql = this.ctx.storage.sql;
    sql.exec("CREATE TABLE IF NOT EXISTS poc(id INTEGER PRIMARY KEY, v TEXT)");
    sql.exec("INSERT INTO poc(v) VALUES (?)", "one");
    const lastRowId = sql.exec("SELECT last_insert_rowid() AS id").one().id;

    // rollback semantics: transactionSync must undo the write when the
    // callback throws (this is what has to carry store.begin()).
    let raised = "none";
    try {
      this.ctx.storage.transactionSync(() => {
        sql.exec("INSERT INTO poc(v) VALUES (?)", "doomed");
        throw new Error("deliberate");
      });
    } catch (e) {
      raised = String(e.message ?? e);
    }
    const doomed = sql.exec("SELECT COUNT(*) AS n FROM poc WHERE v='doomed'").one().n;

    // BEGIN inside exec() must be refused (the reason store.begin() has to be
    // rewritten rather than passed through).
    let beginError = "accepted";
    try {
      sql.exec("BEGIN IMMEDIATE");
    } catch (e) {
      beginError = String(e.message ?? e).slice(0, 160);
    }

    return {
      rows: sql.exec("SELECT COUNT(*) AS n FROM poc").one().n,
      lastRowId,
      rollback_raised: raised,
      doomed_rows_after_rollback: doomed,
      begin_inside_exec: beginError,
    };
  }
}

export default {
  async fetch(request, env) {
    const stub = env.STORE.get(env.STORE.idFromName("poc"));
    const body = await stub.probe();
    return new Response(JSON.stringify(body, null, 2), {
      headers: { "Content-Type": "application/json" },
    });
  },
};
