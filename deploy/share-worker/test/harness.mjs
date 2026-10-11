// Loads src/index.js unchanged over the workerd stand-in and drives it the way
// the edge does: a Request into the default export's `fetch`, a Response out.
//
// The resolve hook is registered here rather than through `--import`, so a
// bare `node --test test/` works: every test file imports this module first,
// and index.js is only imported (dynamically) after the hook is in place.

import { timingSafeEqual } from "node:crypto";
import { register } from "node:module";

import { DurableObjectNamespace } from "./workerd.mjs";

register("./hooks.mjs", import.meta.url);

// workerd's extension to Web Crypto; same contract (throws on unequal lengths).
if (typeof crypto.subtle.timingSafeEqual !== "function") {
  crypto.subtle.timingSafeEqual = (a, b) => timingSafeEqual(a, b);
}

const { default: worker, ShareInbox } = await import("../src/index.js");

export const MAINTAINER_TOKEN = "test-maintainer-token";
const ORIGIN = "https://share.test";
const DAY_MS = 86400 * 1000;

export async function sha256Hex(text) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Buffer.from(digest).toString("hex");
}

/** Freeze the clock at a fixed instant for this test; returns `advanceDays`. */
export function useClock(t, start = "2026-03-01T00:00:00Z") {
  t.mock.timers.enable({ apis: ["Date"], now: Date.parse(start) });
  return (days) => t.mock.timers.setTime(Date.now() + days * DAY_MS);
}

/** A fresh Worker with its own empty storage. */
export function startWorker({ maintainerToken = MAINTAINER_TOKEN } = {}) {
  const env = maintainerToken === null ? {} : { MAINTAINER_TOKEN: maintainerToken };
  env.INBOX = new DurableObjectNamespace(ShareInbox, env);
  let addresses = 0;

  /** One request through the edge. `body` objects go as JSON; a real client
   * declares Content-Length, so it is set unless `contentLength: false`
   * (a chunked upload, where the Worker has only the bytes to go by). */
  async function send(method, path, { token, maintainer, ip, body, headers = {}, contentLength = true } = {}) {
    const all = { ...headers };
    if (token !== undefined) all["X-Share-Token"] = token;
    if (maintainer !== undefined) all["X-Maintainer-Token"] = maintainer;
    if (ip !== undefined) all["CF-Connecting-IP"] = ip;
    let payload;
    if (body !== undefined) {
      payload = typeof body === "string" ? body : JSON.stringify(body);
      if (contentLength) all["Content-Length"] = String(Buffer.byteLength(payload, "utf8"));
    }
    return worker.fetch(new Request(ORIGIN + path, { method, headers: all, body: payload }), env);
  }

  /** `send` for a JSON endpoint: `{ status, body }`. */
  async function call(method, path, options) {
    const response = await send(method, path, options);
    return { status: response.status, body: await response.json() };
  }

  /** Register from a fresh address (the per-address daily limit is 3). */
  async function register(ip = `198.51.100.${++addresses}`) {
    const { status, body } = await call("POST", "/register", { ip });
    if (status !== 200) throw new Error(`register failed: ${status} ${JSON.stringify(body)}`);
    return body.token;
  }

  const asMaintainer = (options = {}) => ({ maintainer: MAINTAINER_TOKEN, ...options });

  /** The ShareInbox method names the Worker asked for, in order. */
  const rpcNames = () => env.INBOX.calls.map((c) => (c.method === "call" ? c.args[0] : c.method));

  return { env, inbox: env.INBOX, send, call, register, asMaintainer, rpcNames };
}
