// Module resolve hook for the Worker's unit tests (registered by harness.mjs).
//
// Two jobs, both so that src/index.js is imported exactly as it ships:
// - `cloudflare:workers` exists only inside workerd; map it to the stand-in.
// - The Worker has no package.json saying it is ESM. Node >= 22.7 would detect
//   that from the syntax, but 22.5-22.6 (the oldest with node:sqlite) would
//   load it as CommonJS and fail on `import`, so say it outright.

const WORKERD = new URL("./workerd.mjs", import.meta.url).href;
const WORKER_SOURCE = new URL("../src/", import.meta.url).href;

export async function resolve(specifier, context, nextResolve) {
  if (specifier === "cloudflare:workers") {
    return { url: WORKERD, format: "module", shortCircuit: true };
  }
  const resolved = await nextResolve(specifier, context);
  if (resolved.url.startsWith(WORKER_SOURCE)) return { ...resolved, format: "module" };
  return resolved;
}
