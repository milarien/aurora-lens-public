import test from "node:test";
import assert from "node:assert/strict";
import {createHmac, webcrypto} from "node:crypto";
import {handleEvidenceReview} from "../integrations/cloudflare/evidence_review.js";
globalThis.crypto ??= webcrypto;
const env = {AURORA_RAILWAY_ORIGIN: "https://auroralens-production.up.railway.app", AURORA_EDGE_TOKEN: "synthetic-transport"};
const wrap = (body, status = 200, type = "text/html") => new Response(body, {status, headers: {"content-type": type, "cache-control": "private, no-store"}});
const owner = "synthetic-reviewer@example.test";
const req = (query = "", method = "GET") => new Request("https://aurora-lens.ai/admin/audit/review" + query,
  {method, headers: {Cookie: "synthetic-cookie", "Cf-Access-Jwt-Assertion": "synthetic-jwt", "x-aurora-review-actor": "forged"}});
test("denied owner is rejected before any backend request", async () => {
  const r = await handleEvidenceReview({request: req(), env}, async () => false, wrap, owner,
    () => assert.fail("anonymous request reached backend"));
  assert.equal(r.status, 403);
  assert.match(await r.text(), /Access denied/);
});
test("verified owner gets a selection-bound signed request without browser credentials", async () => {
  const id = "a".repeat(64);
  const r = await handleEvidenceReview({request: req("?decision=" + id), env}, async () => true, wrap, owner,
    async (url, opts) => {
      assert.equal(url, env.AURORA_RAILWAY_ORIGIN + "/v1/audit/evidence-review?decision=" + id);
      assert.equal(opts.redirect, "manual");
      assert.equal(opts.method, "GET");
      assert.equal(opts.headers.Cookie, undefined);
      assert.equal(opts.headers["Cf-Access-Jwt-Assertion"], undefined);
      assert.equal(opts.headers["x-aurora-review-actor"], undefined);
      const payload = opts.headers["x-aurora-review-assertion"];
      const actor = JSON.parse(Buffer.from(payload, "base64url"));
      assert.equal(actor.actor, owner);
      assert.equal(actor.purpose, "forensic_review");
      const key = createHmac("sha256", env.AURORA_EDGE_TOKEN).update("aurora-evidence-review-v1").digest();
      const message = `evidence-review-v1\nGET\n/v1/audit/evidence-review\ndecision=${id}\n${payload}`;
      assert.equal(opts.headers["x-aurora-review-signature"], createHmac("sha256", key).update(message).digest("hex"));
      return wrap('<h1>Synthetic evidence</h1><a href="?decision=' + id + '">Open decision</a>');
    });
  assert.equal(r.status, 200);
  assert.ok(!(await r.text()).includes(env.AURORA_EDGE_TOKEN));
});
test("invalid query and mutation are rejected without backend access", async () => {
  for (const [query, method, status] of [["?file=../secret", "GET", 400], ["?decision=" + "a".repeat(64) + "&decision=" + "b".repeat(64), "GET", 400], ["", "POST", 405]]) {
    const r = await handleEvidenceReview({request: req(query, method), env}, async () => true, wrap, owner,
      () => assert.fail("invalid request reached backend"));
    assert.equal(r.status, status);
  }
});
test("redirects, unexpected bodies and credential exposure fail closed", async () => {
  for (const reply of [new Response(null, {status: 307, headers: {location: "https://evil.example"}}),
    wrap(env.AURORA_EDGE_TOKEN), new Response("secret diagnostic", {status: 500}),
    new Response('{"private":true}', {headers: {"content-type": "application/json"}})]) {
    const r = await handleEvidenceReview({request: req(), env}, async () => true, wrap, owner, async () => reply);
    assert.equal(r.status, 503);
    const body = await r.text();
    assert.ok(!body.includes(env.AURORA_EDGE_TOKEN) && !body.includes("secret diagnostic"));
  }
});
test("backend denial and unavailable states are preserved", async () => {
  for (const status of [403, 404, 503]) {
    const r = await handleEvidenceReview({request: req(), env}, async () => true, wrap, owner,
      async () => wrap("<h1>Access denied or unavailable</h1>", status));
    assert.equal(r.status, status);
    assert.equal(r.headers.get("cache-control"), "private, no-store");
  }
});
