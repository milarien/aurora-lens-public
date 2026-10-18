/** Owner audit adapter. Supply the site's existing Access verifier and response
 * wrapper. No encryption key, cookies or user-supplied identity reach Railway.
 */
export async function handleEvidenceReview({request, env}, verifyOwner, ownerResponse, ownerEmail, fetcher = fetch) {
  if (!(await verifyOwner(request, fetcher))) {
    // Fixed metadata only: never log claims, headers, cookies or evidence.
    console.warn("evidence-review-access-denied");
    return ownerResponse("Access denied. Owner authentication required.", 403, "text/plain; charset=utf-8");
  }
  if (request.method !== "GET")
    return ownerResponse("Read-only endpoint.", 405, "text/plain; charset=utf-8");
  const q = new URL(request.url).searchParams, decision = q.get("decision");
  if ([...q.keys()].some(k => k !== "decision") || q.getAll("decision").length > 1 ||
      (decision !== null && !/^[0-9a-f]{64}$/.test(decision)))
    return ownerResponse("Invalid decision selection.", 400, "text/plain; charset=utf-8");
  const origin = String(env.AURORA_RAILWAY_ORIGIN || "").replace(/\/$/, "");
  const secret = String(env.AURORA_EDGE_TOKEN || "");
  try {
    if (origin !== "https://auroralens-production.up.railway.app" || !secret)
      throw new Error("configuration");
    const query = decision ? "decision=" + decision : "";
    const path = "/v1/audit/evidence-review";
    const bytes = new TextEncoder();
    const hmac = async (key, value) => {
      const imported = await crypto.subtle.importKey("raw", key,
        {name: "HMAC", hash: "SHA-256"}, false, ["sign"]);
      return new Uint8Array(await crypto.subtle.sign("HMAC", imported, bytes.encode(value)));
    };
    const nonce = crypto.randomUUID().replaceAll("-", "");
    const payload = btoa(JSON.stringify({actor: ownerEmail, purpose: "forensic_review",
      iat: Math.floor(Date.now() / 1000), nonce})).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
    const key = await hmac(bytes.encode(secret), "aurora-evidence-review-v1");
    const signature = [...await hmac(key, `evidence-review-v1\nGET\n${path}\n${query}\n${payload}`)]
      .map(b => b.toString(16).padStart(2, "0")).join("");
    const r = await fetcher(origin + path + (query ? "?" + query : ""), {
      method: "GET", redirect: "manual", signal: AbortSignal.timeout(30000),
      headers: {Accept: "text/html", "x-aurora-edge-token": secret,
        "x-aurora-review-assertion": payload, "x-aurora-review-signature": signature},
    });
    if (![200, 400, 403, 404, 405, 503].includes(r.status) ||
        !r.headers.get("content-type")?.startsWith("text/html")) throw new Error("backend");
    const html = await r.text();
    if (html.includes(secret)) throw new Error("credential exposure");
    // Preserve the exact escaped evidence text. Relative links work on the
    // owner route as well as the built-in review route; no content rewriting.
    return ownerResponse(html, r.status);
  } catch {
    return ownerResponse("Evidence review unavailable. No original evidence was changed.", 503, "text/plain; charset=utf-8");
  }
}
