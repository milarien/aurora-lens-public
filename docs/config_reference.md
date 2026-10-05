# Configuration Reference

Aurora-Lens uses one explicit YAML file (default on Windows:
`%LOCALAPPDATA%\Aurora-Lens\config\aurora-lens.yaml`).

## Minimal required fields

```yaml
upstream:
  provider: openai
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY

listen:
  host: 127.0.0.1
  port: 8081

governance:
  default_policy: strict
  mode: public
  audit_log: ./audit.jsonl
```

## Credential rule

- Do not store secrets in YAML.
- Set `upstream.api_key_env` to the credential variable name.
- Aurora-Lens fails startup if that variable is missing or empty.

## Public-demo edge token

`AURORA_EDGE_TOKEN` is not part of a licensed local install. Leave it unset.
Chat on your own machine does not send or require `x-aurora-edge-token`.

The hosted public demo runs on Railway, which sets `RAILWAY_ENVIRONMENT_NAME`
(older images set `RAILWAY_ENVIRONMENT`). Only then does the proxy require
`x-aurora-edge-token` on `POST /v1/chat/completions` and
`POST /v1/session/new-scenario`. The header must match `AURORA_EDGE_TOKEN`.
A missing or wrong token is rejected with 403. The comparison hashes both
values and uses `hmac.compare_digest`, so it does not leak the secret through
timing. If that deployment has no secret configured, those two routes fail
closed. Do not put the secret in YAML.

## Audit reads

`GET /v1/audit/recent` and `GET /v1/audit/search` are open to the local
forensics page when `listen.host` is `127.0.0.1`, `localhost`, or `::1`,
including when inbound auth is enabled. On any other listen address, such as
the production template's `0.0.0.0`, those two routes require the inbound API
key whenever `auth.enabled` is true. That keeps a network bind from publishing
the audit log. Other forensics routes used by the operator page are unchanged.

## Provider notes

- `openai`: set `api_key_env: OPENAI_API_KEY`
- `anthropic`: set `api_key_env: ANTHROPIC_API_KEY`
- `local`: set `base_url` (for example `http://127.0.0.1:11434/v1`); `api_key_env` only if your endpoint requires auth
