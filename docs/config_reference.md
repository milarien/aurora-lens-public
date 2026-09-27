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

## Provider notes

- `openai`: set `api_key_env: OPENAI_API_KEY`
- `anthropic`: set `api_key_env: ANTHROPIC_API_KEY`
- `local`: set `base_url` (for example `http://127.0.0.1:11434/v1`); `api_key_env` only if your endpoint requires auth
