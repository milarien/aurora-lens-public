# Aurora-Lens

Aurora-Lens sits between your application and a language model. It checks each turn before output becomes consequence-bearing, records the decision in an audit log, and can expose an operator web page for health, audit review, and session inspection.

This README is the main setup guide for **this tree** (install from source, configure, run, use the operator page).

## Architecture

Aurora-Lens is a governance layer, not a replacement language model. Its main runtime path is:

1. A client sends an OpenAI-compatible chat request to the proxy.
2. Pre-model state and policy checks determine whether the request can proceed, must be clarified, or must be stopped.
3. An upstream provider adapter sends admitted requests to the configured model.
4. Verification and commitment-control components evaluate the candidate response before release.
5. The governor returns a governed outcome such as pass, contain, clarify, or hard stop.
6. Session state and tamper-evident audit records preserve the basis for later review.

The principal code areas are:

| Area | Location | Responsibility |
|---|---|---|
| Proxy and provider boundary | `aurora_lens/proxy/`, `aurora_lens/adapters/` | OpenAI-compatible transport, provider routing, and lifecycle |
| Interpretation and state | `aurora_lens/interpret/`, `aurora_lens/pef/`, `aurora_lens/state_native_engine/` | Structured turn interpretation, persistent epistemic state, and state-native evaluation |
| Governance | `aurora_lens/govern/`, `aurora_lens/governor/` | Admissibility, continuation, commitment control, and governed outcomes |
| Verification | `aurora_lens/verify/` | Candidate-output checks, hazard handling, and response alignment |
| Evidence and trust | `aurora_lens/corpus/`, `aurora_lens/trust/`, `aurora_lens/sovereign/` | Optional document evidence, source trust, and provider governance |
| Audit and operations | `aurora_lens/govern/forensic_ledger.py`, `aurora_lens/proxy/dashboard.html` | Reviewable records and the operator interface |

The repository includes the filed patent specifications in `patents/`. Application numbers, filing dates, titles, and file mappings are in `PATENTS.md`. Provenance and integrity information are in `PROVENANCE.md` and `SHA256SUMS.txt`.

---

## What you need

- **Python 3.12 only** (`>=3.12,<3.13`)
- An **API key** for your model provider (OpenAI, Anthropic, or an OpenAI-compatible local server) — **not** required for the offline demo below

---

## 1. Install

Create an isolated virtual environment, then install from this folder with a **non-editable** install (do not use `pip install -e`):

```bash
python -m venv .venv
```

Activate the environment:

- Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
- macOS / Linux: `source .venv/bin/activate`

```bash
python -m pip install -U pip
pip install ".[proxy]"
```

On first `aurora-lens start` with the default config, Aurora-Lens checks for spaCy
and `en_core_web_sm` and installs them automatically if missing. You can also
pre-install the model yourself:

```bash
python -m spacy download en_core_web_sm
```

Official release install is a non-editable `pip install` into an isolated virtual environment from the extracted release directory.

For PDF or Word corpus ingestion later (same activated environment):

```bash
pip install ".[ingest]"
```

---

## 2. Try it without an API key (recommended first step)

This runs three short scenarios (PASS, CONTAIN, HARD_STOP) and writes **`start_here_demo_audit.jsonl`** in this folder:

```bash
python tools/run_demo.py
```

If all three show `[OK]`, the governance layer is working on your machine.

---

## 3. Configure for a live proxy

1. Copy the example config:

   ```bash
   cp aurora-lens.yaml.example aurora-lens.yaml
   ```

   On Windows PowerShell: `Copy-Item aurora-lens.yaml.example aurora-lens.yaml`

   Or generate a minimal config at the fixed user config location:

   ```bash
   aurora-lens init-config
   ```

2. Edit **`aurora-lens.yaml`**:

   - **`upstream.provider`** — `openai` or `anthropic` (or use a local OpenAI-compatible server)
   - **`upstream.model`** — model id your provider supports
   - **`listen.port`** — free port on your machine (example uses **8081**)
   - **`governance.audit_log`** — where audit lines are written (example: `./audit.jsonl`)

3. Set your provider credential environment variable **before** starting:

   ```bash
   export OPENAI_API_KEY="your-key"
   ```

   Windows PowerShell:

   ```powershell
   $env:OPENAI_API_KEY = "your-key-here"
   ```

   Environment variable names and overrides are listed in **`docs/config_reference.md`** and the example YAML files.

4. Ensure your YAML uses an explicit env var name:

   - `upstream.api_key_env: OPENAI_API_KEY` (OpenAI)
   - `upstream.api_key_env: ANTHROPIC_API_KEY` (Anthropic)
   - for local/custom OpenAI-compatible endpoints, set `api_key_env` only if that endpoint requires auth

Field-by-field YAML notes: **`docs/config_reference.md`**.

---

## 4. Start the proxy

**Canonical lifecycle commands:**

```bash
aurora-lens start
aurora-lens status
aurora-lens stop
```

`--config <absolute-or-relative-path>` takes precedence when provided.
Otherwise Aurora-Lens uses a fixed user config location under the runtime root and prints the resolved path.
On Windows the default is `%LOCALAPPDATA%\Aurora-Lens\config\aurora-lens.yaml`.

Runtime state, logs, support bundles, and audit files are created under the user-data runtime root
(default `%LOCALAPPDATA%\Aurora-Lens`). They are not stored in the installed package directory.

**Windows wrappers (tiny command shims):**

- `Start Aurora-Lens.bat` → `aurora-lens start`
- `Status Aurora-Lens.bat` → `aurora-lens status`
- `Stop Aurora-Lens.bat` → `aurora-lens stop`

Leave the process running. Default listen URL (if port is 8081): `http://127.0.0.1:8081`

Check health: `http://127.0.0.1:8081/health`

---

## 5. Operator web page (Forensics)

With the proxy running, open in a browser:

**`http://127.0.0.1:8081/forensics`**

(Use your **`listen.port`** if not 8081. Same page is also at **`/dashboard`** and **`/operator`**.)

From this page you can:

- see proxy health and where audit logs are written  
- verify the tamper-evident audit chain (when a signing key is configured)  
- browse recent governance decisions  
- load **session state (PEF)** for one chat session when you paste a session id  

Plain-English tour of each section: **`docs/forensics-dashboard.md`**.

HTTP routes used by the page (and by `curl`): **`docs/api.md`**.

---

## 6. Send chat requests

Point any OpenAI-compatible client at this base URL:

```text
http://127.0.0.1:8081/v1/chat/completions
```

Or use the built-in chat helper (proxy must be running):

```bash
python tools/chat_with_lens.py
```

---

## 7. Document library (optional)

To ingest PDFs and ask governed questions against them:

```bash
pip install ".[ingest]"
aurora-lens corpus ingest --record-id my-doc --file data/my-doc.pdf
aurora-lens proxy    # must be running for "ask"
aurora-lens corpus ask --record-id my-doc --question "What does section 2 say?"
```

Step-by-step: **`docs/corpus-guide.md`**.

Phase 1.7 realistic acceptance package:

```bash
python scripts/run_phase17_acceptance.py --mode offline
```

Optional live-provider acceptance:

```bash
python scripts/run_phase17_acceptance.py --mode live --provider openai --model MODEL_ID
```

Reference: realistic corpus governance checks are covered in **`docs/corpus-guide.md`** in this release tree.

---

## 8. Windows step-by-step

If you prefer a full Windows walkthrough (venv, PowerShell, browser): **`docs/setup-windows-proxy.md`**.

---

## 9. Licence

Aurora-Lens is proprietary software. All rights reserved.

- **Terms:** see **`LICENSE`**, **`NOTICE`**, and **`LICENSING.md`**
- **Commercial / evaluation rights:** only under a separate written agreement — see **`COMMERCIAL-LICENCE.md`**

```bash
aurora-lens license
```

---

## Documentation in this tree

| File | Purpose |
|------|---------|
| **`INSTALL.txt`** | Short install reminder |
| **`docs/config_reference.md`** | Configuration fields |
| **`docs/setup-windows-proxy.md`** | Windows install and run |
| **`docs/forensics-dashboard.md`** | Operator web page |
| **`docs/api.md`** | HTTP API (health, chat, audit) |
| **`docs/corpus-guide.md`** | Document ingest and ask |
| **`docs/uninstall-and-reset.md`** | Reset, log export, and uninstall workflow |

---

## If something fails

| Problem | What to check |
|---------|----------------|
| `ModuleNotFoundError: aurora_lens` | Activate the release venv and run `pip install ".[proxy]"` from this folder |
| Proxy page will not load | Process running? Correct port in YAML? Firewall? |
| `401` on chat | API key set in environment; matches `upstream.provider` |
| Empty audit table | `governance.audit_log` set in YAML; directory writable |
| Demo shows UNEXPECTED | Re-run `python tools/run_demo.py` and note the full output |
