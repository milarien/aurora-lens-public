# Aurora-Lens

Aurora-Lens is a deterministic commitment-governance architecture. It evaluates whether a candidate state, interpretation, determination, output, release, or action is admissible to become operative or consequential.

Here, deterministic refers to the admissibility outcome produced from the same candidate, authoritative state, applicable policy pack and evaluation context. It does not describe or require deterministic generation of the candidate.

Its central invariant is:

> A candidate does not acquire standing or consequence merely because a model, retrieval system, application, or prior state produced it. Commitment requires independent admissibility.

Where admissibility is not established, Aurora-Lens preserves governed non-commitment rather than forcing resolution.

This repository is the public Aurora-Lens record: architecture and publications, patent status, provenance, and proprietary runtime source, published for inspection. Running it requires a separate written licence.

This repository is a canonical public index and preservation corpus, not an independent timestamping authority. Historical priority and publication dates rest on the cited patent filings, Zenodo deposits and other external records.

**Earliest claimed priority represented in this portfolio:** 27 November 2025, Australian provisional application AU 2025905835. Priority is claim-specific and depends on the disclosure contained in the relevant filing.

## Start here

**Current architecture:** [CURRENT_ARCHITECTURE.md](architecture/CURRENT_ARCHITECTURE.md)

- [Core invariants](architecture/CORE-INVARIANTS.md): the shortest statement of the architecture.
- [Canonical architecture map](architecture/Aurora_Lens_Canonical_Architecture_Map.md): commitment surfaces, authority, unresolved state, outcomes, persistence, and audit.
- [Runtime map](architecture/AURORA_LENS_RUNTIME_MAP.md): the public component separation and runtime vocabulary.
- [Publication catalogue](publications/zenodo/PUBLICATION-CATALOGUE.md): twenty distinct works in the current publication register.
- [Patent notice](PATENT_NOTICE.md): public specification deposits (Zenodo, TDCommons, IP Australia OPI reference).
- [Patent portfolio](patents/PATENT-PORTFOLIO.md): bibliographic filing table (no specifications in this tree).
- [Release manifest](RELEASE_MANIFEST.md) and [changelog](CHANGELOG.md): what this public repository includes and excludes.
- [Public provenance timeline](provenance/PROVENANCE-TIMELINE.md): event dates separated from later public deposits.
- [Private-file hash manifest](provenance/EVIDENCE-MANIFEST.txt): fingerprints only; the underlying private files are not present.

The public canonical statement is deposited in [Zenodo record 21930519](https://doi.org/10.5281/zenodo.21930519). The records preserved under `publications/zenodo/records/` retain the exact deposited filenames, metadata, and checksums.

## Repository structure

```text
architecture/             Public invariants and architecture maps
aurora_lens/              Runtime source (published for inspection; see LICENSE)
patents/                  Bibliographic patent-status record (specifications via PATENT_NOTICE.md)
provenance/               Sourced chronology and private-file fingerprints
publications/zenodo/      Exact current Zenodo deposits, metadata, and manifests
tests/                    Automated tests
scripts/                  Operator and acceptance utilities
tools/                    Demo and chat helpers
```

## Public surfaces

- Website: [aurora-lens.ai](https://aurora-lens.ai/)
- Publications: [Zenodo catalogue](publications/zenodo/PUBLICATION-CATALOGUE.md)
- ORCID: [0009-0004-6422-4174](https://orcid.org/0009-0004-6422-4174)
- Licensing enquiries: [margaret.stokes.ai@gmail.com](mailto:margaret.stokes.ai@gmail.com)

Licensing varies by document and deposit. See [LICENSES.md](LICENSES.md), [LEGAL-NOTICE.md](LEGAL-NOTICE.md), and the metadata for each Zenodo record.

---

## Running the runtime (licensed use only)

These instructions are for parties holding a written licence to run Aurora-Lens. The commands are included for transparency; they do not grant permission to use the software.

Aurora-Lens sits between your application and a language model. It checks each turn before output becomes consequence-bearing, records the decision in an audit log, and can expose an operator web page for health, audit review, and session inspection.

### What you need

- **Python 3.12 only** (`>=3.12,<3.13`)
- An **API key** for your model provider (OpenAI, Anthropic, or an OpenAI-compatible local server) — **not** required for the offline demo below

### 1. Install

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

### 2. Try it without an API key (recommended first step)

This runs three short scenarios (PASS, CONTAIN, HARD_STOP) and writes **`start_here_demo_audit.jsonl`** in this folder:

```bash
python tools/run_demo.py
```

If all three show `[OK]`, the governance layer is working on your machine.

### 3. Configure for a live proxy

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

### 4. Start the proxy

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

### 5. Operator web page (Forensics)

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

### 6. Send chat requests

Point any OpenAI-compatible client at this base URL:

```text
http://127.0.0.1:8081/v1/chat/completions
```

Or use the built-in chat helper (proxy must be running):

```bash
python tools/chat_with_lens.py
```

### 7. Document library (optional)

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

### 8. Windows step-by-step

If you prefer a full Windows walkthrough (venv, PowerShell, browser): **`docs/setup-windows-proxy.md`**.

### 9. Licence

Aurora-Lens is proprietary software. All rights reserved.

- **Terms:** see **`LICENSE`**, **`NOTICE`**, and **`LICENSING.md`**
- **Commercial / evaluation rights:** only under a separate written agreement — contact **margaret.stokes.ai@gmail.com**

```bash
aurora-lens license
```

---

## Documentation in this tree

| File | Purpose |
|------|---------|
| **`PATENT_NOTICE.md`** | Public patent deposits and IP Australia OPI reference |
| **`RELEASE_MANIFEST.md`** | Included / excluded public material |
| **`CHANGELOG.md`** | Repository release history |
| **`INSTALL.txt`** | Short setup reminder (licensed use only) |
| **`docs/config_reference.md`** | Configuration fields |
| **`docs/setup-windows-proxy.md`** | Windows setup (licensed use only) |
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
