# How to use the document library and ask questions

**Licensed use only.** These commands are for parties holding a written licence to run Aurora-Lens. They document how the runtime is operated; they do not grant permission to use the software. See [LICENSE](../LICENSE) and [README.md](../README.md).

Use the **`aurora-lens corpus`** commands below. You do **not** need to know programming. Copy the commands, paste them in a terminal, and press Enter.

---

## Two ways to use a document

| Door | Command | Needs the running chat server? |
|------|---------|--------------------------------|
| **Door 1 — Read and review** | `aurora-lens corpus review --record-id ID` | **No** |
| **Door 2 — Ask a question** | `aurora-lens corpus ask --record-id ID --question "…"` | **Yes** (`aurora-lens proxy`) |

**Door 1** goes straight to the AI (like emailing the document to a helper).  
**Door 2** goes through Aurora-Lens (like asking a librarian who checks the answer).

---

## Before you start (one time)

1. Open a terminal in the project folder.

2. Install the PDF helper (one time):
   ```powershell
   pip install pypdf
   ```

3. For **Door 1** (review), set an AI key:
   ```powershell
   $env:OPENAI_API_KEY = "your-key-here"
   ```

4. Put your document under `data\` (example: `data\policy.pdf`).

---

## Step 1 — Load the document (ingest)

```powershell
aurora-lens corpus ingest --record-id my-policy --file data\policy.pdf
```

List ingested records:

```powershell
aurora-lens corpus list
```

---

## Step 2 — Ask a question (Door 2)

Start the governed proxy in one terminal:

```powershell
aurora-lens proxy
```

In another terminal:

```powershell
aurora-lens corpus ask --record-id my-policy --question "What are the approval criteria?"
```

---

## Step 3 — Review without Lens (Door 1)

```powershell
aurora-lens corpus review --record-id my-policy
```

---

## More detail

- **`README.md`** — runtime setup (licensed use only) and proxy operation  
- **`docs/api.md`** — HTTP routes for audit and chat  
- **`docs/acceptance-phase17.md`** — realistic corpus acceptance pack

---

## Legacy tools

The **`aurora-lens corpus`** CLI is the supported path. Use **`README.md`** for proxy setup before Door 2 (`ask`).

---

## Phase 1.7 acceptance commands

Deterministic offline acceptance:

```powershell
python scripts/run_phase17_acceptance.py --mode offline
```

Optional live-provider acceptance:

```powershell
python scripts/run_phase17_acceptance.py --mode live --provider openai --model MODEL_ID
```
