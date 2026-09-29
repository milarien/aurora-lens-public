# Run Aurora-Lens (governed proxy) on Windows

**Licensed use only.** These steps are for parties holding a written licence to run Aurora-Lens. They document how the runtime is operated; they do not grant permission to use the software. See [LICENSE](../LICENSE) and [README.md](../README.md).

Plain step-by-step: put the project on the PC, install Python, configure Aurora-Lens, set your API key, start the proxy, open the browser.

---

## 1. Put the project on the computer

- Clone or copy the **aurora-lens** repository to a folder you control, for example `C:\projects\aurora-lens`.
- You need the full **`aurora_lens`** package tree and **`pyproject.toml`** in that folder (a normal git clone is enough).

---

## 2. Install Python

- Install **Python 3.12** from [python.org](https://www.python.org/downloads/). Do not use 3.11 or 3.13.
- During setup, enable **“Add python.exe to PATH”** (or equivalent) so PowerShell can find `python`.

---

## 3. Open PowerShell and go to the project folder

```powershell
cd C:\projects\aurora-lens
```

(Change the path if your project lives somewhere else.)

---

## 4. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If execution policy blocks activation, run PowerShell **once** as Administrator:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

---

## 5. Install the project and its required extras

Upgrade pip, then install this repo with a **non-editable** install into the active virtual environment. Use the **proxy** stack (FastAPI, Uvicorn, PyYAML). Do not use `pip install -e`.

```powershell
python -m pip install -U pip
pip install ".[proxy]"
```

---

## 6. spaCy setup on first start

The generated config defaults to **`extraction.backend: spacy`**. On `aurora-lens start`, Aurora-Lens checks for spaCy and `en_core_web_sm` and installs missing pieces automatically.

Optional pre-install:

```powershell
python -m spacy download en_core_web_sm
```

If you later set **`extraction.backend: llm`** in YAML and do not use spaCy, you can skip this step.

---

## 7. Create a config file for Aurora-Lens

- In the project folder, **copy** `aurora-lens.yaml.example` to **`aurora-lens.yaml`** (same folder is simplest).
- Open **`aurora-lens.yaml`** in a text editor.

Environment variable names and more options are listed in **`.env.example`** and **`docs/config_reference.md`**.

---

## 8. In that config file, choose

### Which upstream provider to use

Under **`upstream`**, set **`provider`** to:

- **`openai`** — OpenAI or an OpenAI-compatible endpoint, or  
- **`anthropic`** — Anthropic Claude.

### Which model to use

Under **`upstream`**, set **`model`** to a model id your account supports (examples: `gpt-4o-mini`, or a Claude model name your dashboard lists).

### Which port the server should run on

Under **`listen`**, set **`port`** to any free port on your PC (the example uses **`8081`**).  
Remember this number for the browser step.

Optional: set **`listen.host`** to **`127.0.0.1`** if the server should only accept connections from your machine.

---

## 9. Set the matching API key in PowerShell for that provider

Use the **same** PowerShell window where the virtual environment is **activated**.

**OpenAI** (when `provider: openai`):

```powershell
$env:OPENAI_API_KEY = "your-secret-key-here"
```

**Anthropic** (when `provider: anthropic`):

```powershell
$env:ANTHROPIC_API_KEY = "your-secret-key-here"
```

Set `upstream.api_key_env` in YAML to the variable name you export (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`).

Do not commit real keys into git. Prefer env vars or a local-only file that is ignored.

---

## 10. Start the proxy with the config file

From the project folder (venv still active):

```powershell
aurora-lens start
```

This starts the managed proxy process, waits for health, and prints resolved config/log paths.

Or double-click **`START_TRIAL_WINDOWS.bat`**.

Optional: point to a specific config file:

```powershell
aurora-lens start --config .\aurora-lens.yaml
```

Equivalent module entrypoint:

```powershell
python -m aurora_lens.proxy --config aurora-lens.yaml
```

Leave this window open while the server runs. Stop with **Ctrl+C**.

---

## 11. Open the browser using whatever port you chose

Replace **`8081`** with the **`listen.port`** you set in YAML.

- **Forensics / operator dashboard:**  
  `http://127.0.0.1:8081/forensics`

- **OpenAI-compatible HTTP API** (for clients that call chat completions):  
  base URL `http://127.0.0.1:8081` and path **`/v1/chat/completions`** (same port).

If the page does not load, check that the proxy started without errors and that no other program is using the same port.

More detail on the operator page: **`docs/forensics-dashboard.md`**. Overview: **`README.md`**.
