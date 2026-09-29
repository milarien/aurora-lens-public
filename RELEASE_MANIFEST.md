# Release manifest

**Release:** Aurora-Lens public repository (runtime 3.0.1)  
**Manifest date:** 29 September 2026

This file states what this public tree intentionally includes and excludes.

## Included

| Category | Location |
|----------|----------|
| README | [README.md](README.md) |
| Source licence | [LICENSE](LICENSE) (published source, no licence granted; see [LICENSING.md](LICENSING.md)) |
| Patent notice | [PATENT_NOTICE.md](PATENT_NOTICE.md) |
| Citation metadata | [CITATION.cff](CITATION.cff) |
| Release notes | [CHANGELOG.md](CHANGELOG.md) |
| Architecture (public) | [architecture/](architecture/) |
| Demo and synthetic examples | [tools/run_demo.py](tools/run_demo.py), [examples/](examples/), [eval/](eval/) (fixtures and manifests only) |
| Public-behavior tests | [tests/](tests/) |
| Hash / provenance notes | [provenance/EVIDENCE-MANIFEST.txt](provenance/EVIDENCE-MANIFEST.txt), [provenance/PROVENANCE-TIMELINE.md](provenance/PROVENANCE-TIMELINE.md) |
| Publication corpus | [publications/zenodo/](publications/zenodo/) |
| Runtime source (published for inspection) | [aurora_lens/](aurora_lens/) under [LICENSE](LICENSE) |
| Operator documentation | [docs/](docs/) (public operator and config guides) |

Public patent links are listed in [PATENT_NOTICE.md](PATENT_NOTICE.md).

## Explicitly excluded (not in this repository)

- Environment files, API keys, tokens, and credentials
- Railway or other production hosting configuration
- Private wheels or internal package artefacts under `dist/` / `eval/wheels/`
- Filing receipts and private patent-office correspondence
- Claim-exposure analysis, buyer notes, acquisition memos, and commercial-licence negotiation material
- Runtime audit logs, session exports, and user data
- Local `.venv-test/` and other developer-only trees (gitignored)

Production deployment is operator-local. Configure credentials via environment variables and local YAML; do not commit secrets.

## Verification

- Full automated suite: `pytest tests` (from an isolated virtual environment with `pip install ".[proxy,dev]"`).
- Offline demo: `python tools/run_demo.py`
- Release packaging guard: `python tools/validate_release_clean_audit.py` (requires a clean working tree without local audit artefacts)
