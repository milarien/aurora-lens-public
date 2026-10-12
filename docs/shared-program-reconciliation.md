# Shared-program reconciliation and release contract

This supersedes earlier claims of whole-program equivalence. Production runs private source; a selective repair copy never established complete equivalence. The two installations remain distinct. No tree-wide synchronization was used.

## Proven divergence history

| Difference | Introducing evidence | Why it survived synchronization |
|---|---|---|
| Expiry residue storage | Public `71cc5a962b5ad17b5cdc4c29da0a5e669239a10b`, 2026-10-06 | The helper and session/store bridge were not included in selective later repair ports. Private had no separate hold store. |
| Typed T13 context on expiry | The older public residue schema predates hiring-criterion typing | Later PEF/Lens ports did not extend the expiry residue schema. Pending clarification alone did not preserve its typed criterion. |
| Concealment detection and local illicit scope | Public `8b97015599923328a0e2a37a521a821f9a011ea7`, 2026-09-29; private integration `ff80cc40e21f97ae04cfa0bb9be4bb4f43b4cdd9`, 2026-10-10 | The selected financial repair integration omitted expanded targets and localized helper. Private `585ad9892c63ed12835938182f7f198614e27a26`, 2026-10-11, reintroduced a response-wide explicit-illicit early return. |
| Hosted versus licensed local edge gate | Public `86bfaf4ae91e40e1d27c06748b24fa98fd60fe44`, 2026-10-05 | Intentional local installation repair was not represented in an explicit parity contract. It remains permitted and pinned. |
| Non-loopback audit inbound auth | Public `22dad8fec6bf5b6249360f73d309f2f59500d482`, 2026-10-05 | Public hardening was not ported to private. Disabled inbound API auth in production also made that change insufficient for anonymous protection. Both now have the same hosted forensic edge gate. |
| Managed launcher/licensing/configuration | Earliest public snapshot `225e9b4e50b61dd2145c81543ff520c057e708cb`, 2026-09-27 | These pre-existing installation differences were not inventoried by selective repair checks. History cannot establish their earlier private development origin from this snapshot. |

Private CI previously had no peer-program inventory. The ship script compared private source to a staging ship tree, not public to private. Neither detects omissions from a selected repair list. Earlier claims of matching trees were therefore unsupported. Exact before-repair diffs, path histories and introducing-commit patches are retained in the local `reports/reconciliation-history/` and `reports/repository-divergence-*` evidence.

## Complete gate

The contract inventories every file under `aurora_lens/` and `governor/`, plus explicitly enumerated root build/startup files and the verifier. Text normalization is CRLF to LF only. Bytecode/cache, backup `.bak` files and coding-agent metadata directories `.claude`, `.agents`, `.codex` are excluded as non-program artifacts. Unexpected or missing program files fail. Shared assets and governance code must have the same fingerprint. Installation exceptions require exact per-repository hashes and reasons; they are not wildcard exclusions. Two Python files project only exact hash-pinned installation nodes; every remaining AST node is compared. Changing an exempted installation node also fails until an explicit contract review.

Private CI performs a full paired immutable-revision check; public CI checks the same complete contract without requiring private repository credentials. Both Docker builds fail on a contract mismatch before installation. The ship command also gates before writing staging files. A one-sided check is labelled one-sided, never a repository match. Deployment verification requires the health-reported commit to equal the explicitly checked private Git revision and compares that revision against an explicit public revision. These are source release gates, not a guarantee of identical environments/dependencies or a GitHub branch-protection claim.

Required release command:

```powershell
python scripts/verify_shared_program.py --role private --ref PRIVATE_SHA --peer C:\Git\Aurora-Lens-Public --peer-ref PUBLIC_SHA --deployed-commit PRIVATE_SHA
```

## Explicit permitted installation differences

| Path | Reason | First/last public history evidence |
|---|---|---|
| `.dockerignore` | Distribution-specific build-context exclusions; private governor inputs required by the complete contract are explicitly retained. | 22dad8fec6bf / 22dad8fec6bf |
| `Dockerfile` | Public non-editable licensed installation versus private editable source and config startup. | 225e9b4e50b6 / 2f47ebde9ab9 |
| `aurora-lens.yaml` | Private production startup configuration, strict policy and spaCy extraction; public installations supply their own config. Exact private settings are pinned. | Private-only; public absence pinned |
| `aurora_lens/BUILD_INFO.json` | Public generated distribution build provenance metadata. | 225e9b4e50b6 / 225e9b4e50b6 |
| `aurora_lens/__init__.py` | Distribution version, authorship and licence metadata; no governance rules. | 225e9b4e50b6 / 225e9b4e50b6 |
| `aurora_lens/cli.py` | Public licensed managed launcher and installation diagnostics. | 225e9b4e50b6 / 225e9b4e50b6 |
| `aurora_lens/launcher/controller.py` | Managed installation process lifecycle; pinned public packaging implementation. | 225e9b4e50b6 / 225e9b4e50b6 |
| `aurora_lens/launcher/lifecycle.py` | Public managed installation launcher lifecycle/support component. | 225e9b4e50b6 / 225e9b4e50b6 |
| `aurora_lens/launcher/paths.py` | Public managed installation launcher lifecycle/support component. | 225e9b4e50b6 / 8b9701559992 |
| `aurora_lens/launcher/state.py` | Managed installation state path and typed installation metadata. | 225e9b4e50b6 / 22dad8fec6bf |
| `aurora_lens/launcher/support.py` | Public managed installation launcher lifecycle/support component. | 225e9b4e50b6 / 225e9b4e50b6 |
| `aurora_lens/proxy/__main__.py` | Public managed launcher integration and environment credential resolution. | 225e9b4e50b6 / 225e9b4e50b6 |
| `aurora_lens/proxy/config.py` | Public credential environment-variable names for installation; governance configuration remains pinned. | 225e9b4e50b6 / 225e9b4e50b6 |
| `aurora_lens/proxy/demo.html` | Installed demo explanatory copy differs, not runtime adjudication. | 225e9b4e50b6 / 042944f2847f |
| `aurora_lens/scripts/demo_healthcare.py` | Installation/demo invocation harness only. | 225e9b4e50b6 / 225e9b4e50b6 |
| `aurora_lens/setup/core.py` | Installation setup paths, environment credential names, noneditable install instructions and support bundle build provenance; pinned per role. | 225e9b4e50b6 / 8b9701559992 |
| `aurora_lens_build_backend.py` | Public distribution build metadata and licence checks. | 225e9b4e50b6 / d928ae690d84 |
| `governor/IMPLEMENTATION_REPORT.md` | Private design documentation/test harness; common runtime governor is compared completely. | Private-only; public absence pinned |
| `governor/POLICY_MATRIX_XRAY.md` | Private design documentation/test harness; common runtime governor is compared completely. | Private-only; public absence pinned |
| `governor/tests/test_governor.py` | Private design documentation/test harness; common runtime governor is compared completely. | Private-only; public absence pinned |
| `pyproject.toml` | Public proprietary licence/build backend/package inclusion versus private development distribution. | 225e9b4e50b6 / 86bfaf4ae91e |
| `railway.toml` | Private hosted build/start commands; public licensed installations do not deploy with this Railway configuration. Required verifier precedes the hosted build command. | Private-only; public absence pinned |
| `release_guard.py` | Public distribution licence release guard. | 225e9b4e50b6 / 225e9b4e50b6 |
| `setup.py` | Public distribution build backend bootstrap. | 225e9b4e50b6 / 225e9b4e50b6 |
| `aurora_lens/proxy/app.py` → `function:_public_demo_edge_token_required` | Private retains its historical local chat edge gate; public licensed local installs do not require a hosted credential. Both require the same hosted forensic credential. | Exact AST hashes in contract |
| `aurora_lens/governor/mutations.py` → `function:_get_nlp` | Only the exact spaCy loader installation-error message/guard differs; all mutation governance is compared outside this pinned node. | Exact AST hashes in contract |

## Governance and persistence limits

Conversation TTL now retains uncommitted approval holds, typed hiring criterion/year/basis/clarification evidence and authorised-disposition records separately from conversational world state. A clarification is evidence, not permission to screen. Start-over closes unresolved context explicitly; it cannot authorize a later dependent “Do it” or original-request replay. Memory residue survives expiry within the process; Redis residue keys are non-expiring. A memory-process restart is not durable storage and is not claimed repaired. Translation work remains stopped. English-only admission has known unsupported-input gaps; these repairs do not establish multilingual governance.

The approved production access change requires the existing server-only edge credential on every `/v1/audit/*` and `/v1/session/operator-pef` request before handlers. The public website cannot forward forensic reads with that secret. Anonymous health exposes only status/version/commit; summary exposes aggregate counts. Authenticated direct diagnostics remain supported. No credential is sent to browsers.
