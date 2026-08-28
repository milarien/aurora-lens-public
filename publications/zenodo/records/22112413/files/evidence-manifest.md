# Evidence Manifest

Companion to: *The Maintained Unresolved State: Priority Record and Documented
Chronology*

Margaret Stokes — ORCID 0009-0004-6422-4174
Manifest prepared: 26 August 2026 (version 2)

---

## Purpose

This manifest fixes the content of the materials underlying the accompanying
record. Deposit of this manifest with a persistent identifier establishes that the
listed artifacts existed in the stated form as at the deposit date. Any later
production of an original can be checked against the digests below; a single
altered byte changes the digest.

Digests are SHA-256, uppercase hexadecimal.

---

## 1. Repository exhibits

Extracted from the PEACE Protocol repository (GitHub account `nsolland`, Apache
License 2.0). Each exhibit carries its source commit identifier, author date and
subject line in its own header.

| Exhibit | File | Bytes | SHA-256 |
|---|---|---|---|
| 1 | `ex1-baseline-a172287.txt` | 502 | `736714D64F949F2AD67AFD34DB23390E3659FA19AC1DE520946669145FEA8CEA` |
| 2 | `ex2-unresolved-4dd26ac.diff` | 20,309 | `8C2E318F6622746BEA3D5F9F1D0CA9AB05C3E19DA3CF5DBF036C05D1D71B77A0` |
| 3 | `ex3-notice-00ecb4f.diff` | 958 | `20BD0B7EF504A871E090A27F810F42677A32A457EB7C0FF77B654EAC58EBA28C` |
| 4 | `ex4-minimize-136fdb5.diff` | 109,022 | `2F81909CEAB55291CCCF98596BC26F3E68B506CF411879E839F062FAA35239C5` |
| 5 | `ex5-commit-log.txt` | 3,278 | `F2F3320BED79769FBAB6CBD9C942C4C05D7427EFD1518F68CE5A5C465766B5BD` |
| 6 | `ex6-redact-8b82872.diff` | 1,228 | `0DD341441754F90C6325B8315437D32F2AC0C58A604287DA214A51FE4F325CE6` |
| 7 | `ex7-hidevocab-d86958b.diff` | 6,890 | `FC4F7D81F45DD1CA93A4A97585EED18EC4B3B04A14233E94B521CA3420493DD6` |
| 8 | `ex8-generalize-5fc445e.diff` | 3,404 | `75497FF138D0F6EC2C4E3798212B6009E02FE044F35AAE9553C1A9015011C81A` |
| 9 | `ex9-fingerprints-a02e118.diff` | 3,000 | `88FC984BCE0EC6ACAFF08BDEC0ACCB5EED642A993573C2259E95E3E4FA0CE6C3` |

**Exhibit contents**

1. Commit `a172287`, 19 Aug 2026 12:11:29 +0200 — "Merge pull request #5 from
   nsolland/cleanup/public-facing-repo". The public baseline. Summary statistics
   only.
2. Commit `4dd26ac`, 19 Aug 2026 14:21:17 +0200 — "Define governed unresolved
   admission semantics". Full diff. Six files, 148 insertions, 11 deletions.
3. Commit `00ecb4f`, 21 Aug 2026 10:08:58 +0200 — "license: name Njål Gaute
   Solland as copyright holder". Full diff. Two files, two lines replaced.
4. Commit `136fdb5`, 22 Aug 2026 09:02:27 +0200 — "Minimize public PEACE surface".
   Full diff. Fifteen files, 233 insertions, 1,732 deletions.
5. Complete commit log from `a172287` to `68e1a43`, in chronological order, with
   full commit identifiers, author dates, author names and subject lines.
6. Commit `8b82872`, 25 Aug 2026 19:55:17 +0200 — "redact internal work anchor from
   public surface". Full diff.
7. Commit `d86958b`, 25 Aug 2026 20:07:40 +0200 — "hide protected research
   vocabulary from public publication gate". Full diff.
8. Commit `5fc445e`, 26 Aug 2026 09:07:09 +0200 — "Generalize PEACE non-public
   categories to avoid research-direction map". Full diff.
9. Commit `a02e118`, 26 Aug 2026 09:07:33 +0200 — "Remove reversible protected-term
   fingerprints from public validator". Full diff.

---

## 2. Repository clones

Three complete clones of the public PEACE Protocol repository are retained by the
author. A git commit identifier is a cryptographic digest over the commit's tree
and full ancestry; the identifiers below therefore fix the content of each clone
in its entirety.

| Clone | Head commit | Head date | Commits in history |
|---|---|---|---|
| Baseline, taken 19 Aug 2026 prior to the 22 Aug removals | `a172287450d7a07a58d4668c23d0c2466b2fd70c` | 2026-08-19 12:11:29 +0200 | 18 |
| Intermediate state, taken 25 Aug 2026 | `68e1a43a2189fefe2963d6e71500828c8ae9ae9e` | 2026-08-25 13:54:31 +0200 | 42 |
| Current state, taken 26 Aug 2026 | `a02e118b02103838aac5ec075c7038e73b857464` | 2026-08-26 09:07:33 +0200 | 46 |

Twenty-eight commits separate the baseline from the current state.

A third clone, of the Aurora-Lens repository formerly held under the `nsolland`
account and confirmed deleted by its holder on 24 August 2026, is also retained by
the author with complete commit history.

---

## 3. Correspondence

Complete export of the LinkedIn message thread between the parties, 12 June 2026
to 25 August 2026.

| File | Bytes | SHA-256 |
|---|---|---|
| `Njal Margaret convo.txt` | 361,533 | `E368F1979732803D1DC73D40FEF4982A1B404AFC16D3AD838F1602992FEB0C85` |

The export is not published with this deposit. The digest permits the original to
be produced later and shown to be unaltered.

---

## 4. Timezone conversion basis

Repository commit dates are recorded in Central European Summer Time (UTC+2).
LinkedIn displayed message timestamps in the reader's local timezone, Australian
Eastern Standard Time (UTC+10). An eight-hour offset has been applied throughout
the accompanying record, converting displayed message times to CEST.

The conversion is corroborated within the artifacts themselves: the message
transmitting the repository link is displayed at 20:01 AEST, converting to 12:01
CEST, and the baseline merge `a172287` is recorded at 12:11:29 +0200 — an interval
of ten minutes.

---

## 5. Patent records

The following Australian provisional applications are cited in the accompanying
record and are held by IP Australia. They are not reproduced in this deposit;
their content is verifiable through that registry.

| Application | Filed |
|---|---|
| 2025905835 | 27 November 2025 |
| 2025905860 | 28 November 2025 |
| 2025905885 | 29 November 2025 |
| 2025906132 | 8 December 2025 |
| 2025906680 | 19 December 2025 |

---

## 6. Derivation test materials

The prompt as administered and the complete, unedited output of each run, captured
26 August 2026.

| File | Bytes | SHA-256 |
|---|---|---|
| `Prompt 26 Aug 2026.txt` | 2,393 | `E89BCEB8B57ECC2098D35ECE65621154AE72CBDFE17362F1DDE04A0F1F32E4B4` |
| `Grok 26 Aug 2026.txt` | 8,706 | `9D35D1077A12F8CD014768200D7DB6659D5AFF93A0B329A8B6556539C39B102D` |
| `Claude 26 Aug 2026.txt` | 8,491 | `C06B5D5A6F8998BCC05EF2099046AE01A324A3757E9EC70934F489A50ADDCE9D` |
| `Gemini 26 Aug 2026.txt` | 5,744 | `B22DE5A3F55F087EFADC3CE4212AB694B21854D0BD2D167ADC68425BCB40516F` |
| `Chatgpt temporary chat 26 Aug 2026.txt` | 25,526 | `59D5A36C20AF7D211880FE51D35350EDBC4C614F09CF0B0BD2020C2DBFE0C385` |
| `Chatgpt contaminated 26 Aug 2026.txt` | 12,973 | `3D894FFC7F6A20F62DF01E4C6A7A3FD3C81CC779EC2BF74C4B6B81AF8C2A314D` |

**Run conditions.** The Grok, Claude, Gemini and ChatGPT-temporary-chat runs were
cold: fresh sessions with no stored memory, custom instructions or prior context.
The ChatGPT temporary chat was conducted in a mode that does not use or update
stored memory.

The run identified as contaminated was conducted in an account with persistent
memory and custom instructions active, holding extensive prior working context on
the author's architecture. It is reported in the accompanying record as a
deliberately contaminated condition and as a control against the cold ChatGPT run,
the two differing only in whether stored context was retrievable.

The scoring criteria were registered before the runs and are set out in §5 of the
accompanying record. A subsequent version will report an expanded run set.

---

## 7. Verification

For any file listed above, compute the SHA-256 digest and compare it to this
manifest. On Windows PowerShell:

```
Get-FileHash -Algorithm SHA256 -Path "<file>"
```

For the repository clones, confirm the head commit identifier:

```
git -C "<clone>" rev-parse HEAD
git -C "<clone>" rev-list --count HEAD
```

A digest or commit identifier that matches establishes that the artifact is
byte-identical to the one held by the author at the date of this manifest.
