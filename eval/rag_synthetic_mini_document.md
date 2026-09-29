# Synthetic RAG mini-corpus (v0)

Single fictional mini-document for **harness design**, not production evaluation. Split on `### Section N` boundaries into 8–12 retrieval chunks (here: **10 sections**).

Design goals: mixed **family / travel / work / medical / timeline / distractor** material; facts placed so some answers need one chunk, two chunks, or cannot be grounded; one **timeline conflict** and one **identity/distractor** ambiguity.

---

### Section 1 — Family bulletin (Metroville Weekly)

Emma Chen told the Metroville Weekly that **her sister moved back from Canada last week**. Emma still lives on Maple Street. The paper did not print the sister’s name.

---

### Section 2 — Household registration update

City records list **Nora Park** as Emma Chen’s sibling. Before returning, Nora was **based in Vancouver** for two years. Nora now uses Emma’s address for mail.

---

### Section 3 — Harbor Labs HR note (internal)

**Nora Park** is employed at **Harbor Labs** on a standard full-time contract. Her manager is Priya Nair. The note covers badge access only.

---

### Section 4 — Cousin Lucy (unrelated move)

Emma’s cousin **Lucy Chen** lives in Montreal. Lucy mentioned a sister in casual conversation at a picnic; **no move from Canada** is described in this file. Lucy works in retail.

---

### Section 5 — Rachel — medical admin line

**Rachel Park** had a follow-up with **Dr. Okonkwo on March 15**. The chart entry lists “routine follow-up” only. No family relationship to Nora or Emma is stated here.

---

### Section 6 — Nora’s text thread (timestamped)

Nora texted Emma: **“Landed March 8, finally home.”** The thread does not name the airport or city of departure.

---

### Section 7 — Family email chain (informal)

Subject: “Welcome back.” The body says Nora **“arrived last week”** and asks Emma to bring lasagna. There is no calendar date in this email.

---

### Section 8 — Travel noise (Emma’s trip)

Emma booked a **Calgary conference** for Q2. This trip is unrelated to Nora’s relocation. The itinerary does not mention Nora.

---

### Section 9 — Timeline index (Metroville civic blog)

The civic blog lists **bus detours on Pine Avenue** starting April 1. It does not mention Nora, Emma, or Canada.

---

### Section 10 — Another “sister” line (distractor)

A community post says **“Lucy’s sister moved to Ottawa last spring.”** It does not name Vancouver, Nora, or Emma. Readers often confuse this with other family news.

---

## Harness intent (not executed here)

Use **`eval/rag_synthetic_mini_document.manifest.yaml`**. Each item separates:

- **`truth_status`** — what the document evidence structure is (not policy).
- **`expected_governance`** — what Lens should do given that structure + policy.
- **`answer_constraint`** — allowed answer shape (exact vs bounded absence vs conflict vs ambiguity).
- **`gold_sections` / `forbidden_sections`** — source section IDs (1-based `### Section N` headings), not assumptions about how a chunker will slice markdown.

String-only match against a single blended label is intentionally avoided.
