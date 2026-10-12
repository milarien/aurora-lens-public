# Language-support admission: assessment not released

2026-10-12: translation development remains stopped. The separate unsupported-language admission task is unresolved. The language-gate prototype was withdrawn from the runtime because it failed the requested boundary. No production language repair was deployed, and the existing governance fixes remain intact.

## Required boundary

Establish English support before ordinary interpretation, clarification binding, answer generation or world-state admission. Unsupported or uncertain requests should remain held and receive: **Please rephrase your question in English.** English names and quoted data must be distinguished from foreign instructions or decision criteria. Neither a plausible English parse nor paragraph-level English identification establishes that every instruction-bearing span is supported.

## Dedicated components assessed

The initial prototype assessed py3langid 0.3.0 and Lingua 2.1.1. It used whole-request identification, mixed-language segmentation and local phrase observations. Both components labelled `Approve the transfer si no hay dos firmas.` English at the whole-request level. Broader regressions exposed English false positives, including short requests. Lingua's documentation explicitly describes its mixed-language detection as experimental and less reliable for short phrases.

The follow-up assessed fasttext-predict 0.9.2.4 with the full published fastText `lid.176.bin` model. Model SHA-256: `7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e`. Local phrase identification countered dilution by an English prefix. Isolated-word labels misidentified English homographs, so contextual phrase evidence was assessed instead. Source text was preserved; classifier preprocessing was not a translated request or admitted replacement. A scoped projection distinguished proper-name/quotation data from governing predicates. No accent, alphabet or punctuation-only language rule, word-specific exemption or translation-fidelity claim was introduced.

## Exact remaining failures

Controlled local runs used the actual Lens interpreter and PEF with an explicitly labelled diagnostic adapter; these were not live-model or production verification. All rows below are reconstructed diagnostics. Results are recorded in local `reports/language-boundary-remaining-misses.json`.

| Input | Proposed boundary | Actual subsequent path |
| --- | --- | --- |
| `Shortlist applicants si son mayores de 50.` | Held | Zero ordinary extraction calls, zero model calls, zero entities/relationships |
| `Ella no aprobo eso si faltaban dos firmas.` | Held | Zero ordinary extraction calls, zero model calls, zero entities/relationships |
| `Emma has dos apples.` | Held | Zero ordinary extraction calls, zero model calls, zero entities/relationships |
| `Emma owns tres cars.` | Held | Zero ordinary extraction calls, zero model calls, zero entities/relationships |
| `Only select applicants who are viejos.` | Incorrectly identified as supported English; score 0.7474 | Ordinary extraction ran, then the pre-existing translation path contained the request; no world entities/relationships were written |
| `Retira all applicants over 50.` | Incorrectly identified as supported English; score 0.6055 | Ordinary extraction and controlled answer generation ran; PASS with one world entity written |

The containment in the penultimate row is not a completed language-admission repair: the input had already crossed the proposed boundary. The last row directly violates the requested no-processing/no-state-write requirement. These findings do not establish a permissible English interpretation of either request.

Focused controls covered normal and stream-requested paths, prior T07 approval/T13 criterion holds, refusal finality, bounded failures, English names/quoted mentions and zero world writes on detected unsupported input. An earlier focused revision passed 133 tests, but the expanded scope matrix reproduced six failures (two missed inputs across identification and both entry paths). Earlier whole-suite prototype runs also failed and are retained as diagnostics; none is a successful release qualification. A few passing inputs do not establish complete coverage.

## Capability limit and remaining work

The assessed identifiers return statistical language predictions. They do not establish the language of every governing action, qualifier, negation, quantity or condition, and paragraph-level English evidence can conceal a foreign instruction-bearing fragment. Mention scoping based on an English parser is also fallible and cannot itself establish language support. Unknown languages, transliteration, very short code switching, shared words, unusual names/quotations and adversarial context remain limitations.

The missing capability is an admissible English-support basis for the instruction-bearing spans, with explicit unsupported/uncertain outcomes and sufficiently verified name/quotation scope. Whole-text classifier confidence alone is not an adequate basis for the requested boundary. A new scoped identification/interpretation-capability contract or a more restrictive supported-input contract would need to be established and tested; neither was silently added. This is separate from translation fidelity. No multilingual governance capability is claimed.

The existing public demo supplies no `translation_required` flag. Its deployed interpreter still lacks a reliable automatic language-support boundary. Unsupported Latin-script/mixed-language input that misses interpretation-limit triggers can still reach ordinary processing and state admission. This deployed gap remains open.

## Preserved evidence and deployed state

The withdrawn implementation and its regression cases are retained locally in `reports/language-support-assessment-20261012/`, including the exact runtime patch, assessor and test source. Model downloads and assessment dependencies remain diagnostic artifacts, not runtime dependencies. Existing local diagnostic files were preserved.

After withdrawal, the runtime, package dependencies, Dockerfile, NOTICE and translation-stage tests match their prior committed versions. The preserved governance/hold regressions passed: 131 tests. Production health and both remote refs were checked on 2026-10-12: deployed/private commit `47f93efdc1588aa6502278d291592b0af9c0ab60`, public main `6a4008a9af9359d472d3fe718372fe247b050d29`. Documentation changes after this checkpoint do not constitute deployment of the rejected language gate.

Primary assessment sources: [fastText model documentation](https://fasttext.cc/docs/en/language-identification.html), [fasttext-predict](https://github.com/searxng/fasttext-predict), [Lingua](https://github.com/pemistahl/lingua-py), [py3langid](https://github.com/adbar/py3langid).
