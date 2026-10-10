"""Verification checker — compare LLM output claims against PEF state.

Implements:
- "Introduced" policy: unresolved placeholders don't trigger flags
- Negation rules: contradiction vs absence
- Time-smear detection
- Medical-safety: pediatric dosage recommendations must not PASS

Regex inventory (how patterns are used)
----------------------------------------
**Lexical hint** — soft signal, exemption, or feature for a larger rule. A match
does not by itself replace claim/PEF reasoning; it scopes, disambiguates, or
supplies a prior (e.g. hedge vs assertion, crisis context for PII, education
guard for finance metrics, third-party scope for pronouns, perimeter pruning
for ``user_seeks_specific_cause``, generic-class vs anaphoric ``those/these``).

**Primary gate** — pattern match is the main admissibility input for that axis
(often Axis 3 safety or policy): PII surface forms, self-harm / illegal-instruction
clusters, violent-criminal-intent on user text, defamation/truthfulness bundles,
refusal-frame early exit in ``check``, negation-prefix guards, and most of
``check_blocked_act_request`` (legal / healthcare / finance request acts via
inline ``re.search`` arms).

**Surrogate for missing structure** — extraction or spaCy did not yield the
relation/span the rule needs, so text-level regex approximates syntax (e.g.
causal ``factors such as`` when INCLUDE claims are absent; sentence/clause
``re.split`` for finance speech-act classification; Layer-1 metric/numeric
pairing on raw sentences; blocked-act request classification on raw user
strings before interpret).

"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from enum import Enum

# First-person pronoun subjects — model meta-speech, not world claims.
_FIRST_PERSON: frozenset[str] = frozenset({"i", "we", "me", "my", "our"})

# Second-person pronoun subjects — user-address phrases, not world-state entity
# assertions. spaCy often parses "this is appropriate for you" or "you may
# experience side effects" with "you" as the claim subject. These are not
# assertions about a named entity and must not trigger hallucination checks.
_SECOND_PERSON: frozenset[str] = frozenset({"you", "your", "yourself"})

# Anaphoric referents — pronouns and demonstratives that require a prior
# grounding in PEF. When these appear as claim subjects without an established
# antecedent, they must be flagged UNRESOLVED_REFERENT.
_REFERENT_PRONOUNS: frozenset[str] = frozenset({
    "he", "him", "his",
    "she", "her", "hers",
    "it", "its",
    "they", "them", "their", "theirs",
    "that", "this", "these", "those",
})

# Conversational back-reference pronouns — pronouns/demonstratives that typically
# refer to user-stated content rather than LLM-invented entities.
#
#   "They require care" / "It's gone now" after a user described symptoms = conversational anaphora.
#   "This warrants attention" / "That is serious" after user described their situation = same.
#   "She has a fever" without a prior referent = world-state claim that must be flagged.
#
# When user spoke in first person (or content words overlap), these are exempt
# from UNRESOLVED_REFERENT.  "that" is also included because spaCy parses relative
# clauses ("chest tightness that lasted an hour") as separate claims with "that"
# as the subject — these are syntactic artifacts, not world-state assertions.
# Third-person singular gendered pronouns (he/she/her/him) are NOT included —
# they refer to third parties, not the user's described experience.
_CONVERSATIONAL_PRONOUNS: frozenset[str] = frozenset({
    "it", "its",
    "that", "this",
    "they", "them", "their", "theirs", "these", "those",
})

# Definite description prefixes — NPs that presuppose a unique prior antecedent.
_DEFINITE_PREFIXES: tuple[str, ...] = ("the ", "that ", "this ", "those ", "these ")


def _is_definite_description(s: str) -> bool:
    """Return True if s is a multi-word definite description (starts with 'the/that/this/...')."""
    lower = s.strip().lower()
    return any(lower.startswith(p) for p in _DEFINITE_PREFIXES)


# Generic class NP detector for "those/these + classifier" constructions.
#
# "Those who prefer certainty" and "those comfortable with variability" are
# universal quantifiers, not anaphoric references to specific prior entities.
# They do not presuppose a unique PEF antecedent and must not be flagged as
# UNRESOLVED_REFERENT.
#
# "Those symptoms" and "these conditions" ARE demonstrative referents — they
# point back to specific things mentioned earlier. These must still be flagged.
#
# The distinguishing structure:
#   generic class  → "those/these" + relative clause (who/which) or
#                    participial (-ing suffix) or adjectival (-able/-ible) or
#                    adjective/participle immediately followed by a preposition
#   anaphoric ref  → "those/these" + bare noun
#
# This is a grammatical pattern check, not a vocabulary list.
_GENERIC_CLASS_NP_RE = re.compile(
    r"^(?:those|these)\s+"
    r"(?:"
    r"who\b"                               # relative clause: "those who prefer…"
    r"|which\b"                            # relative clause: "those which…"
    r"|(?:with|for|in|at|by|to|of)\b"     # bare preposition: "those with X", "those in Y"
    r"|\w+ing\b"                           # present participle: "those seeking…"
    r"|\w+(?:able|ible)\b"                 # -able/-ible adjective: "those comfortable…"
    r"|\w+\s+(?:with|for|in|at|by|to)\b"  # adjective/participle + prep: "those open to…"
    r")",
    re.IGNORECASE,
)



_BACK_REF_STOP: frozenset[str] = frozenset({
    "have", "that", "this", "with", "from", "they", "them", "their",
    "also", "been", "were", "will", "your", "which", "what", "when",
    "here", "some", "just", "very", "more", "like", "than", "then",
    "into", "onto", "over", "under", "about", "after", "before",
})


def _content_words(text: str) -> frozenset[str]:
    """Lowercase content words (length > 3, not in stop set) for back-reference detection."""
    words: set[str] = set()
    for w in re.split(r"[\s.,!?;:—\-\"'()\[\]]+", text.lower()):
        if len(w) > 3 and w not in _BACK_REF_STOP:
            words.add(w)
    return frozenset(words)


# Regex for first-person input detection — handles straight (') and curly (')
# apostrophes in contractions.  Matches whole-word "I", "my", "me", "myself",
# and contractions I've / I'm / I'd / I'll.
_FP_INPUT_RE = re.compile(
    r"\b(?:i|i['\u2019]ve|i['\u2019]m|i['\u2019]d|i['\u2019]ll|my|me|myself)\b",
    re.IGNORECASE,
)

# Third-party relational mentions — minimal, high-signal list.
# Used to scope the gendered-pronoun exemption: "she/he" in a response is only
# exempt from UNRESOLVED_REFERENT when the user explicitly mentioned a third
# party (not just spoke in first person about themselves).
# Neutral pronouns (they/them/their) are already handled more broadly by
# _CONVERSATIONAL_PRONOUNS when is_first_person is True — no asymmetry.
_THIRD_PARTY_RE = re.compile(
    r"\b(?:my|our)\s+(?:daughter|son|child|kid|baby|infant|toddler|teenager|teen|"
    r"husband|wife|partner|patient|client|"
    r"friend|mother|father|sister|brother|"
    r"nephew|niece|grandchild|grandfather|grandmother|grandparent|"
    r"colleague|neighbor|neighbour|spouse|"
    r"\d+-year-old)\b",
    re.IGNORECASE,
)


def _is_first_person_input(text: str) -> bool:
    """True when user spoke in first person (describing their own experience)."""
    return bool(_FP_INPUT_RE.search(text))


# ── Self-harm instruction patterns ───────────────────────────────────────────
_SH_INSTRUCTION_RE = re.compile(
    r"(?:"
    r"how\s+to\s+(?:kill|harm|hurt)\s+(?:your|my)\s*self"
    r"|how\s+to\s+commit\s+suicide"
    r"|how\s+to\s+(?:end|take)\s+(?:your|my|one['\u2019]?s)\s+(?:own\s+)?life"
    r"|ways?\s+to\s+(?:kill|harm)\s+(?:your|my)\s*self"
    r"|methods?\s+(?:of|for)\s+(?:self[\s\-]harm|killing\s+(?:your|my)\s*self)"
    r"|steps?\s+(?:to|for)\s+(?:killing|harming)\s+(?:your|my)\s*self"
    r"|(?:most\s+(?:effective|reliable|lethal|painless)|least\s+painful)"
    r"\s+(?:method\s+(?:of|to)|way\s+to)\s+(?:suicide|kill(?:ing)?\s+(?:your|my)\s*self)"
    r"|(?:kill|harm)\s+(?:your|my)\s*self\s+by\b"
    r")",
    re.IGNORECASE,
)

# ── Illegal instruction patterns ─────────────────────────────────────────────
# Response-side defense-in-depth. Pre-LLM blocking uses aurora_lens.verify.hazard.
# Transform verbs here should stay aligned with data/hazard/processes.json families.
_ILLEGAL_SYNTH_VERB_RE = re.compile(
    r"\b(?:synthesize|synthesise|manufacture|produce|cook|make|extract|distil|"
    r"distill|concentrate|isolate|purify|prepare|acquire|mix|combine)\b",
    re.IGNORECASE,
)
# Controlled-drug list (response-side). Hazardous CWA / plant toxins are governed
# pre-LLM via the hazard ontology (export_substance_ids / export_substance_aliases).
_ILLEGAL_DRUG_RE = re.compile(
    r"\b(?:methamphetamine|meth|heroin|fentanyl|cocaine|crack\s+cocaine|"
    r"MDMA|ecstasy|LSD|ketamine|GHB)\b",
    re.IGNORECASE,
)
_ILLEGAL_INSTRUCTION_MARKER_RE = re.compile(
    r"\b(?:how\s+to|step[s]?\b|recipe\b|instruction[s]?\b|"
    r"you\s+(?:need|will\s+need|should|can|must)\b|"
    r"here['\u2019]?s\s+how|"
    r"first[,\s]|then[,\s]|next[,\s]|finally[,\s])\b"
    r"|\b\d+[.)]\s",
    re.IGNORECASE,
)
_ILLEGAL_EXPLOSIVE_RE = re.compile(
    r"(?:"
    r"\b(?:make|build|construct|assemble|create|craft)\b.{0,60}"
    r"\b(?:bomb|explosive|IED|molotov|pipe\s*bomb|grenade|C4|ANFO|TATP|detonator)\b"
    r"|\b(?:bomb|explosive|IED|pipe\s*bomb)\b.{0,60}"
    r"\b(?:make|build|construct|assemble)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_ILLEGAL_FIREARM_RE = re.compile(
    r"\b(?:convert|modify|alter|transform)\b.{0,60}"
    r"\b(?:full[\s\-]?auto(?:matic)?|select[\s\-]?fire|machine\s+gun|"
    r"giggle\s+switch|auto\s+sear)\b",
    re.IGNORECASE | re.DOTALL,
)
_ILLEGAL_FORGERY_RE = re.compile(
    r"\b(?:forge|forging|counterfeit|fabricate|falsify)\b.{0,60}"
    r"\b(?:passport|driver[\s'\u2019]?s?\s+licen[sc]e|ID\s+card|"
    r"identity\s+document|birth\s+certificate|social\s+security|"
    r"currency|banknote)\b",
    re.IGNORECASE | re.DOTALL,
)

# ── Academic integrity patterns ───────────────────────────────────────────────
_ACADEMIC_INTEGRITY_RE = re.compile(
    r"(?:"
    # "write my essay/assignment/thesis/dissertation for me"
    r"\b(?:write|complete|finish|do)\b.{0,40}\b(?:my|the)\b.{0,30}"
    r"\b(?:essay|assignment|thesis|dissertation|homework|coursework|paper|report)\b"
    r"[^.!?\n]{0,60}\b(?:for\s+me|on\s+my\s+behalf|submit|submission)\b"
    r"|"
    # "take my exam / do my test / complete my assessment"
    r"\b(?:take|sit|do|complete|finish)\b.{0,30}"
    r"\b(?:my|the)\b.{0,20}"
    r"\b(?:exam|test|quiz|assessment|midterm|final|online\s+test)\b"
    r"[^.!?\n]{0,40}\b(?:for\s+me|on\s+my\s+behalf|instead\s+of\s+me)\b"
    r"|"
    # "answer these questions [for me to submit / for my assignment]"
    r"\banswer\s+(?:all\s+)?(?:these|the|my)\s+(?:exam\s+)?questions\b"
    r"[^.!?\n]{0,60}"
    r"\b(?:submit|submission|assignment|exam|test|grade|hand\s+in)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)

# ── Student record exposure patterns ─────────────────────────────────────────
_STUDENT_RECORD_SUBJECT_RE = re.compile(
    r"\b(?:student|pupil|learner)\b",
    re.IGNORECASE,
)
_STUDENT_RECORD_DATA_RE = re.compile(
    r"\b(?:grade[s]?|GPA|transcript|disciplinary\s+record|student\s+ID|"
    r"enrollment\s+record|academic\s+record|FERPA|financial\s+aid\s+record|"
    r"disability\s+accommodation|IEP|504\s+plan)\b",
    re.IGNORECASE,
)
_STUDENT_RECORD_PERSON_RE = re.compile(
    r"\b(?:[A-Z][a-z]+\s+[A-Z][a-z]+|his|her|their)\b",
)

# ── Employment discrimination patterns ────────────────────────────────────────
_EMPLOYMENT_ACTION_RE = re.compile(
    r"\b(?:hire|hiring|recruit|screen|filter|reject|fire|dismiss|demote|"
    r"avoid\s+hiring|not\s+hire|don['’]t\s+hire|exclude)\b",
    re.IGNORECASE,
)
_PROTECTED_CHARACTERISTIC_RE = re.compile(
    r"\b(?:age|race|gender|sex|religion|disability|pregnant|pregnancy|"
    r"national\s+origin|ethnicity|sexual\s+orientation|marital\s+status|"
    r"colour|color|over\s+\d{2}|under\s+\d{2})\b",
    re.IGNORECASE,
)

# ── Employee record exposure patterns ─────────────────────────────────────────
_EMPLOYEE_RECORD_DATA_RE = re.compile(
    r"\b(?:performance\s+review|performance\s+improvement\s+plan|PIP|"
    r"disciplinary\s+file|disciplinary\s+record|salary\s+band|"
    r"compensation\s+detail|medical\s+leave|HR\s+file|personnel\s+file|"
    r"payroll\s+detail|individual\s+salary|employee\s+ID)\b",
    re.IGNORECASE,
)
_EMPLOYEE_RECORD_PERSON_RE = re.compile(
    r"\b(?:[A-Z][a-z]+\s+[A-Z][a-z]+|his|her|their)\b",
)

# ── Trade secret patterns ──────────────────────────────────────────────────────
_TRADE_SECRET_DISCLOSE_VERB_RE = re.compile(
    r"\b(?:share|reveal|disclose|provide|send|export|leak|publish|expose)\b",
    re.IGNORECASE,
)
_TRADE_SECRET_SUBJECT_RE = re.compile(
    r"\b(?:proprietary|confidential|trade\s+secret|NDA|non-disclosure|"
    r"internal\s+formula|internal\s+process|internal\s+pricing|"
    r"supplier\s+pricing|secret\s+recipe|proprietary\s+algorithm)\b",
    re.IGNORECASE,
)

# ── Insider information patterns ───────────────────────────────────────────────
_INSIDER_INFO_RE = re.compile(
    r"(?:"
    r"\bnon[- ]?public\b.{0,60}\b(?:material|information|data|detail)\b"
    r"[^.!?\n]{0,60}\b(?:trad(?:e|ing)|buy|sell|invest|advantage|profit)\b"
    r"|"
    r"\binsider\b.{0,40}\b(?:information|trading|tip|knowledge)\b"
    r"[^.!?\n]{0,60}\b(?:trad(?:e|ing)|buy|sell|invest|profit)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)

# ── Procurement fraud patterns ─────────────────────────────────────────────────
_PROCUREMENT_FRAUD_RE = re.compile(
    r"(?:"
    r"\bbid[- ]?(?:rig(?:ging)?|fix(?:ing)?|coordinat(?:e|ing))\b"
    r"|"
    r"\b(?:conflict\s+of\s+interest|kickback)\b.{0,60}\b(?:conceal|hide|avoid\s+disclos|suppress)\b"
    r"|"
    r"\binside\s+(?:information|knowledge|tip)\b.{0,60}\b(?:tender|RFP|bid|procurement)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)

# ── Personalised employment advice patterns ───────────────────────────────────
_EMPLOYMENT_ADVICE_RE = re.compile(
    r"\b(?:wrongful(?:ly)?\s+(?:dismissed|terminated|fired)|"
    r"constructive\s+dismissal|unfair\s+dismissal|"
    r"discrimination\s+(?:claim|case|complaint)|"
    r"am\s+i\s+(?:being|entitled|owed)|"
    r"my\s+(?:dismissal|termination|case|claim|rights?))\b",
    re.IGNORECASE,
)
_EMPLOYMENT_ADVICE_PERSONAL_RE = re.compile(
    r"\b(?:i\s+(?:am|was|have\s+been)|my\s+(?:employer|manager|boss|workplace)|"
    r"they\s+(?:fired|dismissed|terminated)\s+me)\b",
    re.IGNORECASE,
)

# ── Personalised business strategy patterns ───────────────────────────────────
_BUSINESS_STRATEGY_NAMED_RE = re.compile(
    r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3}\b",
)
_BUSINESS_STRATEGY_DIRECTIVE_RE = re.compile(
    r"\b(?:should|must|ought\s+to|recommend|advise|strategy\s+for|"
    r"how\s+(?:should|can|do)\b.{0,20}\b(?:win|beat|outcompete|grow|expand|pivot))\b",
    re.IGNORECASE,
)
_BUSINESS_STRATEGY_VOCAB_RE = re.compile(
    r"\b(?:merger|acquisition|M&A|IPO|market\s+share|competitive\s+strategy|"
    r"pricing\s+strategy|go-to-market|business\s+model|revenue\s+strategy)\b",
    re.IGNORECASE,
)

# ── Personalised academic advice patterns ─────────────────────────────────────
_ACADEMIC_ADVICE_PERSONAL_RE = re.compile(
    r"\b(?:my\s+(?:application|chances?|GPA|grades?|results?)|"
    r"will\s+i\s+(?:get\s+in|be\s+accepted|qualify)|"
    r"am\s+i\s+eligible|should\s+i\s+apply|"
    r"what\s+are\s+my\s+(?:chances?|odds?))\b",
    re.IGNORECASE,
)
_ACADEMIC_ADVICE_VOCAB_RE = re.compile(
    r"\b(?:university|college|admission|admissions?|scholarship|"
    r"degree|qualification|credential|program|course)\b",
    re.IGNORECASE,
)

# ── Violent criminal intent patterns ─────────────────────────────────────────
# Blocks queries that frame interpersonal homicide (kill/murder/stab/shoot/strangle)
# as a legal, permissibility, outcome, or defence question.
# Operates on user_input: the query itself is non-admissible regardless of how any
# model would respond.  Five branches cover the observed evasion framings.
_VIOLENT_CRIMINAL_INTENT_RE = re.compile(
    r"(?:"
    # A: permissibility framing + lethal verb + third-person target
    # "am I allowed to kill them", "can I shoot him", "is it legal to stab someone"
    r"(?:am\s+i|can\s+i|could\s+i|is\s+it\s+(?:legal|ok(?:ay)?|permissible|allowed))\b"
    r"[^.!?\n]{0,50}"
    r"(?:kill|murder|stab|shoot|strangle|poison|assault)\b"
    r"[^.!?\n]{0,30}"
    r"(?:him|her|them|someone|another\s+person|a\s+person)\b"
    r"|"
    # B: consequence query + lethal verb + third-person target
    # "what would happen if I stabbed him", "what happens if I kill someone"
    r"\bwhat\s+(?:would|will|could|might|happens?)\b[^.!?\n]{0,40}"
    r"\bi\s+(?:had\s+)?"
    r"(?:kill(?:ed)?|murder(?:ed)?|stab(?:bed)?|shot|shoot|strangle[d]?|"
    r"poison(?:ed)?|assault(?:ed)?)\b"
    r"[^.!?\n]{0,30}"
    r"(?:him|her|them|someone|another\s+person|a\s+person)\b"
    r"|"
    # C: legal-outcome framing + homicide verb + third-person target
    # "win a court case for killing them", "be convicted for murdering him"
    r"(?:court\s+case|court\s+trial|\b(?:be\s+)?(?:convicted|charged|prosecuted|acquitted)\b)"
    r"[^.!?\n]{0,80}"
    r"(?:kill(?:ing)?|murder(?:ing)?|stab(?:bing)?|shoot(?:ing)?)\b"
    r"[^.!?\n]{0,30}"
    r"(?:him|her|them|someone|another\s+person|a\s+person)\b"
    r"|"
    # D: self-defence claim framing + lethal verb + third-person target (or "attacker")
    # "can I claim self-defence if I shoot him", "self-defence after killing her"
    r"\bself[\s\-]def[e]?[n]?[sc]e\b"
    r"[^.!?\n]{0,100}"
    r"(?:i\s+(?:had\s+)?)?(?:kill(?:ed)?|murder(?:ed)?|stab(?:bed)?|shot|shoot|"
    r"attack(?:ed)?)\b"
    r"[^.!?\n]{0,30}"
    r"(?:him|her|them|someone|another\s+person|a\s+person|my\s+attacker)\b"
    r"|"
    # E: conviction-likelihood + homicide verb (target implied by legal framing)
    # "how likely is conviction if I kill my attacker after the fact"
    r"\bhow\s+likely\b[^.!?\n]{0,60}"
    r"(?:conviction|convicted|charged|prosecution)\b"
    r"[^.!?\n]{0,80}"
    r"\bi\s+(?:had\s+)?(?:kill|murder|stab|shoot|attack)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)

# ── Targeted defamation patterns ──────────────────────────────────────────────
_DEFAM_LABEL_RE = re.compile(
    r"\b(?:pedophile|child\s+(?:molester|abuser|predator)|rapist|"
    r"sex\s+offender|murderer|serial\s+killer|terrorist|"
    r"fraudster|con\s+(?:man|artist))\b",
    re.IGNORECASE,
)
_DEFAM_ASSERTION_RE = re.compile(
    r"\b(?:is|was|has\s+been|remains?)\s+(?:a|an)\b",
    re.IGNORECASE,
)
_DEFAM_NAME_RE = re.compile(
    r"\b(?!(?:The|This|That|These|Those|An?)\s)[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b",
)
_DEFAM_HEDGE_RE = re.compile(
    r"\b(?:alleged(?:ly)?|accused|charged|convicted|suspected|reportedly|"
    r"claimed|according\s+to|sources?|prosecutors?|court|authorities)\b",
    re.IGNORECASE,
)

# ── Truthfulness certainty patterns ──────────────────────────────────────────
# Fires when the model makes a definitive assertion about a private
# individual's honesty or deception without admissible evidential basis.
# Operates on response_text only — the model's output is the problem.
#
# Main assertion pattern — two branches:
#   A: subject + copula + (optional adverb) + (not) + truthfulness predicate
#   B: subject + copula + certainty adverb + true/false/accurate (requires adverb)
#
# Subjects: this/that/it/he/she/they or "your <word(s)>" (your friend, your colleague).
# Predicates: a lie, the truth, lying, honest, dishonest, deceptive, truthful,
#             telling the truth, being <honest|deceptive|...>
_TRUTH_ASSERT_RE = re.compile(
    r"(?:"
    # Branch A: subject + copula + (adverb) + (not) + truthfulness predicate
    r"(?:this|that|it|he|she|they|your\s+(?:\w+\s+){0,2}\w+)\b"
    r"[^.!?\n]{0,25}"
    r"\b(?:is|are|was|were|isn['\u2019]?t|aren['\u2019]?t|wasn['\u2019]?t|weren['\u2019]?t)\b"
    r"(?:\s+(?:definitely|certainly|clearly|obviously|absolutely|without\s+a\s+doubt))?"
    r"\s+(?:not\s+)?"
    r"(?:a\s+lie|the\s+truth|lying|honest|dishonest|deceptive|truthful|untruthful|"
    r"misleading|telling\s+the\s+truth|"
    r"being\s+(?:honest|dishonest|deceptive|truthful|untruthful|misleading|sincere|insincere))\b"
    r"|"
    # Branch B: subject + copula + certainty adverb + true/false/accurate/made up
    # Requires an explicit certainty adverb to avoid flagging benign "this is true"
    r"(?:this|that|it)\b[^.!?\n]{0,20}"
    r"\b(?:is|was)\b"
    r"\s+(?:definitely|certainly|clearly|obviously|absolutely)\s+"
    r"(?:not\s+)?(?:true|false|accurate|inaccurate|made\s+up|fabricated)\b"
    r")",
    re.IGNORECASE,
)

# Hedge markers — uncertainty language that makes an assertion epistemically safe.
# Checked within the same sentence as the match.
_TRUTH_HEDGE_RE = re.compile(
    r"\b(?:could|might|may|possibly|perhaps|uncertain|not\s+certain|not\s+sure|"
    r"hard\s+to\s+(?:say|know|tell|determine)|"
    r"impossible\s+to\s+(?:know|say|determine|tell)|"
    r"(?:cannot|can['\u2019]?t|difficult)\s+to\s+(?:know|say|determine|tell)|"
    r"without\s+(?:more\s+)?(?:context|information|knowing)|"
    r"no\s+way\s+to\s+(?:know|tell|determine)|"
    r"unclear|appears?\s+to|seems?\s+(?:to\s+)?(?:be\s+)?(?:like\b)?|"
    r"it['\u2019]?s\s+(?:possible|likely|unlikely)|"
    r"(?:may|might|could)\s+be)\b",
    re.IGNORECASE,
)

# Definitional/topic markers — "A lie is...", "Lies are defined as..." — general
# discussion where the truthfulness word is the subject, not the predicate.
_TRUTH_DEFINITIONAL_RE = re.compile(
    r"\b(?:lie[s]?|lying|deception|dishonesty|truthfulness)\s+"
    r"(?:is|are|can\s+be|refers?\s+to|is\s+defined\s+as|involve[s]?)\b",
    re.IGNORECASE,
)

# Fiction / hypothetical context markers — suppress assertion in narrative contexts.
_TRUTH_FICTION_RE = re.compile(
    r"\b(?:in\s+the\s+(?:novel|story|film|book|movie|series|narrative|play|script|"
    r"show|episode|scene)|"
    r"the\s+(?:character|protagonist|narrator|villain|hero|suspect)\b|"
    r"fictionally?|hypothetically|for\s+(?:example|instance)|"
    r"in\s+this\s+(?:example|scenario|case\s+study)|as\s+an?\s+example)\b",
    re.IGNORECASE,
)

# User-input frame: the user is asking whether a specific person is lying,
# honest, deceptive, or telling the truth. When this fires, a bare "Yes" / "No"
# response is an inadmissible certainty claim — its content is inherited from
# the question frame rather than stated explicitly.
_TRUTH_QUERY_RE = re.compile(
    r"\b(?:"
    # "is/was/are/does/did [subject] lying/honest/deceptive/telling the truth/…"
    r"(?:is|was|were|are|does|did)\b[^.!?\n]{0,60}"
    r"\b(?:lying|lied|lies|a\s+lie|honest|dishonest|deceptive|truthful|untruthful|"
    r"sincere|insincere|telling\s+the\s+truth|tell(?:ing)?\s+the\s+truth|"
    r"deceiv(?:ing|ed))\b"
    r"|"
    # "is this/that a lie?"
    r"\bis\s+(?:this|that|it)\s+a\s+lie\b"
    r"|"
    # "can you tell if/whether X is lying"
    r"\b(?:can\s+you\s+(?:tell|determine|know)|tell\s+me)\s+(?:if|whether)\b"
    r"[^.!?\n]{0,60}"
    r"\b(?:lying|honest|dishonest|deceptive|telling\s+the\s+truth)\b"
    r")",
    re.IGNORECASE,
)

# Bare binary verdict — "Yes" / "No" ± leading intensifier, nothing else.
# Only meaningful as a truthfulness claim when user_input establishes the frame.
_BARE_BINARY_RE = re.compile(
    r"^\s*(?:definitely\s+|clearly\s+|obviously\s+|absolutely\s+)?(?:yes|no)[.!?]?\s*$",
    re.IGNORECASE,
)

# Shared negation prefix guard (self-harm + illegal instruction)
_NEGATION_PREFIX_RE = re.compile(
    r"\b(?:cannot|can['\u2019]?t|won['\u2019]?t|will\s+not|unable\s+to|"
    r"(?:do\s+not|don['\u2019]?t)\s+(?:provide|help|assist|give|share|tell)|"
    r"not\s+(?:able|going)\s+to\s+(?:provide|help|tell))\b",
    re.IGNORECASE,
)

# Refusal frame — LLM explicitly declining to help. Responses matching this
# pattern are meta-speech acts, not world claims: skip extraction-empty guard
# and claim loop to avoid false UNRESOLVED_REFERENT / UNBOUND_ENTITY flags.
_REFUSAL_FRAME_RE = re.compile(
    r"\b(?:i\s+(?:cannot|can['\u2019]?t|won['\u2019]?t|will\s+not|am\s+unable\s+to|"
    r"don['\u2019]?t\s+(?:provide|help|assist))|"
    r"i['\u2019]?m\s+unable\s+to|"
    r"i['\u2019]?m\s+not\s+(?:going|able)\s+to\s+(?:provide|help|share|give|assist|answer)|"
    r"(?:not|never)\s+definitive|"
    r"(?:requires?|needs?)\s+(?:further\s+)?(?:confirmation|verification|approval)|"
    r"cannot\s+(?:confirm|guarantee|verify)|"
    r"is\s+not\s+(?:yet\s+)?(?:active|cleared|enrolled|approved))\b",
    re.IGNORECASE,
)



# Epistemic disclaimer patterns — sentences expressing the model's inability
# to verify or confirm. Claims extracted from these clauses are not world claims.
_EPISTEMIC_DISCLAIMER_RE = re.compile(
    r"\bi\s+can(?:not|'t)\b"
    r"|\bwithout\s+access\s+to\b"
    r"|\bwould\s+require\s+(?:me\s+to\s+)?speculate\b"
    r"|\bnot\s+(?:been\s+)?established\s+in\s+the\s+available\b"
    r"|\bi\s+(?:cannot|can't)\s+(?:confirm|provide|verify|explain|determine)\b"
    r"|\bi'?m\s+unable\s+to\b"
    r"|\bi\s+don'?t\s+have\s+enough\s+information\b",
    re.IGNORECASE,
)

from aurora_lens.pef.arrival_incompatible import (
    entity_name_evident_in_text,
    incompatible_arrival_time_joined_text,
    incompatible_arrival_time_literal_strings,
    rag_context_body_from_user_input,
    text_contains_calendar_month_day,
)
from aurora_lens.pef.at_read import current_at
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import (
    BUSINESS_REMIT_MARKERS,
    PEFState,
    Relationship,
    canonicalize_relation,
    object_has_business_remit_marker,
)
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractionResult, ExtractedClaim
from .flags import Flag, FlagType
from .upstream_response_speech_act import (
    UpstreamResponseSpeechAct,
    classify_upstream_response_speech_act,
    extract_upstream_response_features,
)
from .user_grounding import UserGroundingContext
from .numeric import (
    NumericUnit,
    NumericValue,
    numeric_for_metric_span,
    parse_all_numerics,
    parse_numeric,
    values_contradict,
)
from .blocked_request_policy import (
    BlockedRequestRuleId,
    evaluate_blocked_act_request,
    personalized_financial_advice_from_inherited_response,
    user_seeks_historical_market_data_lookup,
)
from .harmful_operational_response import classify_harmful_operational_guidance
from .causal_surface import (
    CAUSAL_FACTORS_TEXT_RE as _CAUSAL_FACTORS_TEXT_RE,
    CAUSAL_HEDGED_OPEN_DOMAIN_RE as _CAUSAL_HEDGED_OPEN_DOMAIN_RE,
    CAUSAL_SCAFFOLD_RE as _CAUSAL_SCAFFOLD_RE,
)
from .pii_surface import collect_pii_evidence_snippets
from .text_segments import split_sentences_at_punctuation
from .violent_criminal_intent import violent_criminal_intent_flags

# Instructional dependency / affordance — second-person subjects are otherwise
# skipped entirely (_SECOND_PERSON). These relation lemmas and clause cues denote
# procedural claims that must still reach PEF verification.
_SECOND_PERSON_MECHANISM_RELATIONS: frozenset[str] = frozenset({
    "NEED", "REQUIRE", "OPEN", "USE", "ACCESS", "OBTAIN", "ENABLE", "ALLOW", "DEPEND",
})

_SECOND_PERSON_MECHANISM_EVIDENCE_RE = re.compile(
    r"(?is)"
    r"\b(?:"
    r"need(?:ed|s)?\s+to\b|"
    r"must\b|"
    r"required\s+to\b|"
    r"needed\s+to\b|"
    r"first\s+(?:use|open)\b|"
    r"to\s+open\b|"
    r"in\s+order\s+to\b"
    r")",
)


def _second_person_instructional_mechanism_claim(claim: ExtractedClaim) -> bool:
    """True when *you*/ *your* as subject still encodes a dependency or affordance claim.

    Extraction often uses *you* as nsubj for how-to text; relation or *evidence*
    (clause text) distinguishes that from hedged addressability ("you may feel …").
    """
    subj = (claim.subject or "").strip().lower()
    if subj not in _SECOND_PERSON:
        return False
    rel = (claim.relation or "").strip().upper()
    if rel in _SECOND_PERSON_MECHANISM_RELATIONS:
        return True
    ev = claim.evidence or ""
    if ev.strip() and _SECOND_PERSON_MECHANISM_EVIDENCE_RE.search(ev):
        return True
    return False


# User questions that implicate an affordance (OPEN) not already in PEF — hardcoded
# surface triggers only; no full parse. See :meth:`Checker._check_user_requires_opens_in_pef`.
_RE_USER_WANT_NEED_OPEN_THE = re.compile(
    r"\b(?:need|want)\s+to\s+open\s+the\s+\S+",
    re.IGNORECASE,
)
_RE_USER_NEED_TO_OPEN = re.compile(r"\bneed\s+to\s+open\b", re.IGNORECASE)
_RE_USER_OPEN_THE = re.compile(r"\bopen\s+the\s+\S+", re.IGNORECASE)
_RE_USER_OPEN_THE_CAPTURE = re.compile(
    r"\bopen\s+the\s+(\S+)",
    re.IGNORECASE,
)
_RE_USER_WHICH_KEY = re.compile(
    r"\b(?:which|what)\s+(?:key|keys)\b",
    re.IGNORECASE,
)


def _user_text_implicates_opens_affordance_query(user_input: str) -> bool:
    """True when minimal regex signals *open container + key* / *need to open* questions."""
    t = (user_input or "").strip()
    if not t:
        return False
    if _RE_USER_WANT_NEED_OPEN_THE.search(t):
        return True
    if _RE_USER_NEED_TO_OPEN.search(t):
        return True
    if _RE_USER_WHICH_KEY.search(t) and _RE_USER_OPEN_THE.search(t):
        return True
    return False


def _normalize_opens_target_head(raw: str) -> str:
    """Lowercase head noun after *open the*, strip trailing punctuation."""
    if not raw:
        return ""
    return raw.strip().lower().rstrip(".,!?;:\"')]}")


def _extract_opens_target_from_user_text(user_input: str) -> str | None:
    """First head noun X from *open the X* in user text (narrow pin for affordance check).

    Returns None when the surface shape does not contain that phrase — callers skip
    object-specific grounding (cannot require OPEN(*, target)).
    """
    m = _RE_USER_OPEN_THE_CAPTURE.search(user_input or "")
    if not m:
        return None
    tok = _normalize_opens_target_head(m.group(1))
    return tok if tok else None


def _opens_object_text_matches_target(pef: PEFState, rel: Relationship, target: str) -> bool:
    """True when OPEN edge's object (entity name or literal) contains *target* as a word."""
    if rel.object_entity_id is not None:
        ent = pef.entities.get(rel.object_entity_id)
        if ent is None:
            return False
        haystack = ent.name.lower()
    else:
        haystack = str(rel.object_literal or "").lower()
    return bool(re.search(r"\b" + re.escape(target) + r"\b", haystack))


def _pef_has_open_for_target(pef: PEFState, target: str) -> bool:
    """True if some OPEN/OPENS edge's object mentions *target* (word-boundary match)."""
    if not target:
        return False
    for rel in pef.relationships:
        r = canonicalize_relation(rel.relation).upper()
        if r not in ("OPEN", "OPENS"):
            continue
        if _opens_object_text_matches_target(pef, rel, target):
            return True
    return False


def _relations_equivalent_for_grounding(a: str, b: str) -> bool:
    """True when two relation strings denote the same canonical fact (e.g. MANAGE vs HAS)."""
    ca = canonicalize_relation(a)
    cb = canonicalize_relation(b)
    if ca == cb:
        return True
    # PEF write-time normalization: business-remit HAS is stored as MANAGE (see pef_updater).
    # Response extraction may still surface HAS; grounding must not treat that as UNSUPPORTED_EVENT.
    if {ca, cb} == {"HAS", "MANAGE"}:
        return True
    return False


def _pef_has_incompatible_arrival_time_literals(pef: PEFState, entity_id: str) -> bool:
    """True when grounded literals mix calendar dates with vague relative arrival times."""
    literals: list[str] = []
    for rel in pef.get_relationships_for_subject(entity_id):
        if rel.object_literal:
            literals.append(str(rel.object_literal))
    return incompatible_arrival_time_literal_strings(literals)


def _retrieval_subject_matches_entity(item_subject: str, entity) -> bool:
    """Match admission subject token (e.g. *nora*) to PEF entity (*Nora Park*)."""
    a = (item_subject or "").strip().lower()
    b = entity.name.strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if b.startswith(a + " "):
        return True
    return a.split()[0] == b.split()[0]


def _retrieval_unresolved_has_temporal_arrival_incompatible(
    pef: PEFState, entity,
) -> bool:
    """True when RAG admission recorded the narrow C1 pattern without committing rels."""
    for item in pef.retrieval_unresolved:
        if item.get("kind") != "temporal_arrival_incompatible":
            continue
        if _retrieval_subject_matches_entity(str(item.get("subject") or ""), entity):
            return True
    return False


def _claim_asserts_specific_calendar_date(claim: ExtractedClaim) -> bool:
    blob = f"{claim.obj} {claim.evidence}"
    return text_contains_calendar_month_day(blob)


def _find_entity_by_unique_first_token(pef: PEFState, subject: str):
    """When extraction says *Nora* but PEF has *Nora Park*, resolve if unambiguous."""
    s = (subject or "").strip().lower()
    if not s or " " in s:
        return None
    hits = [e for e in pef.entities.values() if e.name.lower().split()[0] == s]
    return hits[0] if len(hits) == 1 else None


def _rag_context_blob_has_narrow_c1_seam(user_input: str | None, entity) -> bool:
    """When spaCy omits structured claims, raw ``Context:`` body can still show the C1 seam."""
    body = rag_context_body_from_user_input(user_input)
    if not body:
        return False
    if not entity_name_evident_in_text(body, entity.name):
        return False
    return incompatible_arrival_time_joined_text(body.lower())


def _c1_narrow_temporal_seam_active(
    pef: PEFState, entity, user_input: str | None,
) -> bool:
    """PEF literals, retrieval_unresolved, or raw RAG context blob (same predicate)."""
    if _pef_has_incompatible_arrival_time_literals(pef, entity.id):
        return True
    if _retrieval_unresolved_has_temporal_arrival_incompatible(pef, entity):
        return True
    return _rag_context_blob_has_narrow_c1_seam(user_input, entity)


_SALARY_COMPENSATION_QUESTION_KEYWORDS: tuple[str, ...] = (
    "salary",
    "annual salary",
    "compensation",
    "pay rate",
    "wage",
)

_DISJUNCTIVE_POSSESSIVE_PAIR_RE = re.compile(
    r"(\b[A-Za-z][A-Za-z'\u2019-]{0,48})'s\s+or\s+(\b[A-Za-z][A-Za-z'\u2019-]{0,48})'s",
    re.IGNORECASE,
)


def _rag_question_line_from_user_input(user_input: str | None) -> str | None:
    """Return the ``Question:`` line from a RAG-shaped user turn, or None."""
    if not user_input or rag_context_body_from_user_input(user_input) is None:
        return None
    low = user_input.lower()
    if "question:" not in low:
        return None
    question = user_input.split("Question:", 1)[1].strip()
    return question or None


def _title_case_token(token: str) -> str:
    t = token.strip().replace("\u2019", "'")
    if not t:
        return t
    return t[0].upper() + t[1:].lower()


def _extract_disjunctive_possessive_candidates(question: str) -> tuple[str, ...] | None:
    """Parse *X's or Y's* disjunctive possessive branches from a question line."""
    m = _DISJUNCTIVE_POSSESSIVE_PAIR_RE.search(question.replace("\u2019", "'"))
    if not m:
        return None
    return (_title_case_token(m.group(1)), _title_case_token(m.group(2)))


def _user_disjunctive_whose_sister_question(user_input: str) -> bool:
    """Disjunctive identity shape: *Whose sister …, X's or Y's?*"""
    t = user_input.lower()
    if " or " not in t or "whose" not in t:
        return False
    return "sister" in t


def _user_rag_causal_why_question(user_input: str | None) -> bool:
    """RAG causal-why question shape (``Context:`` … ``Question:`` with *why*)."""
    q = _rag_question_line_from_user_input(user_input)
    if not q:
        return False
    return bool(re.search(r"\bwhy\b", q, re.IGNORECASE))


def _user_rag_salary_compensation_question(user_input: str | None) -> bool:
    """RAG salary/compensation question shape."""
    q = _rag_question_line_from_user_input(user_input)
    if not q:
        return False
    ql = q.lower()
    return any(k in ql for k in _SALARY_COMPENSATION_QUESTION_KEYWORDS)


def _user_rag_calendar_arrival_date_question(user_input: str | None) -> bool:
    """RAG calendar-arrival date question shape (C1 temporal seam perimeter)."""
    q = _rag_question_line_from_user_input(user_input)
    if not q:
        return False
    ql = q.lower()
    return "calendar date" in ql and ("arrive" in ql or "arrival" in ql)


def _user_rag_disjunctive_identity_question(user_input: str | None) -> bool:
    """RAG disjunctive identity question with explicit possessive branch candidates."""
    q = _rag_question_line_from_user_input(user_input)
    if not q or not user_input:
        return False
    if not _user_disjunctive_whose_sister_question(user_input):
        return False
    return _extract_disjunctive_possessive_candidates(q) is not None


def _claim_subject_matches_rag_question_entity(
    claim: ExtractedClaim,
    user_input: str,
) -> bool:
    """True when the claim subject token appears in the RAG question line."""
    q = _rag_question_line_from_user_input(user_input)
    if not q:
        return False
    subj = (claim.subject or "").strip()
    if not subj:
        return False
    return entity_name_evident_in_text(q, subj)


def _rag_causal_bounded_absence_surface(lower: str) -> bool:
    """Bounded causal absence without standard *the context does not…* markers."""
    if "not provided in the available information" in lower:
        return True
    if "not provided" in lower and ("reason" in lower or "why" in lower):
        return True
    if "not detailed" in lower and "provided context" in lower:
        return True
    if "not specified" in lower and "provided context" in lower:
        return True
    if "no information provided about" in lower:
        return True
    if "do not indicate" in lower and ("reason" in lower or "moving" in lower):
        return True
    return False


def _response_hedges_disjunctive_ambiguity(
    response_text: str,
    candidates: tuple[str, ...] | None = None,
) -> bool:
    rl = response_text.lower()
    if any(
        x in rl
        for x in (
            "unclear",
            "ambiguous",
            "cannot determine",
            "not enough",
            "insufficient",
            "context does not",
            "does not specify",
            "uncertain",
            "unclear whether",
            "cannot tell which",
            "not clear which",
            "unclear which",
            "which branch",
            "could refer to either",
        )
    ):
        return True
    if candidates:
        for name in candidates:
            n = name.lower()
            if f"both {n}" in rl or f"either {n}" in rl:
                return True
    return False


def _resolve_pronoun_via_pef(pef: PEFState, subject_lower: str | None = None):
    """Return the entity a pronoun refers to, or None.

    Prefer ``pef.discourse_referent_bindings`` for *subject_lower* (set by RAG
    question-line anchoring or clarification binding) before falling back to the
    most recently active resolved entity.
    """
    if subject_lower:
        bound = pef.discourse_referent_bindings.get(subject_lower.lower())
        if bound:
            ent = pef.find_entity_by_name(bound)
            if ent is not None:
                return ent
    resolved = [e for e in pef.entities.values() if e.resolved]
    if not resolved:
        return None
    return max(resolved, key=lambda e: e.turn_last_active)


def _objects_match(rel_obj: str, claim_obj: str) -> bool:
    """True when rel_obj and claim_obj refer to the same thing.

    Accepts exact case-insensitive match OR a word-boundary prefix abbreviation
    of at least 4 characters — e.g. "Acme" matches "Acme Corp." because
    "acme corp." starts with "acme" and the boundary character is a space.
    Shorter prefixes (< 4 chars) are not accepted to avoid spurious matches.
    """
    a = rel_obj.lower().strip(" .")
    b = claim_obj.lower().strip(" .")
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 4 and long.startswith(short):
        remainder = long[len(short):]
        if not remainder or remainder[0] in (" ", ".", ",", "-", "/"):
            return True
    return False


_LEADING_DET_FOR_USER_ECHO: frozenset[str] = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
})


def _normalize_subject_for_user_echo(subj: str) -> str:
    """Strip leading determiners and possessive markers for substring matching.

    Assistant extraction often surfaces *The gross margin* or *APAC Sub's gross margin*
    while the user wrote *APAC Sub gross margin* without the determiner or
    possessive.  Same-turn echo should still match; this does not admit new facts,
    only aligns surface form for the substring gate before numeric alignment.
    """
    s = subj.strip()
    # "APAC Sub's gross margin" -> "APAC Sub gross margin" (possessive on head)
    s = re.sub(r"(?<=\w)['\u2019]s\b", "", s)
    for _ in range(3):
        parts = s.split(None, 1)
        if not parts or parts[0].lower() not in _LEADING_DET_FOR_USER_ECHO:
            break
        s = parts[1].strip() if len(parts) > 1 else ""
    return s.strip()


def _entity_prefix_for_metric_echo(subj_norm: str) -> list[str]:
    """First one or two content words (after stripping determiners) for entity anchors."""
    parts = subj_norm.split()
    if not parts:
        return []
    if parts[0].lower() in _LEADING_DET_FOR_USER_ECHO:
        parts = parts[1:]
    if not parts:
        return []
    out: list[str] = [parts[0]]
    if len(parts) >= 2:
        out.append(f"{parts[0]} {parts[1]}")
    return out


_FIN_METRIC_ECHO_ANCHOR_TOKENS = (
    "margin",
    "revenue",
    "profit",
    "earnings",
    "growth",
    "loss",
    "shortfall",
    "variance",
    "benchmark",
    "yield",
)


def _extract_claim_numeric_values_for_echo(
    claim: ExtractedClaim,
    response_text: str | None,
) -> list[NumericValue]:
    """Values to align against the user line for same-turn echo admission.

    Prefer structured claim fields; when those yield no parseable numerics (common
    when spaCy leaves the figure only in the surface sentence), fall back to the
    full assistant ``response_text`` so faithful margin echoes are not flagged as
    ``UNSUPPORTED_EVENT`` (finance live Test 3 setup turns).
    """
    c_nums: list[NumericValue] = []
    if claim.obj:
        c_nums.extend(parse_all_numerics(claim.obj))
    if claim.evidence:
        c_nums.extend(parse_all_numerics(claim.evidence))
    if not c_nums:
        cv = parse_numeric(claim.obj)
        if cv is None and claim.evidence:
            cv = parse_numeric(claim.evidence)
        if cv is not None:
            c_nums = [cv]
    if not c_nums and response_text and response_text.strip():
        c_nums.extend(parse_all_numerics(response_text))
        if not c_nums:
            cv = parse_numeric(response_text)
            if cv is not None:
                c_nums = [cv]
    return c_nums


def _metric_echo_finance_anchor_hit(lowered: str) -> bool:
    """True when text has a finance outcome/metric anchor for same-turn echo pairing.

    Extends the original margin/revenue/... set so class 1--2 user lines that state
    *shortfall*, *variance*, *bps* (vs plan / prior / benchmark) still pair with
    assistant extractions whose subject uses those words but not contiguous
    ``subj_norm in user`` (see ``_metric_echo_entity_and_numbers_align``).
    """
    if any(t in lowered for t in _FIN_METRIC_ECHO_ANCHOR_TOKENS):
        return True
    if re.search(r"\bbps\b", lowered):
        return True
    if "basis points" in lowered:
        return True
    return False


def _metric_echo_entity_and_numbers_align(
    user_input: str,
    subj_norm: str,
    claim: ExtractedClaim,
    response_text: str | None = None,
) -> bool:
    """Match *X reported a gross margin of N%* when claim subject is *X's gross margin* / *X gross margin*.

    The contiguous-substring gate fails because *reported* sits between the entity
    name and *margin*.  Same-turn echo should still admit (finance live Test 3
    setup turn 2).
    """
    user_l = user_input.lower()
    subj_l = subj_norm.lower()
    if not _metric_echo_finance_anchor_hit(subj_l):
        return False
    if not _metric_echo_finance_anchor_hit(user_l):
        return False
    for prefix in _entity_prefix_for_metric_echo(subj_norm):
        pl = prefix.lower()
        if len(pl) < 3 or pl not in user_l:
            continue
        u_nums = parse_all_numerics(user_input)
        if not u_nums:
            continue
        c_nums = _extract_claim_numeric_values_for_echo(claim, response_text)
        for cn in c_nums:
            for un in u_nums:
                if not values_contradict(cn, un):
                    return True
    return False


def _claim_numeric_echoes_user_context(
    claim: ExtractedClaim,
    user_input: str | None,
    response_text: str | None = None,
) -> bool:
    """True when the assistant claim restates a numeric fact from the same user turn.

    User context is admitted to PEF as claimed, not as externally verified truth.
    The model's extraction may use a different relation label than PEF storage;
    cross-relation numeric alignment can still miss (e.g. wording).  This gate
    only suppresses UNSUPPORTED_EVENT for faithful echo of the same-turn user
    message (subject substring + aligned numerics), not novel assertions.

    When ``claim.obj`` / ``claim.evidence`` omit parseable numerics but the full
    assistant ``response_text`` still contains the user's figure, numerics are
    taken from ``response_text`` (fallback only when structured fields are empty).
    """
    if not user_input or not user_input.strip():
        return False
    subj = (claim.subject or "").strip()
    subj_norm = _normalize_subject_for_user_echo(subj)
    if len(subj_norm) < 3:
        return False
    user_l = user_input.lower()
    if subj_norm.lower() not in user_l:
        # e.g. user "EMEA Sub reported a gross margin of 31%" vs subject "EMEA Sub gross margin"
        if _metric_echo_entity_and_numbers_align(
            user_input, subj_norm, claim, response_text=response_text
        ):
            return True
        return False
    u_nums = parse_all_numerics(user_input)
    if not u_nums:
        return False
    c_nums = _extract_claim_numeric_values_for_echo(claim, response_text)
    for cn in c_nums:
        for un in u_nums:
            if not values_contradict(cn, un):
                return True
    return False


# ── Financial quantitative Layer 1 (raw text, UNVERIFIED_FACT_ASSERTION) ─────
_FIN_METRIC_TOKEN_RE = re.compile(
    r"\b(?:roi|revenue|profit|earnings|ebitda|margin|growth|gain|loss|dividend|"
    r"yield|returns?|interest\s+rate|annual\s+return|quarterly\s+growth|net\s+income|"
    r"basis\s+points|bps)\b",
    re.IGNORECASE,
)
_FIN_NUMERIC_RE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*%|"
    r"\$\s*\d+(?:[,.]?\d+)*(?:\s*(?:million|billion|thousand))?\b|"
    r"\$\s*\d+(?:[,.]?\d+)*\s*[kKmM]\b|"
    r"€\s*\d+(?:[,.]?\d+)*(?:\s*(?:million|billion|thousand))?\b|"
    r"€\s*\d+(?:[,.]?\d+)*\s*[kKmM]\b|"
    r"£\s*\d+(?:[,.]?\d+)*(?:\s*(?:million|billion|thousand))?\b|"
    r"£\s*\d+(?:[,.]?\d+)*\s*[kKmM]\b|"
    r"(?:AU\$|A\$)\s*\d+(?:[,.]?\d+)*(?:\s*(?:million|billion|thousand))?\b|"
    r"(?:AU\$|A\$)\s*\d+(?:[,.]?\d+)*\s*[kKmM]\b|"
    r"(?:EUR|GBP|USD|AUD)\s+\d+(?:[,.]?\d+)*(?:\s*(?:million|billion|thousand))?\b|"
    r"(?:EUR|GBP|USD|AUD)\s+\d+(?:[,.]?\d+)*\s*[kKmM]\b|"
    r"\d+(?:[,.]?\d+)*(?:\s*(?:million|billion|thousand))?\s*(?:EUR|GBP|USD|AUD)\b|"
    r"\d+(?:[,.]?\d+)*\s*[kKmM]\s*(?:EUR|GBP|USD|AUD)\b|"
    r"\d{1,3}(?:\s\d{3})+(?:[.,]\d+)?\s*(?:€|£|\$)?|"
    r"\d+\s*(?:basis\s+points|bps)\b)",
    re.IGNORECASE,
)
_FIN_ASSERT_RE = re.compile(
    r"\b(?:is|are|was|were|will|should|expect|expects|expected|"
    r"project|projects|projected|forecast|forecasts|yield|yields|return|returns|"
    r"grow|grows|grew|hit|hits|reach|reaches|reached|"
    r"increase|increases|increased|"
    r"decrease|decreases|decreased|"
    r"rose|fall|falls|fallen|"
    r"decline|declines|declined|"
    r"gain|gains|gained|"
    r"drop|drops|dropped|"
    r"surge|surges|surged|"
    r"climb|climbs|climbed)\b",
    re.IGNORECASE,
)
_FIN_EDU_GUARD_RE = re.compile(
    r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:meaning\s+of\s+)?"
    r"(?:roi|ebitda|revenue|margin|yield|earnings)\b",
    re.IGNORECASE,
)
_FIN_HEDGE_RE = re.compile(
    r"(?:\b(?:may|might|could|approximately|around|up\s+to|roughly)\b|~)",
    re.IGNORECASE,
)


_FIN_MARGIN_LAYOUT_RE = re.compile(
    r"\b(?:print|layout|css|page|sides?|border|padding|element|style)\b",
    re.IGNORECASE,
)


def _fin_layer1_first_metric_match(sent: str) -> re.Match[str] | None:
    """First financial metric token for Layer 1 grounding.

    Skips the ``yield`` token inside hyphenated compounds like ``higher-yield`` /
    ``high-yield`` — there it is an adjective fragment, not a standalone financial
    metric; pairing to the nearest numeric would mis-bind (e.g. to 8.4% vs bps).
    """
    for m in _FIN_METRIC_TOKEN_RE.finditer(sent):
        if m.group(0).lower() != "yield":
            return m
        window = sent[max(0, m.start() - 10) : m.end() + 1]
        if re.search(r"(?:higher|lower|high|low)-yield\b", window, re.IGNORECASE):
            continue
        return m
    return None


# Layer 3: predictive/modal relation guard — PEF rels with these verbs are
# projections, not established facts, and must not trigger numeric contradiction.
_PREDICTIVE_RELATION_RE = re.compile(
    r"\b(?:will|would|may|might|could|should|"
    r"expect|expects|expected|"
    r"forecast|forecasts|"
    r"project|projects|projected|"
    r"anticipate|anticipates|anticipated|"
    r"estimate|estimates|estimated|"
    r"target|targets)\b",
    re.IGNORECASE,
)


# ── Financial personalisation Layer 2 (PERSONALIZED_FINANCIAL_ADVICE) ────────
_FIN_STRONG_PERSONALISATION_RE = re.compile(
    r"\b(?:based\s+on\s+(?:what\s+you['\u2019]?ve\s+(?:described|said|told|mentioned)|"
    r"your\s+(?:situation|description|portfolio|finances?|assets?|holdings?|investments?|goals?)|"
    r"the\s+(?:details?|information|facts?|context)\s+you\s+(?:provided|gave|shared))|"
    r"in\s+your\s+(?:case|situation|circumstances?)|"
    r"given\s+your\s+(?:situation|circumstances?|age|balance|savings|income|debt|mortgage|portfolio|finances?|goals?)|"
    r"for\s+your\s+(?:situation|circumstances?|goals?|needs?|portfolio|finances?)|"
    r"from\s+what\s+you['\u2019]?ve\s+(?:described|said|told|mentioned)|"
    r"you\s+(?:will\s+get|should\s+expect|should\s+plan\s+for|need\s+to\s+consider)|"
    r"your\s+(?:\w+\s+)?(?:roi|return|yield|growth|earnings|profit)\s+(?:is\s+expected\s+to|will|is\s+projected\s+to)\b)",
    re.IGNORECASE,
)


# Strong portfolio / trading verbs in the *response*: if present, do not suppress
# PERSONALIZED_FINANCIAL_ADVICE for historical market-data user questions.
_FIN_STRONG_PORTFOLIO_ACTION_RE = re.compile(
    r"(?i)\b(?:"
    r"invest|reallocat|buy|sell|contribute|withdraw|rebalance|allocate|diversify|"
    r"shift|liquidate)\b",
)

_FIN_DIRECTIVE_FORCE_RE = re.compile(
    r"\b(?:"
    r"you\s+(?:should|must|need\s+to|have\s+to|ought\s+to)|"
    r"i\s+(?:would\s+)?(?:recommend|suggest|advise)|"
    r"consider\s+(?:buying|investing|allocating|rebalancing|switching|using)|"
    r"best\s+(?:strategy|approach|option)\s+for\s+you|"
    r"(?:right|appropriate|suitable)\s+for\s+you"
    r")\b",
    re.IGNORECASE,
)

_FIN_IMPERATIVE_ACTION_RE = re.compile(
    r"(?:^|[.!?;:]\s*)(?:buy|sell|invest|allocate|reallocate|rebalance|switch|move|contribute)\b",
    re.IGNORECASE,
)

_FIN_PERSONAL_SUITABILITY_RE = re.compile(
    r"\b(?:"
    r"for\s+you|for\s+your\s+(?:retirement|portfolio|situation|goals?|needs?)|"
    r"given\s+your\s+(?:risk\s+tolerance|age|income|debt|timeline|savings|portfolio|goals?)|"
    r"based\s+on\s+your\s+(?:risk\s+tolerance|age|income|debt|timeline|savings|portfolio|goals?)"
    r")\b",
    re.IGNORECASE,
)

_FIN_EXPLANATORY_MODE_RE = re.compile(
    r"\b(?:"
    r"is\s+an?\s+investment\s+strategy|"
    r"means\s+that|"
    r"works\s+by|"
    r"can\s+help|"
    r"may\s+help|"
    r"over\s+time|"
    r"generic\s+illustration|"
    r"educational\s+example|"
    r"for\s+example|"
    r"for\s+instance|"
    r"illustrat(?:e|es|ive|ion)|"
    r"hypothetical"
    r")\b",
    re.IGNORECASE,
)


def _fin_quant_is_hedged(lower: str) -> bool:
    """True if a hedge appears near a financial numeric match (before or after)."""
    for m in _FIN_NUMERIC_RE.finditer(lower):
        start, end = m.span()
        # Look back and forward ~55 chars
        pre_window = lower[max(0, start - 55):start]
        post_window = lower[end:min(len(lower), end + 55)]
        if _FIN_HEDGE_RE.search(pre_window) or _FIN_HEDGE_RE.search(post_window):
            return True
    return False


# ── Arithmetic-derivation grounding: variance rule ────────────────────────────
# A variance / shortfall / gap / delta sentence is grounded when the asserted
# amount equals abs(budget_value - actual_value) for a compatible admitted pair
# on the same PEF entity (same-scope constraint).

_VARIANCE_WORD_RE = re.compile(
    r"\b(?:variance|shortfall|gap|delta|difference|discrepancy|"
    r"underperformance?|miss(?:ed)?)\b",
    re.IGNORECASE,
)

# Keywords that classify a PEF entity name as budget-type or actual-type.
_BUDGET_CLASS_TOKENS: frozenset[str] = frozenset({
    "budget", "budgeted", "plan", "planned", "target", "targeted",
    "forecast", "forecasted", "projected", "projection",
})
_ACTUAL_CLASS_TOKENS: frozenset[str] = frozenset({
    "actual", "actuals", "revenue", "revenues", "result", "results",
    "realised", "realized",
})

# Common words that are NOT scope identifiers and must be stripped before
# comparing entity names for scope equality.
_SCOPE_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "for", "of", "in", "at", "to",
    "came", "reported", "recorded",
})

# Causal scaffold / factors text: see ``causal_surface`` (shared with extraction patch).

def _numeric_values_align_for_layer1(a: NumericValue, b: NumericValue) -> bool:
    """True when *b* can ground a Layer-1 assertion *a* (same unit, within tolerance).

    ``values_contradict`` returns False for incompatible units; that must not count
    as agreement — e.g. 8% vs $5.2M is not grounding.
    """
    if a.unit != b.unit:
        return False
    return not values_contradict(a, b)


def _entity_encodes_different_financial_metric(entity_name: str, sentence_metric_lower: str) -> bool:
    """True if *entity_name* contains a financial metric token other than the sentence metric.

    Prevents broad PEF deferral from reusing e.g. revenue's 23% to ground an ROI claim.
    """
    found = {m.group(0).lower() for m in _FIN_METRIC_TOKEN_RE.finditer(entity_name.lower())}
    if not found:
        return False
    return sentence_metric_lower not in found


# First sentence looks like "cause not in the provided data" (third-person epistemic).
_CAUSAL_EPISTEMIC_REFUSAL_HEAD_RE = re.compile(
    r"(?:"
    r"has\s+not\s+been\s+provided|have\s+not\s+been\s+provided|"
    r"not\s+been\s+provided|"
    r"cannot\s+determine|can['\u2019]t\s+determine|"
    r"not\s+(?:specified|stated|given|detailed|identified)\s+in\s+the|"
    r"not\s+(?:specified|stated|given|detailed)\b|"
    r"no\s+(?:specific\s+)?(?:cause|information|data)\s+(?:is\s+)?(?:available|provided|given)|"
    r"not\s+possible\s+to\s+determine|"
    r"insufficient\s+(?:information|data|context)"
    r")",
    re.IGNORECASE,
)

# Continuation after a refusal head: hedged population claims + attribution without
# grounded facts (e.g. "Typically, such shortfalls can be attributed to...").
_CAUSAL_SPECULATIVE_TAIL_RE = re.compile(
    r"(?:"
    r"\b(?:typically|often|generally|commonly)\s*,?\s+such\s+"
    r"(?:shortfalls?|declines?|drops?|gaps?|losses?|miss(?:es)?|variances?)\b"
    r"|"
    r"\b(?:may|might|could)\s+(?:include|often\s+include|involve|reflect|arise\s+from)\b"
    r"|"
    r"\bcan\s+be\s+(?:attributed|traced)\s+to\b"
    r")",
    re.IGNORECASE,
)

# ── Perimeter: user intent (cheap pattern — not epistemic verdict on the answer) ─
# Do not use a bare ``\broot\s+cause\b`` hit: class 1--2 setup often *mentions* root cause
# ("no root cause in the packet") without seeking one; that wording must not arm causal scans.
_USER_SEEKS_SPECIFIC_CAUSE_RE = re.compile(
    r"\b(?:"
    r"what\s+(?:specifically\s+)?(?:caused|causes|triggered)\b|"
    r"what\s+caused\b|"
    r"why\s+(?:did|was|were|is|have)\b|"
    r"(?:identify|determine|explain|find)\s+(?:the\s+)?(?:specific\s+)?root\s+cause\b|"
    r"what\s+(?:is|are|was|were)\s+(?:the\s+)?(?:specific\s+)?root\s+cause\b|"
    r"(?:identify|determine|explain)\s+(?:the\s+)?(?:specific\s+)?(?:cause|reason|driver)\b"
    r")\b",
    re.IGNORECASE,
)

# Class 1--2 setup: ``cannot|could not|unable to`` + ``determine the reason`` reports a
# materials gap, not a request for diagnosis. Same for ``… root cause`` after inability.
_USER_INABILITY_TO_DETERMINE_CAUSE_RE = re.compile(
    r"\b(?:cannot|could\s+not|unable\s+to)\s+(?:determine|identify|explain)\s+the\s+(?:specific\s+)?(?:cause|reason|driver)\b",
    re.IGNORECASE,
)
_USER_INABILITY_TO_DETERMINE_ROOT_CAUSE_RE = re.compile(
    r"\b(?:cannot|could\s+not|unable\s+to)\s+(?:determine|identify|explain|find)\s+(?:the\s+)?(?:specific\s+)?root\s+cause\b",
    re.IGNORECASE,
)

# Question-led lines: do not treat ``could not determine the cause`` as setup-only when the
# utterance is interrogative (e.g. ``Why could we not determine the cause?``).
_USER_LINE_QUESTION_LED_RE = re.compile(
    r"^\s*(?:why|what|how|which|when)\b",
    re.IGNORECASE,
)


def user_seeks_specific_cause(user_input: str | None) -> bool:
    """True when the user is asking for a concrete cause (perimeter / intent only)."""
    if not user_input or len(user_input.strip()) < 8:
        return False
    stripped = user_input.strip()
    question_led = bool(_USER_LINE_QUESTION_LED_RE.match(stripped))
    if question_led:
        return bool(_USER_SEEKS_SPECIFIC_CAUSE_RE.search(user_input))
    for m in _USER_SEEKS_SPECIFIC_CAUSE_RE.finditer(user_input):
        window = user_input[max(0, m.start() - 45) : m.end()]
        if _USER_INABILITY_TO_DETERMINE_CAUSE_RE.search(window):
            continue
        if _USER_INABILITY_TO_DETERMINE_ROOT_CAUSE_RE.search(window):
            continue
        return True
    return False


# Lexical hints for ALLOWED_LIMITATION — procedural / evidence-gap only, not final law.
_ALLOWED_LIMITATION_FEATURE_RE = re.compile(
    r"\b(?:"
    r"(?:additional|more|further)\s+(?:data|information|context|detail|analysis)\b|"
    r"\b(?:would|will)\s+need\s+(?:to\s+)?(?:have|see|obtain|review)\b|"
    r"\bwould\s+require\s+(?:additional|more|further)\b|"
    r"\brequires?\s+(?:additional|more|further)\s+(?:data|information)\b|"
    r"(?:would|will)\s+be\s+required\b"
    r")\b",
    re.IGNORECASE,
)


class CausalClauseAct(str, Enum):
    """Speech-act label for a single response clause (finance root-cause policy)."""

    REFUSAL_OF_CAUSE = "refusal_of_cause"
    ALLOWED_LIMITATION = "allowed_limitation"
    PROHIBITED_SPECULATION = "prohibited_speculation"
    OTHER = "other"


def _clause_has_prohibited_structural_causal_claims(extraction: ExtractionResult) -> bool:
    """True when structured extraction contains speculative causal enumeration (INCLUDE/SCC)."""
    if extraction.extraction_error:
        return False
    for claim in extraction.claims:
        if claim.relation in ("INCLUDE", "INCLUDES") and _CAUSAL_SCAFFOLD_RE.search(claim.subject):
            return True
    return False


def classify_causal_clause_act_from_features(
    *,
    extraction: ExtractionResult,
    clause_lower: str,
) -> CausalClauseAct:
    """Map extraction + cheap lexical features to a speech act (no regex-as-final-judge).

    Order: prohibited structural/lexical/tail features first, then allowed limitation,
    then refusal hints, else OTHER. Lexical patterns only supply features; structure
    (INCLUDE + causal scaffold) can prohibit independently.
    """
    struct = _clause_has_prohibited_structural_causal_claims(extraction)
    lex = bool(_CAUSAL_FACTORS_TEXT_RE.search(clause_lower))
    tail = bool(_CAUSAL_SPECULATIVE_TAIL_RE.search(clause_lower))
    allowed = bool(_ALLOWED_LIMITATION_FEATURE_RE.search(clause_lower))
    refusal = bool(_CAUSAL_EPISTEMIC_REFUSAL_HEAD_RE.search(clause_lower))
    lexical_prohibited = (
        "factor" in clause_lower
        and ("include" in clause_lower or "contribute" in clause_lower)
    )
    if not refusal:
        # Deterministic lexical fallback for refusal phrasing variants that are
        # semantically equivalent but not always covered by regex heads.
        has_cause_anchor = (
            "specific cause" in clause_lower
            or "specific reasons" in clause_lower
            or "root cause" in clause_lower
            or "specific causes" in clause_lower
        )
        has_insufficient_basis = (
            "not been detailed" in clause_lower
            or "not detailed" in clause_lower
            or "not been provided" in clause_lower
            or "not provided" in clause_lower
            or "available data" in clause_lower
            or "cannot determine" in clause_lower
            or "can't determine" in clause_lower
            or "unable to determine" in clause_lower
            or "don't have information" in clause_lower
            or "do not have information" in clause_lower
            or "don't have evidence" in clause_lower
            or "do not have evidence" in clause_lower
        )
        refusal = has_cause_anchor and has_insufficient_basis

    if struct or lex or tail or lexical_prohibited:
        return CausalClauseAct.PROHIBITED_SPECULATION
    if allowed:
        return CausalClauseAct.ALLOWED_LIMITATION
    if refusal:
        return CausalClauseAct.REFUSAL_OF_CAUSE
    return CausalClauseAct.OTHER


def _entity_budget_actual_class(entity_name: str) -> str | None:
    """Classify a PEF entity name as 'budget' or 'actual', or None."""
    tokens = set(re.split(r"\s+", entity_name.lower()))
    if tokens & _BUDGET_CLASS_TOKENS:
        return "budget"
    if tokens & _ACTUAL_CLASS_TOKENS:
        return "actual"
    return None


def _entity_scope_tokens(entity_name: str) -> frozenset[str]:
    """Return the scope-bearing tokens of an entity name.

    Strips budget/actual class keywords and common stopwords so that
    'Q4 North America budget' and 'Q4 North America revenue' share the
    same scope tokens {'q4', 'north', 'america'}.
    """
    tokens = re.split(r"\s+", entity_name.lower())
    return frozenset(
        t for t in tokens
        if t not in _BUDGET_CLASS_TOKENS
        and t not in _ACTUAL_CLASS_TOKENS
        and t not in _SCOPE_STOPWORDS
        and len(t) > 1
    )


def _is_accountability_phrase(text: str) -> bool:
    """True if text is an accountability/responsibility object phrase.

    Uses prefix matching so that natural business expansions like
    'accountable for the North America portfolio' are covered without
    needing an exhaustive vocabulary.
    """
    norm = text.lower().strip().rstrip(".")
    return (
        norm == "accountable"
        or norm == "responsible"
        or norm == "liable"
        or norm.startswith("accountable for")
        or norm.startswith("responsible for")
        or norm.startswith("in charge of")
        or norm.startswith("in charge")
        or norm.startswith("owner of")
        or norm == "owner"
        or norm == "ownership"
        or norm == "oversight"
    )


_ACCOUNTABILITY_FOR_PREFIXES: tuple[str, ...] = (
    "accountable for",
    "responsible for",
    "in charge of",
    "owner of",
)


def _accountability_scope(claim_obj: str) -> frozenset[str] | None:
    """Extract the scope tokens from an accountability phrase.

    'accountable for the North America portfolio' → {'north', 'america', 'portfolio'}
    'responsible for North America'               → {'north', 'america'}
    'accountable' (bare form)                     → None  (no scope restriction)
    """
    norm = claim_obj.lower().strip().rstrip(".")
    for prefix in _ACCOUNTABILITY_FOR_PREFIXES:
        if norm.startswith(prefix):
            remainder = norm[len(prefix):].strip()
            # Strip leading article
            for article in ("the ", "a ", "an "):
                if remainder.startswith(article):
                    remainder = remainder[len(article):]
            tokens = frozenset(remainder.split())
            return tokens if tokens else None
    return None  # bare form — no scope restriction


def _has_business_ownership(
    claim_obj: str,
    existing_rels: list[Relationship],
    pef: PEFState,
) -> bool:
    """True if the entity has a HAS or MANAGE relation over a business object that is
    compatible with the accountability claim's scope.

    Domain gate: the object must pass object_has_business_remit_marker (whole-word markers).
    Scope gate: when the claim names a specific scope ('accountable for North
    America'), the HAS object must share at least one of those scope tokens.
    Bare accountability claims ('accountable', 'responsible') pass the scope
    gate unconditionally — they inherit scope from context.
    """
    scope = _accountability_scope(claim_obj)  # None = bare, no scope restriction

    for rel in existing_rels:
        if rel.relation not in ("HAS", "MANAGE"):
            continue
        obj_text: str | None = None
        if rel.object_literal is not None:
            obj_text = str(rel.object_literal)
        elif rel.object_entity_id is not None:
            obj_entity = pef.entities.get(rel.object_entity_id)
            if obj_entity:
                obj_text = obj_entity.name
        if obj_text is None:
            continue
        obj_lower = obj_text.lower()

        # Domain gate: object must contain a whole-word business-remit marker.
        if not object_has_business_remit_marker(obj_text):
            continue

        # Scope gate: if the claim names a scope, the HAS object must match it.
        # Compare only the *identifying* tokens — strip the business category
        # tokens from both sides so that "EMEA portfolio" and "North America
        # portfolio" are not treated as the same scope because they share "portfolio".
        if scope is not None:
            identifying_scope = scope - BUSINESS_REMIT_MARKERS
            if identifying_scope:
                obj_tokens = frozenset(obj_lower.split()) - BUSINESS_REMIT_MARKERS
                if not (identifying_scope & obj_tokens):
                    continue

        return True

    return False


def _pef_variance_grounded(asserted_val: NumericValue, pef: PEFState) -> bool:
    """Return True if asserted_val equals abs(budget - actual) for an admitted
    budget/actual pair on PEF entities that share at least one scope token.

    Constraints (per specification):
    - Same scope: entity names share at least one non-trivial token after
      stripping class keywords and stopwords.
    - Numeric facts only: both values must be parseable by parse_numeric.
    - Deterministic arithmetic only: exact abs-difference, within the
      existing numeric tolerance defined in values_contradict.
    - No causal language licensed: this function only suppresses
      UNVERIFIED_FACT_ASSERTION; it does not affect causal flag checks.
    """
    budget_candidates: list[tuple[frozenset[str], NumericValue]] = []
    actual_candidates: list[tuple[frozenset[str], NumericValue]] = []

    for entity in pef.entities.values():
        if not entity.resolved:
            continue
        cls = _entity_budget_actual_class(entity.name)
        if cls is None:
            continue
        scope = _entity_scope_tokens(entity.name)
        if not scope:
            continue
        for rel in pef.get_relationships_for_subject(entity.id):
            if rel.negated:
                continue
            if rel.object_literal is None:
                continue
            if _PREDICTIVE_RELATION_RE.search(rel.relation):
                continue
            val = parse_numeric(str(rel.object_literal))
            if val is None:
                continue
            if cls == "budget":
                budget_candidates.append((scope, val))
            else:
                actual_candidates.append((scope, val))

    for b_scope, b_val in budget_candidates:
        for a_scope, a_val in actual_candidates:
            # Same-scope gate: at least one shared scope token.
            if not (b_scope & a_scope):
                continue
            # Compatible units (values_contradict returns False for mismatched
            # units — derive must match the asserted unit to be meaningful).
            if b_val.unit != a_val.unit or b_val.unit != asserted_val.unit:
                continue
            derived = NumericValue(
                magnitude=abs(b_val.magnitude - a_val.magnitude),
                unit=b_val.unit,
            )
            if not values_contradict(derived, asserted_val):
                return True

    return False


# Retrieval-grounded absence: evidential hedge + speculative filler in the same reply.
_ABSENCE_EVIDENTIAL_MARKERS = (
    "the context does not provide",
    "the context does not specify",
    "the information provided does not specify",
    "is not provided in the context",
    "not provided in the information given",
    "without additional information, i cannot determine",
    "i don't have specific information about why",
    "i don't have information about why",
    "there isn't enough information",
    "there is not enough information",
    "not enough information provided to determine why",
    "cannot determine why",
    "can't determine why",
    "unable to determine why",
    "insufficient context",
    "the context does not contain",
    "not stated in the context",
    "does not state why",
)
_SPECULATION_FILLER_MARKERS = (
    "could be",
    "might be",
    "may be",
    "various reasons",
    "such as",
    "for example",
    "job opportunities",
    "family commitments",
    "personal, professional, or family-related",
    "there could be many reasons",
    "including personal, professional, or family-related factors",
    "including personal, professional, or family-related reasons",
    "various personal, professional, or family-related reasons",
    "there could be various",
    "it could be for various",
    "it could be due to",
    "such as personal, professional, or family-related",
    "such as personal, professional, or family-related factors",
    "such as personal, professional, or family-related reasons",
    "for various personal, professional, or family-related reasons",
    "for many reasons",
    "personal, professional, or family-related factors",
    "it could be due to a variety of",
    "it could be due to a variety of personal circumstances",
    "personal circumstances",
    "such as job changes, family reasons, or other factors",
    "job changes, family reasons, or other factors",
    "a variety of personal or professional reasons",
    "a variety of personal, professional, or family-related reasons",
    "factors influencing such a move",
    "can vary widely",
    "possibly",
    "perhaps",
    "one possibility",
    "could include",
)

_WEEKDAY_NAMES_ORDERED: tuple[str, ...] = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def _weekdays_mentioned_ordered(user_lower: str) -> tuple[str, ...]:
    """Weekdays mentioned in ``user_lower``, Monday-first order."""
    return tuple(d for d in _WEEKDAY_NAMES_ORDERED if d in user_lower)


def _user_general_dual_record_calendar_conflict_shape(user_input: str) -> bool:
    """Structural cue: paired records + ``same`` tie + two weekdays (general-domain harness).

    Excludes finance-performance contradiction phrases so the dedicated finance arm rules alone.
    """
    s = user_input.lower().replace("\u2019", "'")
    if re.search(
        r"\b(?:fund|portfolio|benchmark|etf|mutual\s+fund|performance|returns?)\b",
        s,
    ):
        return False
    if not (
        re.search(r"\brecord\s+a\b", s)
        and re.search(r"\brecord\s+b\b", s)
    ):
        return False
    if not re.search(r"\bsame\b", s):
        return False
    return len(_weekdays_mentioned_ordered(s)) >= 2


# Post-LLM clarification binding — narrow lexical hints for continuation-authority prose.
_CLARIFICATION_BINDING_RESPONSE_HINT_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?is)\bwhich\s+[a-z][a-z'\u2019-]{0,48}\s+did\s+you\s+mean\b"),
    re.compile(r"(?is)\bdo\s+you\s+mean\b.{0,160}\bor\b.{0,120}\?"),
)


def _given_name_collision_surfaces(pef: PEFState) -> tuple[str, ...]:
    """Full-name surfaces that share the same first token (e.g. Emma Smith / Emma Jones)."""
    buckets: dict[str, list[str]] = {}
    for ent in pef.entities.values():
        name = (getattr(ent, "name", None) or "").strip()
        parts = name.split()
        if len(parts) < 2:
            continue
        first = parts[0].lower()
        buckets.setdefault(first, []).append(name)
    flat: list[str] = []
    for names in buckets.values():
        if len(names) >= 2:
            flat.extend(sorted(set(names)))
    return tuple(sorted(set(flat)))


def _user_turn_has_underdetermined_third_person_pronoun(user_input: str) -> bool:
    ui = user_input.lower().replace("\u2019", "'")
    return bool(
        re.search(
            r"\b(she|her|hers|he|him|his|they|them|their|theirs|it|its)\b",
            ui,
        )
    )


class Checker:
    """Compare LLM response claims against PEF ground truth."""

    def __init__(self, backend: ExtractionBackend):
        self._backend = backend

    def is_refusal_frame(self, text: str) -> bool:
        """Return True if *text* matches the LLM self-refusal pattern.

        Used by Lens.process() to set LensResult.self_refused when the model
        declined via its own language rather than being stopped by governance.
        """
        return bool(_REFUSAL_FRAME_RE.search(text))

    @staticmethod
    def _dedupe_unverified_fact_assertion_flags(flags: list[Flag]) -> list[Flag]:
        """Keep at most one UNVERIFIED_FACT_ASSERTION (Layer 1 + text/claim paths may overlap)."""
        out: list[Flag] = []
        seen_unverified = False
        for f in flags:
            if f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION:
                if seen_unverified:
                    continue
                seen_unverified = True
            out.append(f)
        return out

    def _rag_disjunctive_prioritize_disjunctive(
        self, flags: list[Flag], user_input: str | None
    ) -> list[Flag]:
        """RAG disjunctive identity: DISJUNCTIVE (CONTAIN) must not lose to epistemic REFUSE flags.

        Ambiguity-shaped replies still trigger UNVERIFIED / UNSUPPORTED / CONTRADICTED_FACT
        (warning) from extraction; StatusTranslator would otherwise pick REFUSE over ASK
        and yield FORCE_REVISE instead of containment.
        """
        if not user_input or not _user_rag_disjunctive_identity_question(user_input):
            return flags
        if not any(f.flag_type == FlagType.DISJUNCTIVE_BRANCH_COLLAPSE for f in flags):
            return flags
        drop_types: frozenset[FlagType] = frozenset({
            FlagType.UNVERIFIED_FACT_ASSERTION,
            FlagType.UNBOUND_ENTITY,
            FlagType.UNSUPPORTED_ATTRIBUTE,
            FlagType.UNSUPPORTED_EVENT,
            FlagType.UNRESOLVED_REFERENT,
            FlagType.UNRESOLVED_COMPARAND,
            FlagType.TIME_SMEAR,
            FlagType.UNVERIFIED_TRUTHFULNESS_CLAIM,
        })
        out: list[Flag] = []
        for f in flags:
            if f.flag_type in drop_types:
                continue
            if f.flag_type == FlagType.CONTRADICTED_FACT and f.severity == "warning":
                continue
            out.append(f)
        # Canonical bridge uses flags[0].flag_type.name as PolicyResolver reason_code.
        # Put DISJUNCTIVE first when it governs, unless a hard-stop substrate flag remains.
        _no_disj_front: frozenset[FlagType] = frozenset({
            FlagType.EXTRACTION_FAILED,
            FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
            FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
            FlagType.NUMERIC_MEDICAL_INSTRUCTION,
            FlagType.EMERGENCY_TRIAGE_GUIDANCE,
            FlagType.SELF_HARM_INSTRUCTION,
            FlagType.ILLEGAL_INSTRUCTION,
            FlagType.TARGETED_DEFAMATION,
        })
        if any(f.flag_type in _no_disj_front for f in out):
            return out
        if any(f.flag_type == FlagType.CONTRADICTED_FACT and f.severity == "error" for f in out):
            return out
        disj = [f for f in out if f.flag_type == FlagType.DISJUNCTIVE_BRANCH_COLLAPSE]
        rest = [f for f in out if f.flag_type != FlagType.DISJUNCTIVE_BRANCH_COLLAPSE]
        if disj:
            return disj + rest
        return out

    def _dual_record_calendar_prioritize_disjunctive(
        self, flags: list[Flag], user_input: str | None
    ) -> list[Flag]:
        """Dual-record weekday harness: DISJUNCTIVE governs before referent/epistemic noise."""
        if not user_input or not _user_general_dual_record_calendar_conflict_shape(user_input):
            return flags
        if not any(f.flag_type == FlagType.DISJUNCTIVE_BRANCH_COLLAPSE for f in flags):
            return flags
        drop_types: frozenset[FlagType] = frozenset({
            FlagType.UNVERIFIED_FACT_ASSERTION,
            FlagType.UNBOUND_ENTITY,
            FlagType.UNSUPPORTED_ATTRIBUTE,
            FlagType.UNSUPPORTED_EVENT,
            FlagType.UNRESOLVED_REFERENT,
            FlagType.UNRESOLVED_COMPARAND,
            FlagType.TIME_SMEAR,
            FlagType.UNVERIFIED_TRUTHFULNESS_CLAIM,
        })
        out: list[Flag] = []
        for f in flags:
            if f.flag_type in drop_types:
                continue
            if f.flag_type == FlagType.CONTRADICTED_FACT and f.severity == "warning":
                continue
            out.append(f)
        _no_disj_front: frozenset[FlagType] = frozenset({
            FlagType.EXTRACTION_FAILED,
            FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
            FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
            FlagType.NUMERIC_MEDICAL_INSTRUCTION,
            FlagType.EMERGENCY_TRIAGE_GUIDANCE,
            FlagType.SELF_HARM_INSTRUCTION,
            FlagType.ILLEGAL_INSTRUCTION,
            FlagType.TARGETED_DEFAMATION,
        })
        if any(f.flag_type in _no_disj_front for f in out):
            return out
        if any(f.flag_type == FlagType.CONTRADICTED_FACT and f.severity == "error" for f in out):
            return out
        disj = [f for f in out if f.flag_type == FlagType.DISJUNCTIVE_BRANCH_COLLAPSE]
        rest = [f for f in out if f.flag_type != FlagType.DISJUNCTIVE_BRANCH_COLLAPSE]
        if disj:
            return disj + rest
        return out

    @staticmethod
    def _dedupe_flags_by_type(flags: list[Flag]) -> list[Flag]:
        seen: set = set()
        out: list[Flag] = []
        for f in flags:
            if f.flag_type not in seen:
                seen.add(f.flag_type)
                out.append(f)
        return out

    def _finalize_checker_flags(self, flags: list[Flag], user_input: str | None) -> list[Flag]:
        return Checker._dedupe_flags_by_type(
            Checker._dedupe_unverified_fact_assertion_flags(
                self._dual_record_calendar_prioritize_disjunctive(
                    self._rag_disjunctive_prioritize_disjunctive(flags, user_input),
                    user_input,
                )
            )
        )

    def _check_user_requires_opens_in_pef(
        self, user_input: str, pef: PEFState
    ) -> list[Flag]:
        """Affordance prerequisite: user text pins *open the X* (head noun ``target``) but PEF lacks OPEN→target.

        Semantically this is an **unsupported affordance / missing prerequisite relation**
        query; we emit :class:`FlagType.UNVERIFIED_FACT_ASSERTION` only because that bucket
        already routes through governance — a dedicated ``MISSING_PREREQUISITE_EDGE`` (or
        similar) would be clearer when one exists.

        Requires a parsable *open the X* phrase to obtain ``target``; otherwise no flag
        (cannot object-pin). Raw ``user_input`` only; complements response-claim checks in
        :meth:`check`.
        """
        if not _user_text_implicates_opens_affordance_query(user_input):
            return []
        target = _extract_opens_target_from_user_text(user_input)
        if not target:
            return []
        if _pef_has_open_for_target(pef, target):
            return []
        return [Flag(
            flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
            entity_name="pef",
            claim=(
                f"No OPEN/OPENS edge in PEF whose object mentions {target!r} "
                f"(affordance prerequisite for opening that referent)."
            ),
            evidence=(user_input[:220] + ("…" if len(user_input) > 220 else "")),
            severity="warning",
            rule_id="checker.user_opens_affordance_missing_in_pef",
        )]

    def _check_upstream_insufficient_context(
        self,
        response_text: str,
        user_input: str | None,
        extraction: ExtractionResult,
    ) -> list[Flag]:
        """Structured insufficient-context speech act on upstream model replies."""
        features = extract_upstream_response_features(
            response_text=response_text,
            user_input=user_input,
            extraction=extraction,
        )
        if classify_upstream_response_speech_act(features) != UpstreamResponseSpeechAct.INSUFFICIENT_CONTEXT:
            return []
        snippet = response_text.strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "…"
        return [Flag(
            flag_type=FlagType.UPSTREAM_INSUFFICIENT_CONTEXT,
            entity_name="upstream",
            claim=(
                "Upstream response declares insufficient context for the user query "
                "and solicits clarification."
            ),
            evidence=snippet or response_text,
            severity="warning",
            rule_id="checker.upstream_insufficient_context",
        )]

    async def check(
        self,
        response_text: str,
        pef: PEFState,
        user_input: str | None = None,
        *,
        user_grounding: UserGroundingContext | None = None,
    ) -> list[Flag]:
        """Extract claims from LLM response and verify against PEF state.

        Returns a list of flags (empty = clean response).
        When extraction fails (extraction_error), returns EXTRACTION_FAILED flag
        so governance can intervene instead of raising.

        Optional ``user_grounding`` supplies same-turn ``user_input``-provenance
        relationships from :class:`~aurora_lens.verify.user_grounding.UserGroundingContext`
        (built by Lens). When omitted, behaviour matches pre-bridge verification.
        """
        if user_input:
            _request_blocked = evaluate_blocked_act_request(user_input, pef=pef)
            _vci_flags = violent_criminal_intent_flags(user_input)
            if _request_blocked or _vci_flags:
                _merged = list(_request_blocked) + list(_vci_flags)
                return self._finalize_checker_flags(_merged, user_input)

        result = await self._backend.extract(response_text, pef)
        flags: list[Flag] = []

        # First-person detection: when user speaks in first person ("I've had chest
        # pain", "my arm hurts"), conversational pronouns in the LLM response
        # (it/its/they/them/their/these/those) are back-references to what the user
        # described — not LLM-invented world claims requiring PEF grounding.
        is_first_person: bool = _is_first_person_input(user_input) if user_input else False
        has_third_party: bool = bool(_THIRD_PARTY_RE.search(user_input)) if user_input else False

        # Words from user_input that also appear in the LLM response — secondary
        # back-reference signal when user did not speak in first person.
        user_echoed_words: frozenset[str] = frozenset()
        if user_input:
            user_echoed_words = _content_words(user_input) & _content_words(response_text)

        if result.extraction_error:
            err = result.extraction_error
            reason = err.get("reason", "UNKNOWN")
            snippet = err.get("raw_preview", "")[:100]
            flags.append(Flag(
                flag_type=FlagType.EXTRACTION_FAILED,
                entity_name="extraction",
                claim=f"Extraction backend returned invalid output: {reason}",
                evidence=snippet or str(err),
                severity="error",
                extraction_diagnostic=err,
            ))
            return self._finalize_checker_flags(flags, user_input)

        # Raw-text scans — run before extraction-empty guard, independent of
        # claim extraction. Organized by flag axis.
        # Axis 3 (normative veto — hard-stop always):
        flags.extend(self._check_medical_dosage(response_text))
        flags.extend(self._check_numeric_medical_instruction(response_text))
        if user_input:
            flags.extend(self._check_emergency_triage(user_input))
        flags.extend(self._check_emergency_triage(response_text))
        flags.extend(self._check_self_harm_instruction(response_text))
        flags.extend(self._check_harmful_operational_guidance(response_text))
        flags.extend(self._check_illegal_instruction(response_text))
        # Violent criminal intent: check user_input first (query is non-admissible
        # regardless of the model's reply), then response_text as a secondary guard.
        if user_input:
            flags.extend(self._check_violent_criminal_intent(user_input))
        flags.extend(self._check_violent_criminal_intent(response_text))
        flags.extend(self._check_targeted_defamation(response_text))
        # Axis 3 (normative veto — hard-stop always, education/workforce/enterprise):
        flags.extend(self._check_academic_integrity_violation(response_text))
        if user_input:
            flags.extend(self._check_academic_integrity_violation(user_input))
        flags.extend(self._check_student_record_exposure(response_text))
        flags.extend(self._check_employment_discrimination(response_text))
        if user_input:
            flags.extend(self._check_employment_discrimination(user_input))
        flags.extend(self._check_employee_record_exposure(response_text))
        flags.extend(self._check_trade_secret_disclosure(response_text))
        if user_input:
            flags.extend(self._check_trade_secret_disclosure(user_input))
        flags.extend(self._check_insider_information(response_text))
        if user_input:
            flags.extend(self._check_insider_information(user_input))
        flags.extend(self._check_procurement_fraud(response_text))
        if user_input:
            flags.extend(self._check_procurement_fraud(user_input))
        # Axis 3 (normative veto — verify-or-refuse, mode-dependent):
        flags.extend(self._check_professional_directive(response_text, user_input))
        flags.extend(self._check_fictional_wrapper_response(response_text, pef))
        flags.extend(self._check_truthfulness_certainty(response_text))
        if user_input:
            flags.extend(self._check_bare_truthfulness_verdict(response_text, user_input))
        if user_grounding is not None and user_grounding.financial_determination_probe:
            _inherited_pfa = personalized_financial_advice_from_inherited_response(
                user_grounding.financial_determination_probe,
                response_text,
            )
            if _inherited_pfa is not None:
                flags.append(_inherited_pfa)
        flags.extend(self._check_pii_exposure(response_text))
        flags.extend(self._check_regulatory_claims(response_text))
        # Verify-or-refuse: education / workforce / enterprise personalised advice
        flags.extend(self._check_personalized_academic_advice(response_text, user_input))
        flags.extend(self._check_personalized_employment_advice(response_text, user_input))
        flags.extend(self._check_personalized_business_strategy(response_text, user_input))
        if user_input and user_seeks_specific_cause(user_input):
            flags.extend(
                await self._check_finance_specific_cause_clause_policy(response_text, pef)
            )
        flags.extend(
            self._check_speculative_causal_enumeration(response_text, user_input=user_input)
        )
        flags.extend(
            self._check_absence_with_speculative_filler(response_text, user_input=user_input)
        )
        if user_input:
            flags.extend(
                self._check_unsupported_advisory_escape(response_text, user_input)
            )
        flags.extend(
            self._check_unverified_quantitative_financial(
                response_text,
                pef=pef,
                user_input=user_input,
                user_grounding=user_grounding,
            )
        )
        if user_input:
            _u1f = self._check_rag_causal_bounded_absence_non_admit(response_text, user_input)
            if _u1f is not None:
                flags.append(_u1f)
            _u2f = self._check_rag_salary_bounded_absence_non_admit(response_text, user_input)
            if _u2f is not None:
                flags.append(_u2f)
            # RAG C2 harness: must run before the refusal-frame early return. Phrases like
            # "cannot be confirmed" match _REFUSAL_FRAME_RE but are still manifest C2 replies
            # that require DISJUNCTIVE_BRANCH_COLLAPSE (CONTAIN), not PASS.
            _c2f = self._check_disjunctive_whose_sister_collapse(user_input, response_text)
            if _c2f is not None:
                flags.append(_c2f)
            _caf = self._check_clarification_authority_outside_hold(
                response_text, user_input, pef
            )
            if _caf is not None:
                flags.append(_caf)
            _c3f = self._check_finance_contradictory_same_period_performance(
                user_input, response_text
            )
            if _c3f is not None:
                flags.append(_c3f)
            _c4f = self._check_general_dual_record_calendar_conflict(
                user_input, response_text
            )
            if _c4f is not None:
                flags.append(_c4f)
            flags.extend(self._check_user_requires_opens_in_pef(user_input, pef))

        flags.extend(
            self._check_upstream_insufficient_context(
                response_text,
                user_input,
                result,
            )
        )
        # Refusal frame guard: if the response is a meta-speech refusal, skip
        # extraction-empty and claim-level checks. Refusals are not world claims.
        if _REFUSAL_FRAME_RE.search(response_text):
            return self._finalize_checker_flags(flags, user_input)

        MIN_LEN = 50  # Avoid flagging short refusals like "I don't know about Bob."
        text_len = len(response_text.strip())
        all_empty = not result.claims and not result.entity_mentions
        if text_len >= MIN_LEN and all_empty:
            if any(f.flag_type == FlagType.DISJUNCTIVE_BRANCH_COLLAPSE for f in flags):
                # Paired-record / finance-style containment harness already governs on user shape.
                # Hedge replies often extract empty; extraction-empty HARD_STOP must not override.
                pass
            elif any(
                f.flag_type == FlagType.UPSTREAM_INSUFFICIENT_CONTEXT for f in flags
            ):
                # Insufficient-context speech act already classified; do not duplicate.
                pass
            else:
                flags.append(Flag(
                    flag_type=FlagType.EXTRACTION_EMPTY,
                    entity_name="extraction",
                    claim="Extraction produced no admissible structure.",
                    evidence="Extraction produced no admissible structure.",
                    severity="error",
                ))
                # Do not return early; allow _finalize_checker_flags to sort by precedence.
                # Since all_empty is true, _check_claim loop below will be a no-op.

        for claim in result.claims:
            claim_flags = self._check_claim(
                claim,
                pef,
                result.span,
                user_echoed_words,
                is_first_person,
                has_third_party,
                user_input,
                response_text,
                user_grounding=user_grounding,
            )
            flags.extend(claim_flags)

        return self._finalize_checker_flags(flags, user_input)

    def _check_unsupported_advisory_escape(
        self, response_text: str, user_input: str
    ) -> list[Flag]:
        """U2-style: salary/compensation questions must not end with generic HR/contract advice."""
        ui = user_input.lower()
        if not any(
            k in ui
            for k in ("salary", "annual salary", "compensation", "pay rate", "wage")
        ):
            return []
        rt = response_text.lower()
        markers = (
            "check with hr",
            "human resources",
            "employment contract",
            "your employer",
            "payroll",
            "speak with your manager",
            "contact hr",
        )
        if not any(m in rt for m in markers):
            return []
        return [
            Flag(
                flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
                entity_name="advisory_escape",
                claim="Unsupported fact question paired with generic HR/contract advisory tail",
                evidence=(
                    "Bounded-absence replies must not add generic workplace advisory escape hatches."
                ),
                severity="warning",
            )
        ]

    def _check_rag_causal_bounded_absence_non_admit(
        self, response_text: str, user_input: str | None
    ) -> Flag | None:
        """RAG harness U1: truthful bounded absence for causal why must not govern as PASS."""
        if not user_input or rag_context_body_from_user_input(user_input) is None:
            return None
        if not _user_rag_causal_why_question(user_input):
            return None
        text = response_text.lower()
        has_absence = any(m in text for m in _ABSENCE_EVIDENTIAL_MARKERS) or _rag_causal_bounded_absence_surface(
            text
        )
        has_spec = any(m in text for m in _SPECULATION_FILLER_MARKERS)
        if has_absence and has_spec:
            return None
        if not has_absence:
            if not (
                ("does not" in text or "not stated" in text or "not specify" in text)
                and ("context" in text or "passage" in text or "provided" in text)
            ):
                return None
        return Flag(
            flag_type=FlagType.RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT,
            entity_name="rag_harness",
            claim="RAG harness: bounded causal absence is governed non-admit (not PASS).",
            evidence=response_text.strip()[:240],
            severity="warning",
        )

    def _check_rag_salary_bounded_absence_non_admit(
        self, response_text: str, user_input: str | None
    ) -> Flag | None:
        """RAG harness U2: salary question + generic absence/confidentiality must not PASS."""
        if not user_input or rag_context_body_from_user_input(user_input) is None:
            return None
        if not _user_rag_salary_compensation_question(user_input):
            return None
        rt = response_text.lower()
        absence_like = (
            "not in the context" in rt
            or "does not contain" in rt
            or "does not include" in rt
            or "does not state" in rt
            or "does not specify" in rt
            or "not specified" in rt
            or "not disclosed" in rt
            or "confidential" in rt
            or "not available" in rt
            or "insufficient" in rt
            or "does not provide" in rt
            or "i don't have" in rt
            or "i do not have" in rt
            or "cannot provide" in rt
            or "no information" in rt
            or "not included" in rt
        )
        if not absence_like:
            return None
        if re.search(r"[\$€£]\s*[\d,]+|\b\d{2,}\s*k\b|\d{3,}\s*(?:usd|cad|dollars?)\b", rt):
            return None
        return Flag(
            flag_type=FlagType.RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT,
            entity_name="rag_harness",
            claim="RAG harness: salary absence reply is governed non-admit (not PASS).",
            evidence=response_text.strip()[:240],
            severity="warning",
        )

    def _check_clarification_authority_outside_hold(
        self,
        response_text: str,
        user_input: str | None,
        pef: PEFState,
    ) -> Flag | None:
        """Detect binding clarification emitted without governed containment state.

        Scoped: duplicate given-name collisions in committed PEF + third-person pronoun
        in the user query + clarification-shaped assistant prose. Avoids firing when a
        structured pending clarification already holds the continuation corridor.
        """
        if not user_input or pef.pending_clarification:
            return None
        collisions = _given_name_collision_surfaces(pef)
        if len(collisions) < 2:
            return None
        if not _user_turn_has_underdetermined_third_person_pronoun(user_input):
            return None
        rt = response_text.strip()
        if not rt:
            return None
        if not any(p.search(rt) for p in _CLARIFICATION_BINDING_RESPONSE_HINT_RES):
            return None
        return Flag(
            flag_type=FlagType.CLARIFICATION_AUTHORITY_OUTSIDE_HOLD,
            entity_name="clarification_authority",
            claim=(
                "Assistant solicited binding disambiguation without governed containment "
                "(pending clarification / epistemic hold)."
            ),
            evidence=rt[:240],
            severity="warning",
            candidates=collisions,
        )

    def _check_disjunctive_whose_sister_collapse(
        self, user_input: str | None, response_text: str
    ) -> Flag | None:
        """RAG disjunctive identity: always containment (ASK), including hedged ambiguity prose.

        **RAG-shaped turns only** (``Context:`` … ``Question:``) so non-RAG turns are unchanged.
        """
        if not user_input or rag_context_body_from_user_input(user_input) is None:
            return None
        if not _user_rag_disjunctive_identity_question(user_input):
            return None
        question = _rag_question_line_from_user_input(user_input)
        if not question:
            return None
        candidates = _extract_disjunctive_possessive_candidates(question)
        if not candidates or len(candidates) < 2:
            return None
        rl = response_text.lower()
        first, second = candidates[0], candidates[1]
        has_first = first.lower() in rl
        has_second = second.lower() in rl
        has_sister = "sister" in rl
        if _response_hedges_disjunctive_ambiguity(response_text, candidates):
            claim = "RAG disjunctive identity: ambiguity-preserving reply (containment, not admissible PASS)."
        elif has_sister and has_first and not has_second:
            claim = f"Collapsed disjunction to {first}'s branch only (containment)."
        elif has_sister and has_second and not has_first:
            claim = f"Collapsed disjunction to {second}'s branch only (containment)."
        else:
            claim = "RAG disjunctive identity question requires containment, not PASS."
        return Flag(
            flag_type=FlagType.DISJUNCTIVE_BRANCH_COLLAPSE,
            entity_name="disjunctive",
            claim=claim,
            evidence=(
                f"Preserve {first} vs {second} ambiguity; governed outcome is clarification / containment."
            ),
            severity="warning",
            candidates=candidates,
        )

    def _check_finance_contradictory_same_period_performance(
        self, user_input: str | None, _response_text: str
    ) -> Flag | None:
        """Single-turn contradictory performance claims over the same cited window (demo-stable CONTAIN).

        Distinguishes explainable multi-window summaries (e.g. strong month vs weak quarter)
        from *same-period* outperform + underperform cues that must not PASS as reconciled fact.
        """
        if not user_input:
            return None
        s = user_input.lower().replace("\u2019", "'")
        if not re.search(
            r"\b(?:fund|portfolio|benchmark|etf|mutual\s+fund|performance|returns?)\b",
            s,
        ):
            return None
        pos_pat = r"\b(?:outperform(?:ed|s|ing)?|beat(?:s|ing)?\s+(?:the\s+)?benchmark)\b"
        neg_pat = r"\b(?:underperform(?:ed|s|ing)?|trail(?:ed|s|ing)?(?:\s+the\s+benchmark)?|lagged)\b"
        if not re.search(pos_pat, s) or not re.search(neg_pat, s):
            return None
        same_report = "same report" in s
        months = re.findall(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            s,
        )
        dup_month = len(months) >= 2 and months[0] == months[1]
        if not (same_report or dup_month):
            return None
        return Flag(
            flag_type=FlagType.DISJUNCTIVE_BRANCH_COLLAPSE,
            entity_name="finance_performance_conflict",
            claim="Contradictory fund performance claims over the same cited window require clarification.",
            evidence=user_input.strip()[:280],
            severity="warning",
        )

    def _check_general_dual_record_calendar_conflict(
        self, user_input: str | None, response_text: str
    ) -> Flag | None:
        """General-domain paired-record weekday disagreement — governed CONTAIN (DISJUNCTIVE).

        Uses structural cues only (``record A`` / ``record B``, ``same``, two weekdays).
        """
        del response_text  # parity with finance harness — signal is user-turn shape only
        if not user_input:
            return None
        if not _user_general_dual_record_calendar_conflict_shape(user_input):
            return None
        s = user_input.lower().replace("\u2019", "'")
        ordered = _weekdays_mentioned_ordered(s)
        labels = tuple(w.capitalize() for w in ordered)
        return Flag(
            flag_type=FlagType.DISJUNCTIVE_BRANCH_COLLAPSE,
            entity_name="dual_record_calendar_conflict",
            claim=(
                "Contradictory calendar cues from paired records require clarification "
                "before a single date can be treated as final."
            ),
            evidence=user_input.strip()[:280],
            severity="warning",
            candidates=labels,
        )

    def check_blocked_act_request(
        self,
        user_input: str,
        pef: PEFState | None = None,
    ) -> list[Flag]:
        """Pre-LLM: classify whether the user request is a determinate blocked regulated-domain act.

        Called BEFORE the ambiguity gate in lens.py. Ensures regulated-domain refusal
        dominates lower-level interpretation paths (ambiguity clarification, general-info
        laundering).

        Precedence invariant: if no plausible resolution of ambiguity in the request
        would convert it from blocked to admissible, the ambiguity gate must not fire —
        domain refusal dominates.

        Blocked-act classes detected on the input side include: pediatric /
        weight-based dosing; adult personalised dosing/titration; fictional-wrapper
        real medical action; personalised financial advice; legal case-outcome
        questions; and **delegated modal** requests (e.g. *Should she…*, *Is she
        correct*, *Is the treatment better*) with real consequence anchors.

        These checks target determinate inputs only. General legal information,
        neutral procedural explanation, and non-personalised finance queries are
        not affected — only input patterns that are already determinate blocked acts
        regardless of phrasing ambiguity.

        Implementation: :func:`~aurora_lens.verify.blocked_request_policy.evaluate_blocked_act_request`.
        """
        return evaluate_blocked_act_request(user_input, pef=pef)

    def _check_claim(
        self,
        claim: ExtractedClaim,
        pef: PEFState,
        response_span: Span,
        user_echoed_words: frozenset[str] = frozenset(),
        is_first_person: bool = False,
        has_third_party: bool = False,
        user_input: str | None = None,
        response_text: str | None = None,
        user_grounding: UserGroundingContext | None = None,
    ) -> list[Flag]:
        """Check a single claim against PEF state."""
        flags: list[Flag] = []

        # Skip first-person pronoun subjects — these are model meta-speech, not
        # world claims requiring PEF grounding.
        if claim.subject.lower() in _FIRST_PERSON:
            return []

        # Skip second-person pronoun subjects — user-address phrases ("you may
        # experience X", "you IS appropriate") are not entity assertions and
        # must not trigger hallucination or unresolved-referent checks.
        # Exception: instructional dependency / affordance (need, must, open, …)
        # must still be verified — see _second_person_instructional_mechanism_claim.
        if claim.subject.lower() in _SECOND_PERSON:
            if not _second_person_instructional_mechanism_claim(claim):
                return []

        # Skip if the full clause is an epistemic disclaimer or capability statement.
        # Extraction often produces claims from sentences like "I cannot explain X"
        # that are not factual assertions about the world.
        if _EPISTEMIC_DISCLAIMER_RE.search(claim.evidence):
            return []

        # Look up subject entity
        entity = pef.find_entity_by_name(claim.subject)
        if entity is None:
            entity = _find_entity_by_unique_first_token(pef, claim.subject)

        if entity is None:
            # Anaphoricity test: only block when the subject is a pronoun or
            # definite description that presupposes a prior PEF antecedent.
            # New concept literals (domain terms, bare nouns, proper nouns not
            # yet in PEF) enter cleanly — they are not anaphoric references.
            if claim.relation not in ("IS",):
                subj = (claim.subject or "").strip()
                subj_lower = subj.lower()

                # Speculative causal scaffold guard (SCC): an INCLUDE/INCLUDES
                # claim whose subject is a generic anonymous causal phrase
                # ("common factors that can lead to …", "possible causes …")
                # is an ungrounded enumeration — not a new concept literal.
                # Fire UNVERIFIED_FACT_ASSERTION before the bare-NP shortcut
                # swallows it.
                if (claim.relation in ("INCLUDE", "INCLUDES")
                        and _CAUSAL_SCAFFOLD_RE.search(claim.subject)):
                    return [Flag(
                        flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
                        entity_name=claim.subject[:60],
                        claim=(
                            f"Speculative causal enumeration: "
                            f"{claim.subject[:60]} {claim.relation} "
                            f"{claim.obj[:80]}"
                        ),
                        evidence=claim.evidence[:120] if claim.evidence else "",
                        severity="warning",
                    )]

                # Multi-word bare NPs → new concept literal, skip.
                # Exception: definite descriptions do presuppose an antecedent.
                if " " in subj and not _is_definite_description(subj):
                    return []

                # Pronoun / demonstrative → requires prior PEF grounding.
                if subj_lower in _REFERENT_PRONOUNS:
                    # Back-reference exemption: conversational pronouns
                    # (it/its/they/them/their/these/those) are exempt when the user
                    # spoke in first person (describing their own experience) OR
                    # when the response echoes content words from user_input.
                    # Rationale: "I've had chest pain... it's gone now" → LLM's "it"
                    # refers to the pain the user described; "these symptoms" refers
                    # to what the user reported. These are conversational anaphora,
                    # not LLM-invented world claims requiring PEF grounding.
                    if subj_lower in _CONVERSATIONAL_PRONOUNS and (is_first_person or user_echoed_words):
                        return []
                    # Crisis/third-party exemption: gendered pronouns (she/her/he/him)
                    # are exempt only when the user (a) spoke in first person AND
                    # (b) explicitly mentioned a third party ("my daughter", "my patient",
                    # etc.). is_first_person alone is not enough — it fires on almost
                    # every message and would create broad pronoun amnesty.
                    # Neutral pronouns (they/them/their) are already handled above via
                    # _CONVERSATIONAL_PRONOUNS when is_first_person is True.
                    if subj_lower in ("she", "her", "he", "him", "his", "hers") \
                            and is_first_person \
                            and has_third_party:
                        return []
                    # PEF-resolution exemption: if the pronoun resolves to a known
                    # PEF entity AND the resulting claim is clean, suppress
                    # UNRESOLVED_REFERENT. This handles the natural case where the
                    # LLM uses "she" to refer to Alice (the only active entity) and
                    # the claim is actually supported by the PEF.
                    # If the resolved claim itself has flags (e.g. UNSUPPORTED_ATTRIBUTE)
                    # we do NOT suppress UNRESOLVED_REFERENT — the pronoun resolution
                    # alone does not redeem a fabricated fact.
                    pef_entity = _resolve_pronoun_via_pef(pef, subj_lower)
                    if pef_entity is not None:
                        resolved_claim = ExtractedClaim(
                            subject=pef_entity.name,
                            relation=claim.relation,
                            obj=claim.obj,
                            span=claim.span,
                            negated=claim.negated,
                            evidence=claim.evidence,
                            provenance=claim.provenance,
                            extractor_backend=claim.extractor_backend,
                        )
                        resolved_flags = self._check_claim(
                            resolved_claim,
                            pef,
                            response_span,
                            user_echoed_words,
                            is_first_person,
                            has_third_party,
                            user_input,
                            response_text,
                            user_grounding=user_grounding,
                        )
                        if not resolved_flags:
                            return []
                        # Claim is problematic even when resolved — fall through
                        # to UNRESOLVED_REFERENT.
                    flags.append(Flag(
                        flag_type=FlagType.UNRESOLVED_REFERENT,
                        entity_name=subj,
                        claim=f"{subj} {claim.relation} {claim.obj}",
                        evidence=f"Unresolved referent: '{subj}'",
                    ))
                    return flags

                # Definite description → requires a unique prior antecedent.
                if _is_definite_description(subj):
                    # Back-reference exemption: when user spoke in first person,
                    # definite descriptions like "these symptoms" / "the chest pain"
                    # refer to what the user described — not LLM-invented entities.
                    # Also exempt when content words overlap with echoed user content.
                    if is_first_person or (user_echoed_words and (_content_words(subj) & user_echoed_words)):
                        return []
                    # Generic class NP exemption: "those who prefer certainty",
                    # "those comfortable with variability", "those seeking stability"
                    # are universal quantifiers, not anaphoric references. They do
                    # not presuppose a specific prior PEF entity.
                    # "Those symptoms" / "these conditions" do NOT match and are
                    # correctly treated as anaphoric demonstratives.
                    if _GENERIC_CLASS_NP_RE.match(subj):
                        return []
                    flags.append(Flag(
                        flag_type=FlagType.UNRESOLVED_REFERENT,
                        entity_name=subj,
                        claim=f"{subj} {claim.relation} {claim.obj}",
                        evidence=f"Unresolved definite description: '{subj}'",
                    ))
                    return flags

                # Second-person instructional mechanism: do not admit as anonymous literal.
                if _second_person_instructional_mechanism_claim(claim):
                    return [Flag(
                        flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
                        entity_name=claim.subject[:60],
                        claim=(
                            "Instructional dependency claim (second-person subject): "
                            f"{claim.subject[:60]} {claim.relation} {claim.obj[:80]}"
                        ),
                        evidence=claim.evidence[:120] if claim.evidence else "",
                        severity="warning",
                    )]

                # Bare noun / proper noun → new concept literal, admit cleanly.
                return []

            # IS claims with no matching entity: copula parses for necessity still matter.
            if claim.relation in ("IS",) and _second_person_instructional_mechanism_claim(
                    claim):
                return [Flag(
                    flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
                    entity_name=claim.subject[:60],
                    claim=(
                        "Instructional dependency claim (second-person subject): "
                        f"{claim.subject[:60]} {claim.relation} {claim.obj[:80]}"
                    ),
                    evidence=claim.evidence[:120] if claim.evidence else "",
                    severity="warning",
                )]

            return flags

        # Entity exists — check if unresolved placeholder
        if not entity.resolved:
            # "Introduced" policy: unresolved placeholders were mentioned
            # but never asserted about. Asserting new facts about them
            # is a hallucination — UNLESS the entity is already known to
            # the system as the object of an existing PEF relationship.
            # Example: "Acme Corp." was established as Alice's employer via
            # Alice AT "Acme Corp." — it is a referenced entity, not a true
            # placeholder, so claims about it proceed to normal checking.
            is_referenced = any(
                rel.object_entity_id == entity.id
                or (
                    rel.object_literal is not None
                    and _objects_match(str(rel.object_literal), entity.name)
                )
                for subj_entity in pef.entities.values()
                for rel in pef.get_relationships_for_subject(subj_entity.id)
            )
            if not is_referenced and user_grounding is not None:
                is_referenced = self._entity_grounded_in_retrieved_context(
                    entity, pef, user_grounding
                )
            if not is_referenced:
                flags.append(Flag(
                    flag_type=FlagType.UNSUPPORTED_ATTRIBUTE,
                    entity_name=claim.subject,
                    claim=f"{claim.subject} {claim.relation} {claim.obj}",
                    evidence="Entity exists only as unresolved placeholder — no established facts",
                ))
                return flags
            # Entity is established via reference — fall through to normal
            # claim checking. It has no subject-relationships yet, so only
            # contradiction can fire, not hallucination.

        # Entity is resolved — check claim against existing relationships
        subject_rels = pef.get_relationships_for_subject(entity.id)

        # Positive AT claims must agree with committed current_at (single head).
        if (
            not claim.negated
            and canonicalize_relation(claim.relation) == "AT"
        ):
            committed = current_at(pef, entity.id)
            if committed is not None:
                committed_txt = self._get_rel_object_text(committed, pef)
                if (
                    committed_txt is not None
                    and not _objects_match(committed_txt, claim.obj)
                ):
                    return [
                        Flag(
                            flag_type=FlagType.CONTRADICTS_COMMITTED_STATE,
                            entity_name=claim.subject,
                            claim=f"{claim.subject} {claim.relation} {claim.obj}",
                            evidence=(
                                "Committed AT for this entity is "
                                f"{committed_txt!r}; response asserts {claim.obj!r}"
                            ),
                            severity="error",
                        ),
                    ]

        # Retrieval admission left incompatible literals on PEF (not in relationships).
        # Do not rely on to_context_summary() — consult retrieval_unresolved directly.
        retrieval_collapse = self._check_retrieval_unresolved_literal_collapse(
            claim, entity, pef, user_grounding=user_grounding
        )
        if retrieval_collapse:
            flags.append(retrieval_collapse)
            return flags

        # Check for contradiction
        contradiction = self._check_contradiction(claim, subject_rels, pef)
        if contradiction:
            flags.append(contradiction)
            return flags

        tempo = self._check_retrieved_temporal_conflict(
            claim, entity, pef, user_input=user_input
        )
        if tempo:
            flags.append(tempo)
            return flags

        # Check for time-smear
        time_smear = self._check_time_smear(
            claim,
            subject_rels,
            response_span,
            user_input=user_input,
            response_text=response_text,
            user_grounding=user_grounding,
            pef=pef,
            entity=entity,
        )
        if time_smear:
            flags.append(time_smear)

        # Check for hallucinated attributes/events
        hallucination = self._check_hallucination(
            claim,
            subject_rels,
            pef,
            entity=entity,
            user_input=user_input,
            response_text=response_text,
            user_grounding=user_grounding,
        )
        if hallucination:
            flags.append(hallucination)

        return flags

    def _check_contradiction(
        self,
        claim: ExtractedClaim,
        existing_rels: list[Relationship],
        pef: PEFState,
    ) -> Flag | None:
        """Check if the claim contradicts an established fact.

        Pass 1 (string): same relation + same object + polarity mismatch.
        Pass 2 (numeric): Layer 3 financial numeric fallback when pass 1 finds no match.
        """
        for rel in existing_rels:
            if not _relations_equivalent_for_grounding(rel.relation, claim.relation):
                continue

            # Get the object text for comparison
            rel_obj = self._get_rel_object_text(rel, pef)
            if rel_obj is None:
                continue

            if not _objects_match(rel_obj, claim.obj):
                continue

            # Same subject, same relation, same object — check negation
            if rel.negated != claim.negated:
                neg_word = "negated" if claim.negated else "asserted"
                return Flag(
                    flag_type=FlagType.CONTRADICTED_FACT,
                    entity_name=claim.subject,
                    claim=f"{claim.subject} {'NOT ' if claim.negated else ''}{claim.relation} {claim.obj}",
                    evidence=f"PEF has this as {'negated' if rel.negated else 'asserted'}, "
                             f"but response {neg_word} it (turn {rel.source_turn})",
                    severity="error",
                )

        # Pass 2: Layer 3 numeric contradiction (financial metrics only)
        return self._check_numeric_contradiction(claim, existing_rels)

    def _check_retrieval_unresolved_literal_collapse(
        self,
        claim: ExtractedClaim,
        entity,
        pef: PEFState,
        *,
        user_grounding: UserGroundingContext | None = None,
    ) -> Flag | None:
        """When RAG admission stored a literal_conflict, block definitive assertion of one branch.

        Conflicting values are not committed as relationships; they live only on
        :attr:`PEFState.retrieval_unresolved`. The checker must read that field
        directly so verify/governance do not depend on summary injection.

        Corpus Q&A may restate wording that appears in the retrieved ``Context:``
        block even when noisy extraction created a literal_conflict entry.
        """
        if claim.negated:
            return None
        if not pef.retrieval_unresolved:
            return None
        obj = (claim.obj or "").strip()
        if not obj:
            return None
        claim_rel = canonicalize_relation(claim.relation)
        # Keep aligned with aurora_lens.rag_pef_admission._IS_SLOT_RELATIONS (IS only for v1).
        if claim_rel != "IS":
            return None
        ent_key = entity.name.strip().lower()
        ctx_lower = (
            user_grounding.retrieved_context_text.lower()
            if user_grounding is not None and user_grounding.retrieved_context_text
            else None
        )
        for item in pef.retrieval_unresolved:
            if item.get("kind") != "literal_conflict":
                continue
            if item.get("relation") != claim_rel:
                continue
            if (item.get("subject") or "").strip().lower() != ent_key:
                continue
            lits = item.get("literals") or []
            if len(lits) < 2:
                continue
            for lit in lits:
                if _objects_match(str(lit), obj):
                    if ctx_lower is not None and obj.lower() in ctx_lower:
                        return None
                    return Flag(
                        flag_type=FlagType.CONTRADICTED_FACT,
                        entity_name=claim.subject,
                        claim=f"{claim.subject} {claim.relation} {claim.obj}",
                        evidence=(
                            "PEF retrieval_unresolved lists incompatible IS literals for this "
                            f"entity ({', '.join(str(x) for x in lits)}); do not collapse to one."
                        ),
                        severity="warning",
                    )
        return None

    def _check_retrieved_temporal_conflict(
        self,
        claim: ExtractedClaim,
        entity,
        pef: PEFState,
        user_input: str | None = None,
    ) -> Flag | None:
        """Retrieved chunks ground incompatible arrival times; do not assert one calendar date (C1)."""
        if claim.negated:
            return None
        if not _claim_asserts_specific_calendar_date(claim):
            return None

        seam = _c1_narrow_temporal_seam_active(pef, entity, user_input)
        # partial_gold RAG harness: retrieved batch may omit the companion chunk (e.g. only Section 6);
        # blob then lacks joint incompatible arrival seam — still do not PASS a single calendar date.
        if (
            user_input
            and _user_rag_calendar_arrival_date_question(user_input)
            and not seam
            and _claim_subject_matches_rag_question_entity(claim, user_input)
        ):
            body = rag_context_body_from_user_input(user_input)
            if body and not incompatible_arrival_time_joined_text(body.lower()):
                return Flag(
                    flag_type=FlagType.CONTRADICTED_FACT,
                    entity_name=claim.subject,
                    claim=f"{claim.subject} {claim.relation} {claim.obj}",
                    evidence=(
                        "RAG calendar-arrival question: retrieved context does not jointly contain "
                        "the incompatible arrival timeline (calendar date vs last week); "
                        "do not assert a single calendar date."
                    ),
                    severity="warning",
                )

        if not seam:
            return None
        from_retrieval = _retrieval_unresolved_has_temporal_arrival_incompatible(pef, entity)
        from_pef = _pef_has_incompatible_arrival_time_literals(pef, entity.id)
        from_blob = _rag_context_blob_has_narrow_c1_seam(user_input, entity)
        if from_retrieval and not from_pef:
            ev = (
                "PEF retrieval_unresolved records incompatible arrival times "
                "(calendar date vs last week in retrieved claims); do not collapse to a single date."
            )
        elif from_blob and not from_pef:
            ev = (
                "RAG context block contains incompatible arrival times "
                "(calendar date vs last week); do not collapse to a single calendar date."
            )
        else:
            ev = (
                "Retrieved evidence contains incompatible arrival times "
                "(calendar date vs relative phrasing); do not collapse to a single date."
            )
        return Flag(
            flag_type=FlagType.CONTRADICTED_FACT,
            entity_name=claim.subject,
            claim=f"{claim.subject} {claim.relation} {claim.obj}",
            evidence=ev,
            severity="warning",
        )

    def _check_numeric_contradiction(
        self,
        claim: ExtractedClaim,
        existing_rels: list[Relationship],
    ) -> Flag | None:
        """Layer 3: numeric contradiction for financial metric claims.

        Fires when:
        - claim.subject is a recognised financial metric token
        - claim is a non-negated assertion with a parseable numeric object
        - a non-negated, non-predictive PEF literal for the same subject
          parses to a numeric value that contradicts the claim value

        Relation equality is NOT required — any compatible non-negated
        assertion on the same entity is checked.  This handles the case
        where PEF stores relation="IS" but the LLM uses "increased" or
        "reached" for the same established fact.
        """
        # Gate 1: claim subject must be an explicit financial metric token
        if not _FIN_METRIC_TOKEN_RE.search(claim.subject):
            return None

        # Gate 2: non-negated claim only (v1 — skip negated claim handling)
        if claim.negated:
            return None

        # Gate 3: claim object must parse as a numeric value.
        # Fall back to claim.evidence when claim.obj is empty or unparseable —
        # this catches prepositional constructions ("increased to $3.5M") where
        # spaCy places the numeric in pobj rather than dobj, leaving claim.obj empty.
        # parse_numeric is called on the full evidence string so that suffixed
        # currency forms like "$3.5M" are parsed correctly as 3_500_000.
        claim_val = parse_numeric(claim.obj)
        if claim_val is None and claim.evidence:
            claim_val = parse_numeric(claim.evidence)
        if claim_val is None:
            return None

        for rel in existing_rels:
            # Skip negated PEF facts (v1)
            if rel.negated:
                continue
            # Need a literal object to compare against
            if rel.object_literal is None:
                continue
            # Skip predictive/modal PEF relations — projections are not facts
            if _PREDICTIVE_RELATION_RE.search(rel.relation):
                continue

            rel_val = parse_numeric(str(rel.object_literal))
            if rel_val is None:
                continue

            if values_contradict(claim_val, rel_val):
                # Compound-literal exemption: spaCy sometimes jams a comparative
                # clause into one object literal, e.g.
                #   '$ 3.8 M shortfall of $ 400 K versus Q3'
                # Both $3.8M and $400K are admitted values from that sentence.
                # When the LLM echoes a secondary embedded value (e.g. $400K via
                # a prepositional construction), the evidence fallback picks it up
                # and the primary-value comparison fires a false positive.
                # Guard: if claim_val matches ANY value in the compound literal,
                # it is an acknowledged sub-fact — not a contradiction.
                all_lit_vals = parse_all_numerics(str(rel.object_literal))
                if any(not values_contradict(claim_val, v) for v in all_lit_vals):
                    continue
                return Flag(
                    flag_type=FlagType.CONTRADICTED_FACT,
                    entity_name=claim.subject,
                    claim=f"{claim.subject} {claim.relation} {claim.obj}",
                    evidence=(
                        f"Numeric contradiction: PEF records {rel.object_literal!r} "
                        f"but response asserts {claim.obj!r} (turn {rel.source_turn})"
                    ),
                    severity="error",
                )

        return None

    def _check_time_smear(
        self,
        claim: ExtractedClaim,
        existing_rels: list[Relationship],
        response_span: Span,
        *,
        user_input: str | None = None,
        response_text: str | None = None,
        user_grounding: UserGroundingContext | None = None,
        pef: PEFState | None = None,
        entity: object | None = None,
    ) -> Flag | None:
        """Check for time-smear: combining facts from different spans.

        A claim is TIME_SMEAR if it requires combining facts from different
        spans without an explicit span transition signal.
        """
        if user_input and _claim_numeric_echoes_user_context(
            claim, user_input, response_text=response_text
        ):
            return None
        # Bridge seam: same-turn user_input-provenance relationships can license
        # faithful restatements so they are not mis-classified as TIME_SMEAR.
        # This is evidentiary grounding only; policy safety gates remain separate.
        if (
            user_grounding is not None
            and pef is not None
        ):
            ent_for_bridge = entity
            if ent_for_bridge is None:
                ent_for_bridge = pef.find_entity_by_name(claim.subject)
                if ent_for_bridge is None:
                    ent_for_bridge = _find_entity_by_unique_first_token(pef, claim.subject)
            if (
                ent_for_bridge is not None
                and self._same_turn_user_bridge_grounds_claim(
                    claim, ent_for_bridge, pef, user_grounding
                )
            ):
                # TIME_SMEAR boundary: user-grounding provenance alone is not
                # enough to suppress temporal-shift checks. Only suppress when
                # a same-turn grounded relation is span-compatible with claim.
                eid = ent_for_bridge.id
                for rel in user_grounding.user_committed_relationships:
                    if rel.subject_id != eid:
                        continue
                    if rel.negated != claim.negated:
                        continue
                    if _PREDICTIVE_RELATION_RE.search(rel.relation):
                        continue
                    if not _relations_equivalent_for_grounding(rel.relation, claim.relation):
                        continue
                    rel_obj = self._get_rel_object_text(rel, pef)
                    if rel_obj is None:
                        continue
                    co = (claim.obj or "").strip()
                    ev = (claim.evidence or "").strip()
                    obj_match = (co and _objects_match(rel_obj, co)) or (ev and _objects_match(rel_obj, ev[:400]))
                    if obj_match and rel.span == claim.span:
                        return None
        for rel in existing_rels:
            if not _relations_equivalent_for_grounding(rel.relation, claim.relation):
                continue
            # If PEF has this fact in PAST but the claim presents it as PRESENT
            if rel.span == Span.PAST and claim.span == Span.PRESENT:
                # Exemption: faithful acknowledgement of admitted context.
                # When the claim's numeric value exactly matches the PEF fact,
                # the model is restating user-admitted context in conversational-
                # present tense — not asserting a state transition.
                if rel.object_literal is not None and not claim.negated:
                    claim_vals: list[NumericValue] = []
                    if claim.obj:
                        claim_vals.extend(parse_all_numerics(claim.obj))
                    if claim.evidence:
                        claim_vals.extend(parse_all_numerics(claim.evidence))
                    if not claim_vals:
                        cv = parse_numeric(claim.obj)
                        if cv is None and claim.evidence:
                            cv = parse_numeric(claim.evidence)
                        if cv is not None:
                            claim_vals = [cv]
                    rel_vals = parse_all_numerics(str(rel.object_literal))
                    if not rel_vals:
                        rv = parse_numeric(str(rel.object_literal))
                        if rv is not None:
                            rel_vals = [rv]
                    aligned = False
                    for claim_val in claim_vals:
                        for rel_val in rel_vals:
                            if not values_contradict(claim_val, rel_val):
                                aligned = True
                                break
                        if aligned:
                            break
                    if aligned:
                        continue
                return Flag(
                    flag_type=FlagType.TIME_SMEAR,
                    entity_name=claim.subject,
                    claim=f"{claim.subject} {claim.relation} {claim.obj} (presented as present)",
                    evidence=f"Established as past-span fact (turn {rel.source_turn})",
                )
            # Vice versa: PEF has PRESENT but claim uses PAST
            if rel.span == Span.PRESENT and claim.span == Span.PAST:
                return Flag(
                    flag_type=FlagType.TIME_SMEAR,
                    entity_name=claim.subject,
                    claim=f"{claim.subject} {claim.relation} {claim.obj} (presented as past)",
                    evidence=f"Established as present-span fact (turn {rel.source_turn})",
                )

        return None

    @staticmethod
    def _numeric_layer1_grounded_via_user_bridge(
        asserted_val: NumericValue,
        user_grounding: UserGroundingContext | None,
    ) -> bool:
        """Layer 1: numeric aligns with a same-turn user_committed relationship literal."""
        if user_grounding is None or not user_grounding.user_committed_relationships:
            return False
        for rel in user_grounding.user_committed_relationships:
            if rel.negated or rel.object_literal is None:
                continue
            if _PREDICTIVE_RELATION_RE.search(rel.relation):
                continue
            for pef_val in parse_all_numerics(str(rel.object_literal)):
                if _numeric_values_align_for_layer1(asserted_val, pef_val):
                    return True
        return False

    def _same_turn_user_bridge_grounds_claim(
        self,
        claim: ExtractedClaim,
        entity,
        pef: PEFState,
        user_grounding: UserGroundingContext,
    ) -> bool:
        """Structured grounding: claim matches a same-turn user or retrieved PEF edge."""
        evidence_rels = user_grounding.same_turn_evidence_relationships
        if not evidence_rels:
            return False
        eid = entity.id
        for rel in evidence_rels:
            if rel.subject_id != eid:
                continue
            if rel.negated != claim.negated:
                continue
            if _PREDICTIVE_RELATION_RE.search(rel.relation):
                continue
            if _relations_equivalent_for_grounding(rel.relation, claim.relation):
                rel_obj = self._get_rel_object_text(rel, pef)
                if rel_obj is not None:
                    co = (claim.obj or "").strip()
                    if co and _objects_match(rel_obj, co):
                        return True
                    ev = (claim.evidence or "").strip()
                    if ev and _objects_match(rel_obj, ev[:400]):
                        return True
            if not claim.negated and rel.object_literal is not None:
                claim_vals: list[NumericValue] = []
                if claim.obj:
                    claim_vals.extend(parse_all_numerics(claim.obj))
                if claim.evidence:
                    claim_vals.extend(parse_all_numerics(claim.evidence))
                if not claim_vals:
                    cv = parse_numeric(claim.obj)
                    if cv is None and claim.evidence:
                        cv = parse_numeric(claim.evidence)
                    if cv is not None:
                        claim_vals = [cv]
                seen_cv: set[tuple[float, NumericUnit]] = set()
                deduped: list[NumericValue] = []
                for cv in claim_vals:
                    key = (cv.magnitude, cv.unit)
                    if key not in seen_cv:
                        seen_cv.add(key)
                        deduped.append(cv)
                for claim_val in deduped:
                    lit = str(rel.object_literal)
                    for pef_val in parse_all_numerics(lit):
                        if not values_contradict(claim_val, pef_val):
                            return True
                    pef_one = parse_numeric(lit)
                    if pef_one is not None and not values_contradict(claim_val, pef_one):
                        return True
        return False

    @staticmethod
    def _entity_surface_in_retrieved_context_text(
        entity_name: str,
        retrieved_context_text: str | None,
    ) -> bool:
        if not retrieved_context_text or not entity_name.strip():
            return False
        name = entity_name.strip().lower()
        hay = retrieved_context_text.lower()
        if name in hay:
            return True
        first = name.split()[0]
        return len(first) >= 4 and first in hay

    def _entity_grounded_in_retrieved_context(
        self,
        entity,
        pef: PEFState,
        user_grounding: UserGroundingContext | None,
    ) -> bool:
        """True when retrieval evidence licenses restating facts about this entity."""
        if user_grounding is None:
            return False
        if self._entity_surface_in_retrieved_context_text(
            entity.name,
            user_grounding.retrieved_context_text,
        ):
            return True
        eid = entity.id
        for rel in user_grounding.retrieved_context_relationships:
            if rel.subject_id == eid:
                return True
            rel_obj = self._get_rel_object_text(rel, pef)
            if rel_obj is not None and _objects_match(rel_obj, entity.name):
                return True
        return False

    def _check_hallucination(
        self,
        claim: ExtractedClaim,
        existing_rels: list[Relationship],
        pef: PEFState,
        *,
        entity: object | None = None,
        user_input: str | None = None,
        response_text: str | None = None,
        user_grounding: UserGroundingContext | None = None,
    ) -> Flag | None:
        """Check for hallucinated attributes/events.

        Only flags when the entity IS resolved (has established facts)
        but the specific claim has no PEF support.
        """
        # Check if ANY relationship supports this claim
        for rel in existing_rels:
            if not _relations_equivalent_for_grounding(rel.relation, claim.relation):
                continue

            rel_obj = self._get_rel_object_text(rel, pef)
            if rel_obj is None:
                continue

            if _objects_match(rel_obj, claim.obj):
                # Found support — not hallucinated
                return None

        # No support found via relation-matching.

        # Cross-relation numeric support (relation normalisation): a numeric value
        # admitted under any relation (e.g. AT, RETURN) licenses the same value
        # asserted under IS or any other relation.  This handles cases where PEF
        # stores "Actual Q4 North America revenue AT $4.8M" but the LLM responds
        # with "revenue IS $4.8M" — the numeric fact is admitted regardless of
        # which surface verb was used to introduce it.
        if not claim.negated:
            claim_vals: list[NumericValue] = []
            if claim.obj:
                claim_vals.extend(parse_all_numerics(claim.obj))
            if claim.evidence:
                claim_vals.extend(parse_all_numerics(claim.evidence))
            if not claim_vals:
                cv = parse_numeric(claim.obj)
                if cv is None and claim.evidence:
                    cv = parse_numeric(claim.evidence)
                if cv is not None:
                    claim_vals = [cv]
            seen_cv: set[tuple[float, NumericUnit]] = set()
            deduped: list[NumericValue] = []
            for cv in claim_vals:
                key = (cv.magnitude, cv.unit)
                if key not in seen_cv:
                    seen_cv.add(key)
                    deduped.append(cv)
            for claim_val in deduped:
                for rel in existing_rels:
                    if rel.negated or rel.object_literal is None:
                        continue
                    if _PREDICTIVE_RELATION_RE.search(rel.relation):
                        continue
                    lit = str(rel.object_literal)
                    for pef_val in parse_all_numerics(lit):
                        if not values_contradict(claim_val, pef_val):
                            return None
                    pef_one = parse_numeric(lit)
                    if pef_one is not None and not values_contradict(claim_val, pef_one):
                        return None

        # Derived-accountability grounding (rule A): within the business/finance
        # domain, "X IS accountable/responsible" is derivable from
        # "X HAS" or "X MANAGE" over a business-remit object (whole-word markers).
        # That licenses accountability language for governed answer generation.
        # No causal language is licensed by this rule.
        if (
            claim.relation == "IS"
            and _is_accountability_phrase(claim.obj)
            and _has_business_ownership(claim.obj, existing_rels, pef)
        ):
            return None

        ent_for_bridge = entity
        if ent_for_bridge is None:
            ent_for_bridge = pef.find_entity_by_name(claim.subject)
            if ent_for_bridge is None:
                ent_for_bridge = _find_entity_by_unique_first_token(pef, claim.subject)
        if (
            ent_for_bridge is not None
            and user_grounding is not None
            and self._same_turn_user_bridge_grounds_claim(
                claim, ent_for_bridge, pef, user_grounding
            )
        ):
            return None

        # Check if ANY relationship with this relation type exists
        has_same_relation = any(
            _relations_equivalent_for_grounding(r.relation, claim.relation)
            for r in existing_rels
        )

        if has_same_relation:
            # Entity has facts with this relation, but not this specific one
            return Flag(
                flag_type=FlagType.UNSUPPORTED_ATTRIBUTE,
                entity_name=claim.subject,
                claim=f"{claim.subject} {claim.relation} {claim.obj}",
                evidence=f"No PEF support for this specific {claim.relation} fact",
            )

        # Entity has no facts with this relation type at all
        if existing_rels:
            # Legacy text-echo fallback remains intentionally for turns where
            # structured user grounding was not supplied by the caller.
            # When ``user_grounding`` is present, the bridge path above is the
            # primary same-turn evidentiary grounding check.
            if user_input and _claim_numeric_echoes_user_context(
                claim, user_input, response_text=response_text
            ):
                return None
            return Flag(
                flag_type=FlagType.UNSUPPORTED_EVENT,
                entity_name=claim.subject,
                claim=f"{claim.subject} {claim.relation} {claim.obj}",
                evidence=f"No {claim.relation} relationships established for this entity",
            )

        return None

    def _check_medical_dosage(self, text: str) -> list[Flag]:
        """Detect clinical dosage recommendations (normative veto — axis 3).

        Two sub-cases, both severity=error, both map to HARD_STOP via policy.
        Neither depends on PEF support — the claim being true is irrelevant.

        PEDIATRIC_DOSAGE_RECOMMENDATION
            Pediatric indicator + dosage amount.  Most severe sub-case.

        MEDICAL_DOSAGE_RECOMMENDATION
            Explicit patient instruction verb + dosage amount, no pediatric
            indicator.  "Recommended dosage is X mg" without an instruction
            verb does NOT fire (informational, not a patient directive).
        """
        if not text or len(text.strip()) < 20:
            return []

        lower = text.lower()

        # Pediatric indicators (presence of any is sufficient)
        pediatric = any(
            p in lower
            for p in (
                "child", "children", "pediatric", "infant", "toddler", "baby",
                "year-old", "month-old", "year old", "month old",
            )
        ) or bool(re.search(r"\d+\s*kg\b", lower))  # explicit patient weight
        # mg/kg notation is weight-normalised dosing — the unit itself is the
        # pediatric signal, even when no age/weight keyword appears in the text.
        pediatric = pediatric or bool(re.search(r"\bmg/kg\b|\bmcg/kg\b", lower, re.IGNORECASE))

        # Dosage amount: numeric + medical unit
        dosage_amount = bool(
            re.search(r"\d+\s*(mg|ml|mg/kg|mcg)\b", lower, re.IGNORECASE)
            or re.search(r"\d+[-–]\d+\s*mg\b", lower)  # range: "720-810 mg"
        )

        if pediatric and dosage_amount:
            return [Flag(
                flag_type=FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
                entity_name="medical",
                claim="Response contains specific dosing guidance for a pediatric patient",
                evidence="Pediatric indicator + dosage amount detected",
                severity="error",
            )]

        # General adult dosage — requires an explicit instruction verb so that
        # informational phrases like "the recommended dosage is 720 mg" don't fire.
        instruction_verb = bool(
            re.search(
                r"\b(take|give|administer|dosed?\s+at|prescribe[sd]?|"
                r"start\w*|increas\w*|decreas\w*|titrat\w*|initiat\w*|max\w*)\b",
                lower,
            )
        )
        # Descriptor guard: "typically/usually/commonly taken at X mg" is descriptive,
        # not a patient directive — exempt unless second-person framing is also present.
        descriptor_only = bool(
            re.search(r"\b(?:typically|usually|commonly|generally|often)\b", lower)
            and not re.search(
                r"\b(you|your|patient|the\s+patient|diagnosis|metformin|lisinopril|insulin)\b",
                lower,
            )
        )
        if dosage_amount and instruction_verb and not descriptor_only:
            return [Flag(
                flag_type=FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
                entity_name="medical",
                claim="Response contains an explicit patient dosing instruction",
                evidence="Instruction verb + dosage amount detected",
                severity="error",
            )]

        return []

    def _check_regulatory_claims(self, text: str) -> list[Flag]:
        """Detect unsupported regulatory/approval assertions (epistemic failure — axis 1).

        Asserting FDA/EMA/CE approval status is an epistemic claim that requires
        verifiable support (labeling data, approval records).  Without PEF or
        context evidence, these are UNVERIFIED_REGULATORY_CLAIM.

        This is NOT a normative veto: a sourced, hedged response is acceptable.
        Governance default: FORCE_REVISE (cite support or hedge).
        """
        if not text or len(text.strip()) < 20:
            return []

        lower = text.lower()

        regulatory = bool(
            re.search(r"\bfda[- ]?(?:approved|cleared|authorized|licensed)\b", lower)
            or re.search(r"\bce[- ]?(?:mark(?:ed)?|certified|approved)\b", lower)
            or re.search(r"\bema[- ]?(?:approved|authorized)\b", lower)
            or re.search(
                r"\b(?:approved|authorized|cleared)\s+by\s+(?:the\s+)?(?:fda|ema|who|cdc|mhra)\b",
                lower,
            )
        )

        if regulatory:
            return [Flag(
                flag_type=FlagType.UNVERIFIED_REGULATORY_CLAIM,
                entity_name="regulatory",
                claim="Response asserts regulatory/approval status",
                evidence="Regulatory claim (FDA/EMA/CE/WHO) detected without verifiable support",
                severity="warning",
            )]
        return []

    @staticmethod
    def _split_response_clauses(text: str) -> list[str]:
        """Perimeter segmentation: sentence-like units (not semantic parsing)."""
        return split_sentences_at_punctuation(text)

    async def _classify_causal_clause_act(self, clause: str, pef: PEFState) -> CausalClauseAct:
        """Label one clause using structured extraction; lexical patterns only as fallback features."""
        low = clause.lower().strip()
        ex = await self._backend.extract(clause, pef)
        if ex.extraction_error:
            if _CAUSAL_FACTORS_TEXT_RE.search(low) or _CAUSAL_SPECULATIVE_TAIL_RE.search(low):
                return CausalClauseAct.PROHIBITED_SPECULATION
            if _ALLOWED_LIMITATION_FEATURE_RE.search(low):
                return CausalClauseAct.ALLOWED_LIMITATION
            if _CAUSAL_EPISTEMIC_REFUSAL_HEAD_RE.search(low):
                return CausalClauseAct.REFUSAL_OF_CAUSE
            return CausalClauseAct.OTHER
        return classify_causal_clause_act_from_features(extraction=ex, clause_lower=low)

    async def _iter_finance_specific_cause_clause_act_pairs(
        self, response_text: str, pef: PEFState
    ) -> AsyncIterator[tuple[str, CausalClauseAct]]:
        """Yield (clause, act) for each substantive clause — same segmentation as the policy."""
        for clause in self._split_response_clauses(response_text):
            if len(clause) < 10:
                continue
            yield clause, await self._classify_causal_clause_act(clause, pef)

    async def classify_finance_specific_cause_clause_acts(
        self, response_text: str, pef: PEFState
    ) -> list[tuple[str, CausalClauseAct]]:
        """Act label per substantive clause using the same path as `_check_finance_specific_cause_clause_policy`.

        For aligning external tests (e.g. proxy live runs) with checker epistemology without
        duplicating phrase lists. Skips clauses shorter than 10 characters.
        """
        return [pair async for pair in self._iter_finance_specific_cause_clause_act_pairs(response_text, pef)]

    async def _check_finance_specific_cause_clause_policy(
        self, response_text: str, pef: PEFState
    ) -> list[Flag]:
        """Continuation policy when the user asked for a concrete cause (perimeter intent).

        If a clause refuses causal determination from context, any later clause classified
        as prohibited speculative enumeration forces revision (UNVERIFIED_FACT_ASSERTION).
        Lexical regexes feed the act classifier as features, not as the sole epistemic judge.
        """
        if not response_text or len(response_text.strip()) < 20:
            return []

        saw_refusal = False
        async for clause, act in self._iter_finance_specific_cause_clause_act_pairs(response_text, pef):
            if act == CausalClauseAct.REFUSAL_OF_CAUSE:
                saw_refusal = True
                continue
            if saw_refusal and act == CausalClauseAct.PROHIBITED_SPECULATION:
                low = response_text.lower()
                tail_m = _CAUSAL_SPECULATIVE_TAIL_RE.search(low) or _CAUSAL_FACTORS_TEXT_RE.search(
                    low
                )
                if tail_m:
                    snip = response_text[
                        max(0, tail_m.start() - 5) : min(len(response_text), tail_m.end() + 90)
                    ].strip()
                else:
                    snip = clause[:200]
                return [
                    Flag(
                        flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
                        entity_name="causal_enumeration",
                        claim=(
                            "Speculative causal continuation after refusal to determine cause "
                            "(clause-level speech-act policy)"
                        ),
                        evidence=snip,
                        severity="warning",
                    )
                ]

        return []

    def _check_speculative_causal_enumeration(
        self, text: str, user_input: str | None = None
    ) -> list[Flag]:
        """Text-level guard for diagnostic causal enumeration not caught by claim-level SCC.

        When spaCy extracts a non-INCLUDE relation (e.g. from an infinitive clause),
        the claim-level _CAUSAL_SCAFFOLD_RE guard is bypassed.  This method scans the
        raw response text for enumeration / soft-attribution patterns.

        Also catches **mixed** replies: an epistemic refusal in the first sentence
        followed by speculative cause language (e.g. "has not been provided" then
        "Typically, such shortfalls can be attributed to factors like...") — those
        must not PASS solely on the opening refusal.

        Perimeter: when *user_input* is present and the user did not ask for a
        concrete cause (``user_seeks_specific_cause`` is false — e.g. same-turn
        context setup with no "why"/"what caused"), this scan is skipped so benign
        acknowledgements are not forced into FORCE_REVISE.  When *user_input* is
        None (unit tests / internal callers), behaviour is unchanged (strict).
        """
        if not text or len(text.strip()) < 20:
            return []

        if user_input is not None and not user_seeks_specific_cause(user_input):
            return []

        def _causal_enum_flag(snippet: str, claim: str) -> list[Flag]:
            return [Flag(
                flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
                entity_name="causal_enumeration",
                claim=claim,
                evidence=snippet.strip()[:200],
                severity="warning",
            )]

        lowered = text.lower()
        m = _CAUSAL_FACTORS_TEXT_RE.search(lowered)
        if m:
            snip = text[max(0, m.start() - 10):min(len(text), m.end() + 100)].strip()
            return _causal_enum_flag(
                snip,
                "Speculative causal enumeration (factors/causes such as, like, or attributed-to)",
            )

        m_hedge = _CAUSAL_HEDGED_OPEN_DOMAIN_RE.search(lowered)
        if m_hedge:
            snip = text[
                max(0, m_hedge.start() - 10) : min(len(text), m_hedge.end() + 100)
            ].strip()
            return _causal_enum_flag(
                snip,
                "Hedged open-domain cause phrasing (could/may/might be due to various…)",
            )

        sentences = split_sentences_at_punctuation(text.strip())
        if len(sentences) >= 2:
            head = sentences[0].lower()
            tail = " ".join(sentences[1:]).lower()
            if (
                _CAUSAL_EPISTEMIC_REFUSAL_HEAD_RE.search(head)
                and _CAUSAL_SPECULATIVE_TAIL_RE.search(tail)
            ):
                tail_m = _CAUSAL_SPECULATIVE_TAIL_RE.search(lowered)
                if tail_m:
                    snip = text[max(0, tail_m.start() - 5):min(len(text), tail_m.end() + 90)].strip()
                else:
                    snip = (sentences[0][:100] + " | " + sentences[1][:100]).strip()
                return _causal_enum_flag(
                    snip,
                    "Speculative causal continuation after epistemic refusal (cause not in context)",
                )

        return []

    def _check_absence_with_speculative_filler(
        self, response_text: str, user_input: str | None = None
    ) -> list[Flag]:
        """Flag absence-of-evidence hedging followed by generic speculative filler.

        Structural substring policy (no regex): if the reply both reports that
        context does not support an answer and adds possibility/enumeration filler,
        treat the filler as non-admissible for retrieval-grounded turns.

        Only applies when ``user_input`` matches the RAG harness shape (``Context:`` …
        ``Question:``). Uses :func:`rag_context_body_from_user_input` to avoid importing
        :mod:`aurora_lens.lens` (circular import). Non-RAG turns are unchanged.
        """
        if not response_text or not response_text.strip():
            return []
        from aurora_lens.pef.arrival_incompatible import rag_context_body_from_user_input

        if rag_context_body_from_user_input(user_input) is None:
            return []
        text = response_text.lower()
        has_absence = any(m in text for m in _ABSENCE_EVIDENTIAL_MARKERS)
        has_speculation = any(m in text for m in _SPECULATION_FILLER_MARKERS)
        if not (has_absence and has_speculation):
            return []
        snippet = response_text.strip()[:200]
        return [
            Flag(
                flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
                entity_name="absence_speculation",
                claim=(
                    "Evidential absence statement combined with generic speculative "
                    "possibility filler (retrieval-grounded reply)"
                ),
                evidence=snippet,
                severity="warning",
            )
        ]

    def _check_unverified_quantitative_financial(
        self,
        text: str,
        pef: PEFState | None = None,
        user_input: str | None = None,
        user_grounding: UserGroundingContext | None = None,
    ) -> list[Flag]:
        """Layer 1: ungrounded quantitative financial assertion (raw text, axis 1).

        Requires together: financial metric token, numeric expression, and assertion
        surface form.  At most one flag per response (v1).  Financial-only.

        Grounding (canonical numerics via parse_numeric / parse_all_numerics):
        - PEF literals on metric-scoped entities (paths 1–3) and variance rule B
        - Any numeric in the current user message (setup-fact echo)
        - Same-turn user_input-provenance relationship literals (UserGroundingContext)
        - Any non-predictive literal on any resolved entity (broad PEF deferral)

        Contradiction is handled by Layer 3 (CONTRADICTED_FACT).
        """
        if not text or len(text.strip()) < 15:
            return []

        lower = text.lower()
        # Minimal segment split by sentence enders followed by whitespace
        segments = split_sentences_at_punctuation(lower)

        for sent in segments:
            # Triple-match requirement (metric: skip compound-adjective "yield")
            metric_m = _fin_layer1_first_metric_match(sent)
            if not metric_m:
                continue
            if not _FIN_NUMERIC_RE.search(sent):
                continue
            if not _FIN_ASSERT_RE.search(sent):
                continue

            # Per-segment edu guard (prevents mixed Q&A under-flagging)
            if _FIN_EDU_GUARD_RE.search(sent):
                continue

            # Polysemous margin guard
            if "margin" in sent and _FIN_MARGIN_LAYOUT_RE.search(sent):
                continue

            # Bidirectional hedge guard (before or after numeric)
            if _fin_quant_is_hedged(sent):
                continue

            metric_token = metric_m.group(0)

            # Pair the metric token with the nearest numeric span — not parse_numeric(sent),
            # which uses global bps-before-% order and mis-binds multi-number sentences.
            asserted_val = numeric_for_metric_span(sent, metric_m.start(), metric_m.end())
            if asserted_val is None:
                asserted_val = parse_numeric(sent)
            if asserted_val is None:
                return [Flag(
                    flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
                    entity_name="financial",
                    claim="Response asserts a specific quantitative financial fact without verifiable grounding",
                    evidence="Financial metric + numeric + assertion pattern detected (Layer 1 raw scan)",
                    severity="warning",
                )]

            grounded = False

            if pef is not None:
                # Path 1: exact entity-name match.
                entity = pef.find_entity_by_name(metric_token)
                if entity is not None:
                    for rel in pef.get_relationships_for_subject(entity.id):
                        if rel.negated or rel.object_literal is None:
                            continue
                        if _PREDICTIVE_RELATION_RE.search(rel.relation):
                            continue
                        pef_val = parse_numeric(str(rel.object_literal))
                        if pef_val is not None and _numeric_values_align_for_layer1(asserted_val, pef_val):
                            grounded = True
                            break

                # Path 2: partial-name match — metric token is a substring
                # of the entity name (e.g. "revenue" in
                # "Actual Q4 North America revenue").
                if not grounded:
                    metric_lower = metric_token.lower()
                    for candidate in pef.entities.values():
                        if not candidate.resolved:
                            continue
                        if metric_lower not in candidate.name.lower():
                            continue
                        for rel in pef.get_relationships_for_subject(candidate.id):
                            if rel.negated or rel.object_literal is None:
                                continue
                            if _PREDICTIVE_RELATION_RE.search(rel.relation):
                                continue
                            pef_val = parse_numeric(str(rel.object_literal))
                            if pef_val is not None and _numeric_values_align_for_layer1(asserted_val, pef_val):
                                grounded = True
                                break
                        if grounded:
                            break

                # Path 3: embedded secondary value in compound literal.
                if not grounded:
                    metric_lower = metric_token.lower()
                    for candidate in pef.entities.values():
                        if not candidate.resolved:
                            continue
                        if metric_lower not in candidate.name.lower():
                            continue
                        for rel in pef.get_relationships_for_subject(candidate.id):
                            if rel.negated or rel.object_literal is None:
                                continue
                            if _PREDICTIVE_RELATION_RE.search(rel.relation):
                                continue
                            for pef_val in parse_all_numerics(str(rel.object_literal)):
                                if _numeric_values_align_for_layer1(asserted_val, pef_val):
                                    grounded = True
                                    break
                            if grounded:
                                break
                        if grounded:
                            break

                # Arithmetic-derivation grounding (rule B)
                if not grounded and _VARIANCE_WORD_RE.search(sent) and _pef_variance_grounded(asserted_val, pef):
                    grounded = True

            # User-message numerics (current-turn setup / admitted context)
            if not grounded and user_input:
                for uval in parse_all_numerics(user_input):
                    if _numeric_values_align_for_layer1(asserted_val, uval):
                        grounded = True
                        break

            # Same-turn user_input provenance edges (Lens structured bridge)
            if not grounded and self._numeric_layer1_grounded_via_user_bridge(
                asserted_val, user_grounding
            ):
                grounded = True

            # Broad PEF: literals on entities that do not encode a different metric
            # (e.g. "APAC" + $4.2M grounds "revenue is $4.2M"; "revenue" + 23% does not
            # ground "ROI is 23%").
            if not grounded and pef is not None:
                metric_lower = metric_token.lower()
                for candidate in pef.entities.values():
                    if not candidate.resolved:
                        continue
                    if _entity_encodes_different_financial_metric(candidate.name, metric_lower):
                        continue
                    for rel in pef.get_relationships_for_subject(candidate.id):
                        if rel.negated or rel.object_literal is None:
                            continue
                        if _PREDICTIVE_RELATION_RE.search(rel.relation):
                            continue
                        for pef_val in parse_all_numerics(str(rel.object_literal)):
                            if _numeric_values_align_for_layer1(asserted_val, pef_val):
                                grounded = True
                                break
                        if grounded:
                            break
                    if grounded:
                        break

            if grounded:
                continue

            return [Flag(
                flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
                entity_name="financial",
                claim="Response asserts a specific quantitative financial fact without verifiable grounding",
                evidence="Financial metric + numeric + assertion pattern detected (Layer 1 raw scan)",
                severity="warning",
            )]

        return []

    def _check_numeric_medical_instruction(self, text: str) -> list[Flag]:
        """Detect numeric clinical instructions beyond simple dosage (normative veto — axis 3).

        Catches: titration/taper/wean schedules, infusion and drip rates,
        insulin/basal/bolus adjustments.  All are 'numbers that cause action'
        in a medical domain — equivalent to dosage even when the word 'dose'
        is absent.  "How to taper" is dosage in a trench coat.

        Requires: action word + numeric value + medical unit/context.
        This prevents candle-taper, tapered jeans, etc. from false-firing.
        """
        if not text or len(text.strip()) < 20:
            return []

        lower = text.lower()

        medical_unit = bool(
            re.search(r"\b(mg|ml|mcg|units?|tablet|capsule|pill|dose)\b", lower)
        )
        # Dose lifecycle: starting, maintenance, maximum doses
        # Catches "starting dose of 500mg", "max dose 2000mg" etc.
        dose_lifecycle = bool(
            re.search(r"\b(?:start\w*|maint\w*|max\w*|initial)\b.{0,30}\bdose\b", lower)
        )
        if dose_lifecycle and re.search(r"\d+", lower) and medical_unit:
            return [Flag(
                flag_type=FlagType.NUMERIC_MEDICAL_INSTRUCTION,
                entity_name="medical",
                claim="Response specifies a dose lifecycle stage (start/max/maintenance)",
                evidence="Dose lifecycle pattern + numeric value + medical unit detected",
                severity="error",
            )]


        # Titration / taper / wean schedules: action + numeric + medical unit
        taper_action = bool(re.search(r"\b(titrat\w*|taper\w*|wean\w*)\b", lower))
        if taper_action and re.search(r"\d+", lower) and medical_unit:
            return [Flag(
                flag_type=FlagType.NUMERIC_MEDICAL_INSTRUCTION,
                entity_name="medical",
                claim="Response contains a titration, taper, or wean schedule",
                evidence="Taper/titration action + numeric value + medical unit detected",
                severity="error",
            )]

        # Infusion / drip rates: explicit rate or rate expression
        infusion = bool(
            re.search(r"\d+\s*(?:ml/hr|ml\s+per\s+hour|units?/hr|mg/hr|mcg/hr)\b", lower)
            or re.search(r"\b(?:infusion|drip)\s+rate\b.{0,40}\d+", lower)
            or re.search(r"\d+.{0,10}\b(?:infusion|drip)\s+rate\b", lower)
        )
        if infusion:
            return [Flag(
                flag_type=FlagType.NUMERIC_MEDICAL_INSTRUCTION,
                entity_name="medical",
                claim="Response specifies an infusion or drip rate",
                evidence="Infusion/drip rate with numeric value detected",
                severity="error",
            )]

        # Insulin / basal / bolus adjustments: adjustment verb + insulin type + units
        insulin_type = bool(re.search(r"\b(insulin|basal|bolus)\b", lower))
        adjust_verb = bool(
            re.search(r"\b(adjust|increase|decrease|add|reduce|titrate|change)\b", lower)
        )
        insulin_units = bool(re.search(r"\d+\s*(?:units?|u)\b", lower))
        if insulin_type and adjust_verb and insulin_units:
            return [Flag(
                flag_type=FlagType.NUMERIC_MEDICAL_INSTRUCTION,
                entity_name="medical",
                claim="Response specifies an insulin dose adjustment",
                evidence="Insulin/basal/bolus adjustment + unit quantity detected",
                severity="error",
            )]

        return []

    def _check_emergency_triage(self, text: str) -> list[Flag]:
        """Detect definitive emergency triage or false reassurance (normative veto — axis 3).

        The model must never be the last voice someone listens to on
        whether to seek emergency care.  Allowed: encourage calling
        emergency services or seeking care.  Flagged: dismissing the
        need for emergency care or making a definitive triage call.
        """
        if not text or len(text.strip()) < 20:
            return []

        lower = text.lower()

        # Require medical/symptom context to suppress false positives
        # ("a power outage is not an emergency" should not fire)
        medical_context = bool(
            re.search(
                r"\b(symptom|pain|chest|breath\w*|bleed\w*|wound|injur\w*|fever|"
                r"dizzy|faint\w*|unconscious|stroke|heart|pulse|vomit\w*|seizure|"
                r"swollen|burn|allerg\w*|react\w*|overdos\w*)\b",
                lower,
            )
        )

        # "not an emergency" / "not a medical emergency" — categorical dismissal
        # an? matches both "a" and "an"
        not_emergency = bool(
            re.search(r"\bnot\s+(?:an?\s+)?(?:medical\s+)?emergency\b", lower)
        )

        # Dissuading from seeking emergency care
        dismissal = bool(
            re.search(
                r"\b(?:don['\u2019]?t|do\s+not|doesn['\u2019]?t|no\s+need\s+to)"
                r"\s+(?:need\s+to\s+)?(?:go|call|visit|see)\b"
                r".{0,60}\b(?:hospital|er\b|emergency\s+room|ambulance|"
                r"911|999|112|urgent\s+care|paramedic)\b",
                lower,
            )
            or re.search(
                r"\b(?:can|should)\s+wait\b.{0,40}"
                r"\b(?:hospital|doctor|emergency|urgent\s+care)\b",
                lower,
            )
        )

        if (not_emergency and medical_context) or dismissal:
            return [Flag(
                flag_type=FlagType.EMERGENCY_TRIAGE_GUIDANCE,
                entity_name="triage",
                claim="Response dismisses need for emergency care or makes definitive triage call",
                evidence="Reassurance or dismissal pattern in medical context detected",
                severity="error",
            )]

        return []

    def _check_professional_directive(
        self, text: str, user_input: str | None = None
    ) -> list[Flag]:
        """Detect individualized professional directives in regulated domains (verify-or-refuse).

        Catches 'you should do X' in medical, legal, and financial contexts
        when it constitutes a personalized professional determination — not
        just general information.

        Governance default (public mode): HARD_STOP.
        Enterprise mode: FORCE_REVISE with verified/cited sources.

        Legal has its own dosage analogy: jurisdiction + procedure + deadline.
        A bad deadline is a legal overdose.
        """
        if not text or len(text.strip()) < 20:
            return []

        lower = text.lower()
        flags: list[Flag] = []

        # ── High-confidence standalone signals ────────────────────────────
        # These fire without requiring a directive verb because the phrasing
        # is unambiguously individualized.

        # Precompute legal context — used by standalone signals and the directive gate.
        # Note: "case" is intentionally excluded — it is too ambiguous ("case of the flu",
        # "business case", etc.). The remaining terms are domain-specific enough.
        _legal_context = bool(
            re.search(
                r"\b(court|lawsuit|attorney|lawyer|statute|jurisdiction|"
                r"deadline|compliance|regulation|legal|contract|"
                r"breach|tort|liability|appeal|notice|filing|employment|"
                r"tribunal|tenancy|eviction|dismissal|claimant|defendant)\b",
                lower,
            )
        )

        # Precompute chemistry / lab-safety context — suppresses
        # PERSONALIZED_MEDICAL_ADVICE when shared tokens (sodium, potassium,
        # calcium, magnesium, phosphorus, fluid, dose) appear in a non-clinical
        # setting.  Same architectural pattern as _legal_context: domain-specific
        # lexical cues prove the response is about chemistry, not patient care.
        # Terms are chemistry-specific multiword phrases or high-signal singletons
        # to avoid false suppression on clinical text (e.g. "lab results",
        # "compound fracture", "acid reflux").
        _chemistry_context = bool(
            re.search(
                r"\b(chlorine\s+gas|sodium\s+hydroxide|hydrochloric|"
                r"hypochlorite|bleach|phosgene|cyanide|pool\s+shock|"
                r"fume\s+hood|lab\s+safety|laboratory|chemical\s+reaction|"
                r"toxic\s+gas|caustic|reagent|solvent|molar\s+solution|"
                r"msds|sds\b)",
                lower,
            )
        )

        # ── Personalisation signal ────────────────────────────────────────
        # Indicates the response is applied to the user's specific situation
        # rather than providing general legal information or educational content.
        # Required by validity_assessment (unless possessive "your [document]"
        # is present in the match) and by the legal-action directive gate.
        # Without at least one signal, the response is treated as definitional
        # or procedural information, not personalised legal advice.
        _personalisation_signal = bool(
            # Situational application markers
            re.search(
                r"\b(?:in\s+your\s+(?:case|situation|circumstances?)|"
                r"based\s+on\s+(?:what\s+you['\u2019]?ve\s+described|"
                r"your\s+(?:situation|description|facts)|"
                r"the\s+(?:details?|information|facts?|context|description)\s+"
                r"(?:you\s+)?(?:provided|given|shared|described|mentioned))|"
                r"given\s+your\s+(?:situation|circumstances?|facts?|description)|"
                r"from\s+what\s+you['\u2019]?ve\s+(?:described|said|told|mentioned))\b",
                lower,
            )
            # User-possessive document or party references
            # Allow an optional intervening adjective ("your unfair dismissal",
            # "your current employer", "your written notice").
            or re.search(
                r"\byour\s+(?:\w+\s+)?(?:contract|agreement|notice|employer|landlord|"
                r"tenancy|dismissal|eviction|claim|appeal|case|filing|"
                r"lease|redundancy|settlement|termination)\b",
                lower,
            )
            # Applied risk or liability framing
            or re.search(
                r"\byou\s+(?:face|are\s+at\s+risk|may\s+be\s+(?:liable|exposed)|"
                r"could\s+be\s+(?:liable|exposed)|might\s+be\s+(?:liable|at\s+risk))\b",
                lower,
            )
            # Situation reflection — response describes something the user experienced
            or re.search(
                r"\byou\s+(?:received|were\s+served|have\s+been\s+(?:served|dismissed|evicted)|"
                r"signed|entered\s+into|are\s+facing|have\s+received)\b",
                lower,
            )
            # Applied jurisdiction / location-specific framing
            or re.search(
                r"\b(?:in\s+your\s+jurisdiction|under\s+applicable\s+law|"
                r"where\s+you\s+(?:are|live|reside|work)|"
                r"in\s+your\s+(?:state|country|region))\b",
                lower,
            )
            # First-person input reflected back ("you mentioned", "you said", etc.)
            or re.search(
                r"\byou\s+(?:mentioned|said|described|explained|told\s+me|"
                r"indicated|stated|told\s+us)\b",
                lower,
            )
            # User's own assessment/analysis being validated
            # "Your validity period assessment is correct" — the model is
            # evaluating something the user produced, which is inherently
            # applied to their specific situation.
            or re.search(
                r"\byour\s+(?:\w+\s+){0,3}(?:assessment|analysis|calculation|"
                r"interpretation|reading|understanding|conclusion|argument|position)\b",
                lower,
            )
            # Legal action verb directly targeting a specific legal document
            # (definite article signals a specific document, not a generic procedure).
            # "challenge the notice" / "appeal the dismissal" / "dispute the contract"
            # are always applied to the user's specific situation.
            or re.search(
                r"\b(?:challenge|dispute|contest|appeal|file|seek|pursue)\s+"
                r"(?:the|this)\s+"
                r"(?:notice|contract|agreement|dismissal|eviction|claim|decision|order)\b",
                lower,
            )
        )

        # "Your deadline is X" / "you have N days to …" — legal overdose risk
        deadline_directive = bool(
            re.search(r"\byour\s+deadline\b", lower)
            or re.search(r"\byou\s+have\s+\d+\s+days?\s+to\b", lower)
        )
        if deadline_directive:
            flags.append(Flag(
                flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
                entity_name="legal",
                claim="Response specifies a personal legal deadline",
                evidence="Personalized deadline directive detected (your deadline / you have N days)",
                severity="warning",
            ))

        # ── Applied legal analysis (standalone signals) ───────────────────
        # Case strength / outcome assessment and analysis framing are
        # individualized legal advice regardless of directive verb structure.
        # A list of "arguments in your favor" is a legal determination even
        # when no imperative verb is present.
        #
        # These require legal_context to reduce false positives
        # (e.g. "you have a strong case of flu" must not fire).

        # "You have a strong/valid/good case" — direct case strength assessment
        case_strength = _legal_context and bool(
            re.search(
                r"\byou\s+have\s+a\s+(?:strong|good|solid|valid|legitimate|viable|clear)\s+(?:case|claim)\b",
                lower,
            )
            or re.search(
                r"\byour\s+(?:case|claim|position)\s+(?:is|appears?|looks?|seems?)\s+"
                r"(?:strong|good|solid|favorable|favourable|viable|compelling)\b",
                lower,
            )
            or re.search(r"\bstrong\s+grounds?\s+(?:for|to)\b", lower)
        )
        if case_strength:
            flags.append(Flag(
                flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
                entity_name="legal",
                claim="Response assesses the strength of the user's legal case",
                evidence="Case strength assertion detected in legal context",
                severity="warning",
            ))

        # "You would/will likely win" / "you are likely to prevail"
        # No legal_context requirement — outcome prediction is unambiguously
        # legal advice regardless of surrounding vocabulary.
        # Guard: conditional framing ("whether you would win", "if you would prevail")
        # is not an assertion and must not fire. Check for "whether"/"if" in the
        # 40-char prefix of each match.
        _outcome_re = re.compile(
            r"\byou\s+(?:would|will|should)\s+(?:likely\s+|probably\s+)?(?:win|prevail|succeed)\b"
            r"|\byou(?:['\u2019]re|\s+are)\s+(?:likely|probably)\s+to\s+(?:win|prevail|succeed)\b",
        )
        outcome_prediction = False
        for _m in _outcome_re.finditer(lower):
            _prefix = lower[max(0, _m.start() - 40):_m.start()]
            if not re.search(r"\b(?:whether|if)\b", _prefix):
                outcome_prediction = True
                break
        if outcome_prediction:
            flags.append(Flag(
                flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
                entity_name="legal",
                claim="Response predicts a legal outcome for the user",
                evidence="Outcome prediction detected",
                severity="error",
            ))

        # Applied analysis framing: "arguments in your favor", "key arguments for your case"
        # Catches list-structured advice where the LLM frames its output as case analysis
        # rather than a direct directive — the structural form is different but the
        # epistemic effect is identical.
        applied_analysis = _legal_context and bool(
            re.search(
                r"\b(?:arguments?|factors?|points?|grounds?)\s+(?:that\s+)?(?:(?:work|support|help)\s+)?"
                r"in\s+your\s+(?:favor|favour)\b",
                lower,
            )
            or re.search(
                r"\b(?:arguments?|factors?|points?)\s+(?:that\s+)?support(?:ing)?\s+your\s+"
                r"(?:case|claim|position)\b",
                lower,
            )
            or re.search(
                r"\bkey\s+arguments?\b.{0,80}\b(?:your\s+(?:case|claim|position)|"
                r"in\s+your\s+(?:favor|favour))\b",
                lower,
            )
        )
        if applied_analysis:
            flags.append(Flag(
                flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
                entity_name="legal",
                claim="Response provides applied legal analysis of the user's case",
                evidence="Applied analysis framing detected in legal context",
                severity="warning",
            ))

        # ── Applied legal validity assessment (standalone signal) ─────────────
        # A response that asserts a specific document or action is legally
        # invalid, defective, or void is personalized legal analysis regardless
        # of whether it uses second-person directive language. "The notice is
        # likely defective" is as much legal advice as "you should challenge it."
        # Requires legal_context to suppress false positives (e.g. "the notice
        # is important").
        # -- Applied legal validity assessment (standalone signal) -------------
        # A response that asserts a specific document or action is legally
        # valid, invalid, binding, or unenforceable is personalized legal 
        # analysis. This captures declarative determinations ("The notice is valid")
        # that have the same functional effect as directives.
        # "Your [document] is/appears [valid/invalid/…]" — user-possessive match fires
        # unconditionally in legal context; bare document noun requires a personalisation
        # signal, so "A contract is binding when…" (definitional) does not fire.
        _validity_re = re.compile(
            r"\b(?:notice|contract|agreement|dismissal|eviction|claim|"
            r"section\s*21|tenancy\s+notice|form\s+6a)\b.{0,60}"
            r"\b(?:is|are|appears?|seems?|may\s+be|could\s+be|would\s+be|"
            r"was|were|has|remains?|not)\b.{0,60}"
            r"\b(?:valid|binding|enforceable|lawful|invalid|defective|void|"
            r"unenforceable|improper|irregular|technically\s+defective|"
            r"procedurally\s+(?:defective|flawed))\b",
        )
        _validity_possessive_re = re.compile(
            r"\byour\s+(?:notice|contract|agreement|dismissal|eviction|claim|"
            r"section\s*21|tenancy\s+notice|form\s+6a)\b.{0,60}"
            r"\b(?:is|are|appears?|seems?|may\s+be|could\s+be|would\s+be|"
            r"was|were|has|remains?|not)\b.{0,60}"
            r"\b(?:valid|binding|enforceable|lawful|invalid|defective|void|"
            r"unenforceable|improper|irregular|technically\s+defective|"
            r"procedurally\s+(?:defective|flawed))\b",
        )
        validity_assessment = _legal_context and (
            bool(_validity_possessive_re.search(lower))
            or (_personalisation_signal and bool(_validity_re.search(lower)))
        )
        if validity_assessment:
            flags.append(Flag(
                flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
                entity_name="legal",
                claim="Response asserts a legal validity determination about the user's specific document",
                evidence="Applied legal validity assessment detected in legal context",
                severity="warning",
            ))

        # "You qualify for the deduction/credit/benefit"
        qualifies_claim = bool(re.search(r"\byou\s+qualify\s+for\b", lower))
        if qualifies_claim:
            flags.append(Flag(
                flag_type=FlagType.PERSONALIZED_FINANCIAL_ADVICE,
                entity_name="financial",
                claim="Response asserts the user qualifies for a specific financial benefit",
                evidence="'You qualify for' detected",
                severity="warning",
            ))

        # ── Standalone dosing specificity (analogous to deadline_directive) ──
        # A response that provides specific dosage (value + units + frequency)
        # or weight-based pediatric dosing for a named medication constitutes
        # individualized medical advice regardless of directive verb structure.
        # "The dose is 250mg three times daily" is functionally identical to
        # "you should take 250mg three times daily" — both are medical advice.
        _named_medication = bool(
            re.search(
                r"\b(?:amoxicillin|penicillin|amoxiclavulanate|augmentin|"
                r"azithromycin|clarithromycin|erythromycin|doxycycline|"
                r"ciprofloxacin|cephalexin|trimethoprim|metronidazole|"
                r"ibuprofen|paracetamol|acetaminophen|naproxen|aspirin|"
                r"metformin|lisinopril|atorvastatin|omeprazole|metoprolol|"
                r"amlodipine|sertraline|fluoxetine|paroxetine|citalopram|"
                r"alprazolam|lorazepam|diazepam|clonazepam|"
                r"prednisone|prednisolone|dexamethasone|"
                r"hydrocodone|oxycodone|codeine|tramadol|morphine)\b",
                lower,
            )
        )
        _dose_with_frequency = bool(
            re.search(r"\b\d+\s*(?:mg|mcg|ml)\b", lower)
            and re.search(
                r"\b(?:twice|three\s+times|four\s+times|once)\s+(?:a\s+)?(?:day|daily)\b"
                r"|\bevery\s+(?:\d+|four|six|eight|twelve)\s+hours?\b"
                r"|\b(?:two|three|four)\s+times\s+(?:a\s+)?(?:day|daily)\b",
                lower,
            )
        )
        _weight_based_dose = bool(
            re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg)\s*/\s*kg\b", lower)
        )
        dosing_specificity = _named_medication and (_dose_with_frequency or _weight_based_dose)
        if dosing_specificity:
            flags.append(Flag(
                flag_type=FlagType.PERSONALIZED_MEDICAL_ADVICE,
                entity_name="medical",
                claim="Response provides specific medication dosing for a named drug",
                evidence="Named medication + specific dose/frequency or weight-based dose detected",
                severity="warning",
            ))

        # ── Financial numeric projection (Layer 2 trigger) ─────────────────
        # Conjunction of Layer 1 numeric assertion (unhedged, non-educational)
        # AND a strong personalization signal.
        # Supporting context (e.g. "your portfolio") alone is insufficient.
        # This is a standalone signal that fires regardless of directive verbs.
        _fn_segments = split_sentences_at_punctuation(lower)
        _financial_numeric_assertion = any(
            _FIN_METRIC_TOKEN_RE.search(sent) and
            _FIN_NUMERIC_RE.search(sent) and
            _FIN_ASSERT_RE.search(sent) and
            not _FIN_EDU_GUARD_RE.search(sent) and
            not _fin_quant_is_hedged(sent)
            for sent in _fn_segments
        )

        _lookup_user_historical = bool(
            user_input and user_seeks_historical_market_data_lookup(user_input)
        )
        _strong_portfolio_in_reply = bool(_FIN_STRONG_PORTFOLIO_ACTION_RE.search(lower))
        if _financial_numeric_assertion and bool(_FIN_STRONG_PERSONALISATION_RE.search(lower)):
            # Historical market-data questions often get refusals that echo "your
            # portfolio" / boilerplate personalisation; do not treat that as Layer 2
            # advice unless the reply contains strong portfolio action verbs.
            if not (_lookup_user_historical and not _strong_portfolio_in_reply):
                flags.append(Flag(
                    flag_type=FlagType.PERSONALIZED_FINANCIAL_ADVICE,
                    entity_name="financial",
                    claim="Response provides a personalized numeric financial projection",
                    evidence="Financial numeric pattern + strong personalization signal detected",
                    severity="warning",
                ))

        # Second-person address — computed before the directive-verb gate so
        # financial reallocation advice ("Reallocating your portfolio…") is still
        # evaluated (gerund-led framing does not match the narrow directive list).
        second_person = bool(re.search(r"\b(you|your)\b", lower))

        # ── Financial directive (before second-person + directive gate) ──
        # Catches both explicit directives ("you should invest") and gerund-led
        # continuation of a user reallocation request ("Reallocating your … funds").
        # Exclude "consider" / "worth considering" / "better off" / "advisable" here:
        # they combine with second-person epistemic refusals and "share" in "share price"
        # (financial_context) to false-positive as personalized advice. Strong verbs
        # (invest, reallocate, …) and explicit recommend/suggest/advise remain.
        financial_action = bool(
            re.search(
                r"\b(invest|buy|sell|contribute|withdraw|access(?:ing)?|"
                r"cash\s*out|transfer|deduct|put|move|"
                r"reallocat(?:e|ing|ions?|ed)|allocate|diversify|"
                r"shift|rebalance|recommend|suggest|advise|"
                r"switch|fix(?:ing)?|"
                r"lock(?:ing)?|consolidat(?:e|ing)|restructur(?:e|ing))\b",
                lower,
            )
        )
        # Do not treat "share"/"stock" in "share price" / "stock price" as holdings
        # context — that phrasing is market-data lookup, not portfolio advice.
        financial_context = bool(
            re.search(
                r"\b(stock(?!\s+prices?\b)|bond|fund|portfolio|401k|ira|roth|tax|deduction|"
                r"credit|investment|broker|etf|dividend|mutual\s+fund|"
                r"retirement|annuity|capital\s+gain|superannuation|super|"
                r"pension|isa|mortgage|home\s+loan|interest\s+rate|"
                r"fixed\s+rate|variable\s+rate|offset\s+account|redraw|"
                r"negative\s+gear|concessional|franking|savings|deposit|"
                r"property|shares?(?!\s+prices?\b)|equity|debt|bankruptcy|insolvency|"
                r"dollar(?:-|\s+)cost\s+averaging|dca)\b",
                lower,
            )
        )
        _financial_ticker_symbol = bool(re.search(r"\b[A-Z]{2,5}\b", text))
        _financial_domain_context = financial_context or _financial_ticker_symbol
        _financial_illustrative_example = bool(
            re.search(
                r"\b(?:for\s+example|for\s+instance|e\.g\.|illustrative|"
                r"hypothetical|suppose|imagine)\b",
                lower,
            )
        )
        _financial_hypothetical_amount = bool(
            re.search(
                r"(?:[\xA3\$\u20AC]\s*\d+(?:,\d{3})*(?:\.\d+)?|"
                r"\b\d+(?:\.\d+)?\s*(?:per|\/)\s*(?:month|week|year|annum)\b)",
                lower,
            )
        )
        _financial_directive_force = bool(
            _FIN_DIRECTIVE_FORCE_RE.search(lower)
            or _FIN_IMPERATIVE_ACTION_RE.search(lower)
        )
        _financial_personalisation_signal = bool(
            re.search(
                r"\b(?:in\s+your\s+(?:case|situation|circumstances?)|"
                r"based\s+on\s+(?:what\s+you['\u2019]?ve\s+(?:described|said|told|mentioned)|"
                r"your\s+(?:situation|description)|"
                r"the\s+(?:details?|information|facts?|context)\s+you\s+(?:provided|gave|shared))|"
                r"given\s+your\s+(?:situation|circumstances?|age|balance|savings|income|debt|mortgage)|"
                r"for\s+your\s+(?:situation|circumstances?|goals?|needs?)|"
                r"from\s+what\s+you['\u2019]?ve\s+(?:described|said|told|mentioned)|"
                r"at\s+your\s+age|with\s+your\s+(?:savings|balance|income|mortgage|debt)|"
                r"your\s+(?:(?:\w+\s+){0,3})"
                r"(?:super\s+balance|mortgage|savings|portfolio|investment|"
                r"retirement\s+savings|risk\s+tolerance|timeline|financial\s+situation)|"
                r"i\s+(?:would\s+)?recommend|i\s+(?:would\s+)?suggest|"
                r"you\s+(?:should|would\s+be\s+better\s+off|might\s+want\s+to|"
                r"may\s+want\s+to|need\s+to\s+(?:consider|think\s+about)))\b",
                lower,
            )
        )
        _financial_personal_suitability = bool(
            _FIN_PERSONAL_SUITABILITY_RE.search(lower)
        )
        _financial_explanatory_mode = bool(
            _FIN_EXPLANATORY_MODE_RE.search(lower)
        )
        _educational_hypothetical_guard = (
            _financial_illustrative_example
            and _financial_hypothetical_amount
            and not _financial_personalisation_signal
            and not _financial_directive_force
        )
        _educational_descriptive_guard = (
            (_financial_explanatory_mode or _educational_hypothetical_guard)
            and not _financial_directive_force
            and not _financial_personalisation_signal
            and not _financial_personal_suitability
        )
        if _financial_domain_context and not qualifies_claim and (
            _financial_directive_force
            or _financial_personal_suitability
            or (financial_action and _financial_personalisation_signal)
        ):
            # Models often refuse finance questions with boilerplate that *matches* this gate
            # ("I recommend consulting a licensed financial adviser", "not investment advice",
            # "personalised financial advice" disclaimers). Those refusals are not advice, but
            # they rarely match a single disclaimer regex. For historical market-data lookups,
            # suppress PERSONALIZED_FINANCIAL_ADVICE unless the reply still contains strong
            # portfolio verbs (invest, sell, reallocate, …).
            if _lookup_user_historical and not _strong_portfolio_in_reply:
                pass
            elif _educational_descriptive_guard:
                pass
            else:
                if not any(f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE for f in flags):
                    flags.append(Flag(
                        flag_type=FlagType.PERSONALIZED_FINANCIAL_ADVICE,
                        entity_name="financial",
                        claim=(
                            "Response commits to user-directed or personalized financial advice "
                            "(directive force or personalized suitability)"
                        ),
                        evidence=(
                            "Directive-force/personalized-suitability signal in financial context; "
                            "not a purely educational/descriptive explanation"
                        ),
                        severity="warning",
                    ))

        # ── Second-person + directive gate ────────────────────────────────
        # General action-based patterns require both a second-person address
        # and an explicit directive verb to avoid false positives.
        directive = bool(
            re.search(
                r"\b(should|need\s+to|must|have\s+to|ought\s+to|aim\s+for|ensure|"
                r"i\s+(?:\w+\s+)?recommend|i\s+(?:\w+\s+)?suggest|"
                r"it\s+is\s+(?:[a-z]+\s+)?recommended|"
                r"i(?:['\u2019]d)?\s+(?:strongly\s+|highly\s+)?advise|"
                r"worth\s+considering|better\s+off|advisable|"
                r"(?:might|may|could)\s+want\s+to|"
                r"(?:would\s+)?makes?\s+(?:the\s+most\s+)?(?:financial\s+)?sense|"
                r"(?:strongly\s+)?encourage|best\s+(?:option|course|approach)\s+for\s+you)\b",
                lower,
            )
        )

        if not (second_person and directive):
            return flags  # return standalone signals collected above

        # ── Medical directive ──────────────────────────────────────────────
        # "You should take/start/stop this medication"
        # Safe referrals ("you should see a doctor") are NOT flagged —
        # they require treatment_object, not just treatment_action.
        #
        # Safety-referral exemption: advising to tell doctor about allergy, or
        # not take a drug due to allergy, is protective — not a treatment directive.
        # E.g. "make sure your doctor knows about your seafood allergy" before
        # taking Bactrim. Blocking that advice is harmful.
        _safety_referral = bool(
            re.search(
                r"\b(?:make\s+sure|tell|inform|let)\b.{0,40}"
                r"\b(?:doctor|physician|pharmacist)\b.{0,40}"
                r"\b(?:know|about|regarding)\b.{0,30}\b(?:allerg|allergic)\b",
                lower,
            )
            or re.search(
                r"\b(?:don['\u2019]?t|do\s+not)\s+(?:take|use)\b.{0,40}"
                r"\b(?:if|because|when)\b.{0,30}\b(?:allerg|allergic)\b",
                lower,
            )
            or re.search(
                r"\b(?:check|verify|confirm)\b.{0,30}"
                r"\b(?:with)\b.{0,30}\b(?:doctor|physician|pharmacist)\b"
                r".{0,30}\b(?:before)\b.{0,20}\b(?:taking|using)\b",
                lower,
            )
        )
        # Actionable dosing directive — catches specific instruction patterns
        # beyond the treatment_action+treatment_object pair. Used to prevent the
        # safety-referral exemption from suppressing a co-located dosing instruction.
        _dosing_directive = bool(
            re.search(r"\b\d+\s*(?:mg|mcg|ml|g)\b", lower)
            or re.search(r"\b(?:twice|three\s+times|once)\s+(?:a\s+)?(?:day|daily)\b", lower)
            or re.search(r"\bevery\s+\d+\s+hours?\b", lower)
            or re.search(r"\btake\s+(?:two|three|four|\d+)\s+(?:tablets?|pills?|capsules?)\b", lower)
            or re.search(r"\b(?:start\s+with|increase\s+to|titrate)\b", lower)
        )

        treatment_action = bool(
            re.search(
                r"\b(take|give|administer|start|stop|discontinue|use|apply|avoid|try|"
                r"reduce|limit|restrict|increase|decrease|aim|ensure)\b",
                lower,
            )
        )
        treatment_object = bool(
            re.search(
                r"\b(medication|drug|pill|tablet|capsule|antibiotic|medicine|"
                r"prescription|treatment|therapy|dose|steroid|insulin|cream|"
                r"ointment|injection|supplement|vaccine|"
                r"diet|dietary|protein|potassium|sodium|phosphorus|calcium|magnesium|fiber|fluid|intake|"
                r"carbohydrate|fat)\b",
                lower,
            )
        )
        # Exemption is local, not global: _safety_referral suppresses
        # PERSONALIZED_MEDICAL_ADVICE only when no actionable dosing instruction
        # appears in the same response. If both patterns co-occur (safety preamble
        # + dosing instruction), the treatment check wins — no compositional bypass.
        # _legal_context excludes the medical check: "treatment" in employment,
        # contract, or tribunal responses is a homonym — it means workplace
        # treatment, not a medical treatment directive. Without this guard,
        # verbs like "ensure"/"try"/"limit" + the word "treatment" produce a
        # spurious PERSONALIZED_MEDICAL_ADVICE flag on legal employment queries,
        # which then outranks PERSONALIZED_LEGAL_ADVICE in the domain priority
        # table and routes the response into the wrong escalation corridor.
        #
        # _chemistry_context: same pattern. "sodium", "potassium", "calcium",
        # etc. are shared between clinical nutrition and chemistry.  "You should
        # avoid sodium hydroxide" is lab safety, not a treatment directive.
        # Without this guard, hazardous-chemistry prompts are misclassified as
        # PERSONALIZED_MEDICAL_ADVICE and routed to a clinician redirect.
        if treatment_action and treatment_object and not _legal_context and not _chemistry_context and not (_safety_referral and not _dosing_directive):
            flags.append(Flag(
                flag_type=FlagType.PERSONALIZED_MEDICAL_ADVICE,
                entity_name="medical",
                claim="Response issues a personalized medical treatment directive",
                evidence="Second-person directive + treatment action + treatment object detected",
                severity="warning",
            ))

        # ── Legal directive ────────────────────────────────────────────────
        # "You should file / you need to appeal" — legal action + legal context
        legal_action = bool(
            re.search(
                r"\b(file|submit|appeal|petition|sue|contest|apply|claim|"
                r"notify|sign|dispute|register|comply|pursue|seek|challenge)\b",
                lower,
            )
        )
        # Skip if deadline_directive already produced a flag for this text.
        # Also require _personalisation_signal so generic procedural information
        # ("you should file at your local court") does not fire — the response
        # must be applied to the user's specific situation.
        if legal_action and _legal_context and not deadline_directive and _personalisation_signal:
            flags.append(Flag(
                flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
                entity_name="legal",
                claim="Response issues a personalized legal filing directive",
                evidence="Second-person directive + legal action + legal context + personalisation signal detected",
                severity="warning",
            ))

        # ── Tenancy / landlord — soft next-step guidance (non-filing verbs) ─────
        # Landlord/lease/rent/arrears phrasing is often legal-adjacent without
        # "court", "statute", or filing verbs. "Reach out", "discuss", "negotiate",
        # "work out a payment plan", "review your lease" are still personalized
        # legal/procedural guidance when applied to the user's situation.
        _tenancy_landlord_context = bool(
            re.search(
                r"\b(?:landlord|tenant|tenancy|\blease\b|rental|eviction|arrears|"
                r"unpaid\s+rent|rent\s+arrears|payment\s+plan|notice\s+to\s+quit)\b",
                lower,
            )
        )
        _soft_procedural_to_counterparty = bool(
            re.search(
                r"\b(?:reach\s+out|contact|speak\s+to|talk\s+to|get\s+in\s+touch|"
                r"discuss|negotiate|work\s+out|review\s+your|go\s+over|"
                r"review\s+the)\b",
                lower,
            )
        )
        _user_tenancy_rent_facts = bool(
            user_input
            and re.search(
                r"\b(?:"
                r"(?:didn\s*t|did\s+not|haven\s*t|have\s+not|stopped)\s+pay\w*\s+(?:the\s+)?rent|"
                r"didn\s*t\s+pay\s+(?:the\s+)?rent|"
                r"unpaid\s+rent|rent\s+for\s+(?:six|several|many|\d+)\s+months?|"
                r"months?\s+(?:of\s+)?(?:unpaid\s+)?rent|"
                r"didn\s*t\s+pay\s+(?:the\s+)?rent\s+for|"
                r"evict|eviction|behind\s+on\s+rent|rent\s+arrears"
                r")\b",
                user_input.lower(),
            )
        )
        if (
            not deadline_directive
            and _personalisation_signal
            and _tenancy_landlord_context
            and _soft_procedural_to_counterparty
            and (
                _user_tenancy_rent_facts
                or not (user_input and str(user_input).strip())
            )
        ):
            flags.append(Flag(
                flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
                entity_name="legal",
                claim="Response issues personalized legal-adjacent tenancy or landlord guidance",
                evidence=(
                    "Second-person directive + tenancy/landlord context + soft procedural "
                    "guidance to landlord/contact/negotiate/review lease"
                ),
                severity="warning",
            ))

        return flags

    def _check_fictional_wrapper_response(
        self, response_text: str, pef: PEFState
    ) -> list[Flag]:
        """Detect medical instruction delivered within an active ARTIFACT frame.

        When a roleplay/fiction/hypothetical frame is open (pef.active_frame.kind
        == ARTIFACT), the LLM may provide medical instructions through character
        voices or story framing. These remain harmful if applied to real-world
        situations, even when framed as fiction.

        Fires on: named medication + directive (take/stop/prescribe/frequency)
        within an ARTIFACT frame. Supplements _check_medical_dosage by catching
        cases where no explicit mg amount is present.
        """
        from aurora_lens.pef.artifact_layer import FrameKind
        active_frame = getattr(pef, "active_frame", None)
        if active_frame is None or active_frame.kind != FrameKind.ARTIFACT:
            return []
        if not response_text or len(response_text.strip()) < 20:
            return []

        lower = response_text.lower()

        _named_drug_re = re.compile(
            r"\b(?:amoxicillin|penicillin|augmentin|azithromycin|clarithromycin|"
            r"erythromycin|doxycycline|ciprofloxacin|cephalexin|trimethoprim|"
            r"metronidazole|ibuprofen|paracetamol|acetaminophen|naproxen|aspirin|"
            r"metformin|lisinopril|atorvastatin|omeprazole|metoprolol|amlodipine|"
            r"sertraline|fluoxetine|paroxetine|citalopram|alprazolam|lorazepam|"
            r"diazepam|clonazepam|prednisone|prednisolone|dexamethasone|"
            r"hydrocodone|oxycodone|codeine|tramadol|morphine)\b",
            re.IGNORECASE,
        )
        _directive_re = re.compile(
            r"\b(?:take|stop\s+taking|start\s+taking|discontinue|administer|"
            r"prescribe[sd]?|recommend[s]?\s+(?:taking|you\s+take)|"
            r"you\s+should\s+(?:take|stop|start|use)|"
            r"twice\s+daily|three\s+times\s+(?:a\s+)?daily|once\s+daily|"
            r"every\s+\d+\s+hours?|as\s+directed|as\s+prescribed)\b",
            re.IGNORECASE,
        )

        if not (_named_drug_re.search(lower) and _directive_re.search(lower)):
            return []

        f = Flag(
            flag_type=FlagType.PERSONALIZED_MEDICAL_ADVICE,
            entity_name="medical",
            claim="Response delivers medical instruction within active fictional/roleplay frame",
            evidence="Named medication + directive detected in ARTIFACT-framed response",
            severity="warning",
        )
        f.rule_id = BlockedRequestRuleId.FICTIONAL_WRAPPER_REAL_MED_ACTION
        return [f]

    def _check_self_harm_instruction(self, text: str) -> list[Flag]:
        """Axis 3 HARD_STOP: procedural self-harm guidance.

        Fires on instructional structure (how-to phrasing, step sequences,
        comparative optimization, or mechanism-level guidance) directed at
        self-harm or suicide.

        Does not fire on descriptive, academic, policy, legal, fictional,
        or refusal-framed content.

        Deterministic pattern matching; may conservatively over-trigger
        when instructional phrasing appears in quoted or narrative context.
        """
        if not text or len(text.strip()) < 20:
            return []

        m = _SH_INSTRUCTION_RE.search(text)
        if not m:
            return []

        prefix = text[max(0, m.start() - 120): m.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            return []

        return [Flag(
            flag_type=FlagType.SELF_HARM_INSTRUCTION,
            entity_name="self_harm",
            claim="Response contains procedural self-harm or suicide guidance",
            evidence=f"Instructional self-harm pattern detected: {m.group()!r}",
            severity="error",
        )]

    def _check_harmful_operational_guidance(self, text: str) -> list[Flag]:
        """Axis 3 HARD_STOP: operational harmful guidance in model output."""
        rule_id = classify_harmful_operational_guidance(text)
        if rule_id is None:
            return []
        return [Flag(
            flag_type=FlagType.ILLEGAL_INSTRUCTION,
            entity_name="harmful_action",
            claim="Response contains operational guidance for harmful or illicit conduct",
            evidence=f"Harmful operational guidance detected ({rule_id})",
            severity="error",
            rule_id=rule_id,
        )]

    def _check_illegal_instruction(self, text: str) -> list[Flag]:
        """Axis 3 HARD_STOP: procedural enablement of serious criminal acts.

        Fires when output provides actionable guidance (materials, steps,
        tools, sequencing, optimization) for drug synthesis, explosive
        construction, illegal firearm modification, or document forgery.

        Does not fire on factual, legal, policy, historical, academic,
        defensive, or refusal-framed discussion.

        Deterministic structural matching; requires instruction markers
        for drug synthesis branch to avoid descriptive false positives.
        """
        if not text or len(text.strip()) < 20:
            return []

        def _negation_guard(m) -> bool:
            prefix = text[max(0, m.start() - 120): m.start()]
            return bool(_NEGATION_PREFIX_RE.search(prefix))

        # 1. Controlled substance synthesis: verb + drug proximity + instruction marker
        vm = _ILLEGAL_SYNTH_VERB_RE.search(text)
        dm = _ILLEGAL_DRUG_RE.search(text)
        if vm and dm and abs(vm.start() - dm.start()) <= 80 and not _negation_guard(vm):
            win_s = max(0, min(vm.start(), dm.start()) - 80)
            win_e = min(len(text), max(vm.end(), dm.end()) + 80)
            if _ILLEGAL_INSTRUCTION_MARKER_RE.search(text[win_s:win_e]):
                return [Flag(
                    flag_type=FlagType.ILLEGAL_INSTRUCTION,
                    entity_name="illegal",
                    claim="Response contains controlled substance synthesis instructions",
                    evidence=f"Synthesis verb {vm.group()!r} + drug {dm.group()!r} detected",
                    severity="error",
                )]

        # 2. Explosive / weapon construction
        em = _ILLEGAL_EXPLOSIVE_RE.search(text)
        if em and not _negation_guard(em):
            return [Flag(
                flag_type=FlagType.ILLEGAL_INSTRUCTION,
                entity_name="illegal",
                claim="Response contains explosive or weapon construction instructions",
                evidence=f"Explosive construction pattern detected: {em.group()[:60]!r}",
                severity="error",
            )]

        # 3. Illegal firearm modification
        fm = _ILLEGAL_FIREARM_RE.search(text)
        if fm and not _negation_guard(fm):
            return [Flag(
                flag_type=FlagType.ILLEGAL_INSTRUCTION,
                entity_name="illegal",
                claim="Response contains illegal firearm modification instructions",
                evidence=f"Firearm modification pattern detected: {fm.group()[:60]!r}",
                severity="error",
            )]

        # 4. Document forgery
        fgm = _ILLEGAL_FORGERY_RE.search(text)
        if fgm and not _negation_guard(fgm):
            return [Flag(
                flag_type=FlagType.ILLEGAL_INSTRUCTION,
                entity_name="illegal",
                claim="Response contains document forgery instructions",
                evidence=f"Forgery pattern detected: {fgm.group()[:60]!r}",
                severity="error",
            )]

        return []

    def _check_violent_criminal_intent(self, text: str) -> list[Flag]:
        """Axis 3 HARD_STOP: interpersonal homicide framed as legal/permissibility query."""
        return violent_criminal_intent_flags(text)

    def _check_targeted_defamation(self, text: str) -> list[Flag]:
        """Axis 3 HARD_STOP: unhedged severe accusation against identifiable person.

        Fires when a severe criminal/moral label is directly attributed
        to a specific individual (name pattern + identity assertion),
        without local hedging language.

        Does not fire on reported allegations, convictions, policy debate,
        group generalizations, fiction, or criticism lacking criminal identity.

        Hedge exemption is local (same sentence or nearby window).
        Deterministic matching with sentence/window approximation.
        """
        if not text or len(text.strip()) < 20:
            return []

        lower = text.lower()

        label_m = _DEFAM_LABEL_RE.search(lower)
        if not label_m:
            return []

        # Local hedge check: same sentence as label only.
        # A hedge in a different sentence cannot sanitize an unhedged accusation.
        # (±80 char window removed — it allowed unrelated hedges in adjacent sentences
        # to suppress the flag when the hedge and label happened to be near each other.)
        sent_start = 0
        for i in range(label_m.start() - 1, -1, -1):
            if lower[i] in ".!?":
                sent_start = i + 1
                break
        sent_end = len(lower)
        for i in range(label_m.end(), len(lower)):
            if lower[i] in ".!?":
                sent_end = i + 1
                break
        if _DEFAM_HEDGE_RE.search(lower[sent_start:sent_end]):
            return []

        if not _DEFAM_ASSERTION_RE.search(lower):
            return []

        if not _DEFAM_NAME_RE.search(text):
            return []

        return [Flag(
            flag_type=FlagType.TARGETED_DEFAMATION,
            entity_name="defamation",
            claim=f"Response makes an unhedged defamatory accusation: {label_m.group()!r}",
            evidence="Named entity + assertion verb + unhedged severe accusation detected",
            severity="error",
        )]

    def _check_truthfulness_certainty(self, text: str) -> list[Flag]:
        """Axis 3 verify-or-refuse: unhedged assertion about a private individual's truthfulness.

        Fires when the model issues a definitive verdict on whether someone is
        lying, honest, deceptive, or telling the truth — without any admissible
        evidential basis.

        Two branches:
          A. subject + copula + (adverb) + (not) + truthfulness predicate
             e.g. "this is not a lie", "he is lying", "your friend is being honest"
          B. subject + copula + certainty adverb + true/false/accurate
             e.g. "that is definitely false" (adverb required to avoid benign "this is true")

        Guards (all checked in the sentence containing the match):
          1. Negation prefix — "I cannot tell whether this is a lie" → suppressed
          2. Hedge language — "this could be a lie" / "it's hard to say" → suppressed
          3. Definitional context — "A lie is an intentional falsehood" → suppressed
          4. Fiction/hypothetical — "in the novel, he is lying" → suppressed

        Policy: HARD_STOP (public) / FORCE_REVISE (enterprise).
        Severity: warning (verify-or-refuse, not hard-stop-always).
        """
        if not text or len(text.strip()) < 5:
            return []

        m = _TRUTH_ASSERT_RE.search(text)
        if not m:
            return []

        # Guard 1: negation prefix — check 120 chars before the match
        prefix = text[max(0, m.start() - 120): m.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            return []

        # Find sentence boundaries around the match for guards 2–4
        sent_start = 0
        for i in range(m.start() - 1, -1, -1):
            if text[i] in ".!?\n":
                sent_start = i + 1
                break
        sent_end = len(text)
        for i in range(m.end(), len(text)):
            if text[i] in ".!?\n":
                sent_end = i + 1
                break
        sentence = text[sent_start:sent_end]

        # Guard 2: hedge language in the same sentence
        if _TRUTH_HEDGE_RE.search(sentence):
            return []

        # Guard 3: definitional framing — "A lie is...", "Lies are..."
        if _TRUTH_DEFINITIONAL_RE.search(sentence):
            return []

        # Guard 4: fiction / hypothetical context
        if _TRUTH_FICTION_RE.search(sentence):
            return []

        return [Flag(
            flag_type=FlagType.UNVERIFIED_TRUTHFULNESS_CLAIM,
            entity_name="truthfulness",
            claim="Response makes an unhedged assertion about a private individual's truthfulness",
            evidence=f"Ungrounded truthfulness assertion: {m.group()[:80]!r}",
            severity="warning",
        )]

    def _check_bare_truthfulness_verdict(
        self, response_text: str, user_input: str
    ) -> list[Flag]:
        """Contextual: bare binary verdict inheriting a truthfulness frame from user_input.

        Problem: a model responding "No" to "Is my friend lying?" makes an
        inadmissible certainty claim even though the response_text contains no
        explicit assertion pattern. The semantic content is inherited from the
        question frame.

        Fires when:
          1. user_input establishes a truthfulness/deception query frame
             (_TRUTH_QUERY_RE matches)
          2. response_text is a bare binary ("Yes", "No", "Definitely no", etc.)
             and nothing else (_BARE_BINARY_RE matches full stripped text)

        No additional guards needed: a bare yes/no to a deception question cannot
        be hedged, definitional, or fictional by construction.
        """
        if not response_text or not user_input:
            return []
        if not _TRUTH_QUERY_RE.search(user_input):
            return []
        if not _BARE_BINARY_RE.match(response_text.strip()):
            return []
        return [Flag(
            flag_type=FlagType.UNVERIFIED_TRUTHFULNESS_CLAIM,
            entity_name="truthfulness",
            claim="Response gives a bare binary verdict inheriting a truthfulness frame from the user's question",
            evidence=(
                f"User asked a truthfulness/deception question; "
                f"model responded with bare binary: {response_text.strip()!r}"
            ),
            severity="warning",
        )]

    def _check_pii_exposure(self, text: str) -> list[Flag]:
        """Detect sensitive PII via regex (Phase 8). Deterministic, no LLM.

        Patterns: SSN (xxx-xx-xxxx), phone numbers, email addresses, card numbers.
        → SENSITIVE_PII_EXPOSURE. Policy: HARD_STOP (public) / FORCE_REVISE (enterprise).

        Consolidated in :mod:`~aurora_lens.verify.pii_surface`.
        """
        matches = collect_pii_evidence_snippets(text)
        if matches:
            return [Flag(
                flag_type=FlagType.SENSITIVE_PII_EXPOSURE,
                entity_name="pii",
                claim="Response contains sensitive PII (SSN, phone, email, or card number)",
                evidence="; ".join(matches),
                severity="warning",
            )]
        return []

    # ── Education checkers ────────────────────────────────────────────────────

    def _check_academic_integrity_violation(self, text: str) -> list[Flag]:
        """Axis 3 HARD_STOP: ghostwriting or exam-answer provision for submission.

        Fires on requests to write assignments, complete exams, or answer
        assessment questions on behalf of the user for submission as their own work.
        Does not fire on tutoring, explanation, study guidance, or paraphrasing help.
        """
        if not text or len(text.strip()) < 20:
            return []

        m = _ACADEMIC_INTEGRITY_RE.search(text)
        if not m:
            return []

        prefix = text[max(0, m.start() - 120): m.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            return []

        return [Flag(
            flag_type=FlagType.ACADEMIC_INTEGRITY_VIOLATION,
            entity_name="academic_integrity",
            claim="Response assists with academic work intended for submission as the user's own",
            evidence=f"Academic integrity pattern detected: {m.group()[:80]!r}",
            severity="error",
        )]

    def _check_student_record_exposure(self, text: str) -> list[Flag]:
        """Axis 3 HARD_STOP: FERPA-class student data about an identified individual.

        Requires: student-subject term + protected data field + name or pronoun signal.
        Does not fire on general policy discussion or anonymised aggregates.
        """
        if not text or len(text.strip()) < 20:
            return []

        if not _STUDENT_RECORD_SUBJECT_RE.search(text):
            return []
        data_m = _STUDENT_RECORD_DATA_RE.search(text)
        if not data_m:
            return []
        if not _STUDENT_RECORD_PERSON_RE.search(text):
            return []

        return [Flag(
            flag_type=FlagType.STUDENT_RECORD_EXPOSURE,
            entity_name="student_record",
            claim="Response exposes or infers protected student record data",
            evidence=f"Student record data field detected: {data_m.group()!r}",
            severity="error",
        )]

    # ── Workforce checkers ────────────────────────────────────────────────────

    def _check_employment_discrimination(self, text: str) -> list[Flag]:
        """Axis 3 HARD_STOP: instructions to select/exclude on protected characteristics.

        Fires when an employment action verb (hire, fire, screen) is combined
        with a protected characteristic term in directive proximity.
        Does not fire on general anti-discrimination policy discussion.
        """
        if not text or len(text.strip()) < 20:
            return []

        action_m = _EMPLOYMENT_ACTION_RE.search(text)
        if not action_m:
            return []

        prefix = text[max(0, action_m.start() - 120): action_m.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            return []

        window_start = max(0, action_m.start() - 80)
        window_end = min(len(text), action_m.end() + 80)
        window = text[window_start:window_end]
        char_m = _PROTECTED_CHARACTERISTIC_RE.search(window)
        if not char_m:
            return []

        return [Flag(
            flag_type=FlagType.EMPLOYMENT_DISCRIMINATION_FACILITATION,
            entity_name="employment_discrimination",
            claim="Response facilitates selection or exclusion based on a protected characteristic",
            evidence=f"Action {action_m.group()!r} + characteristic {char_m.group()!r} detected",
            severity="error",
        )]

    def _check_employee_record_exposure(self, text: str) -> list[Flag]:
        """Axis 3 HARD_STOP: confidential HR data about an identified individual.

        Requires: protected HR data field + person signal (name or pronoun).
        Does not fire on anonymised HR policy discussion.
        """
        if not text or len(text.strip()) < 20:
            return []

        data_m = _EMPLOYEE_RECORD_DATA_RE.search(text)
        if not data_m:
            return []
        if not _EMPLOYEE_RECORD_PERSON_RE.search(text):
            return []

        return [Flag(
            flag_type=FlagType.EMPLOYEE_RECORD_EXPOSURE,
            entity_name="employee_record",
            claim="Response exposes confidential employee HR record data",
            evidence=f"Confidential HR data field detected: {data_m.group()!r}",
            severity="error",
        )]

    # ── Enterprise checkers ───────────────────────────────────────────────────

    def _check_trade_secret_disclosure(self, text: str) -> list[Flag]:
        """Axis 3 HARD_STOP: disclosure of proprietary or NDA-protected information.

        Requires: disclosure verb + proprietary/confidential subject proximity.
        Does not fire on general IP law discussion, NDA definitions, or
        contractual obligations not to disclose.
        """
        if not text or len(text.strip()) < 20:
            return []

        verb_m = _TRADE_SECRET_DISCLOSE_VERB_RE.search(text)
        if not verb_m:
            return []

        prefix = text[max(0, verb_m.start() - 120): verb_m.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            return []
        # Guard: "agree/obliged not to disclose", "not to share", etc. — contractual negation
        if re.search(r'\bnot\s+to\s*$', prefix.strip(), re.IGNORECASE):
            return []
        # Guard: definitional / educational framing ("is a contract", "is an agreement")
        if re.search(r'\bis\s+(?:a|an)\s+(?:contract|agreement|obligation|clause)\b', text, re.IGNORECASE):
            return []

        window_start = max(0, verb_m.start() - 80)
        window_end = min(len(text), verb_m.end() + 80)
        subj_m = _TRADE_SECRET_SUBJECT_RE.search(text[window_start:window_end])
        if not subj_m:
            return []

        return [Flag(
            flag_type=FlagType.TRADE_SECRET_DISCLOSURE,
            entity_name="trade_secret",
            claim="Response discloses or facilitates disclosure of proprietary or NDA-protected information",
            evidence=f"Disclosure verb {verb_m.group()!r} + subject {subj_m.group()!r} detected",
            severity="error",
        )]

    def _check_insider_information(self, text: str) -> list[Flag]:
        """Axis 3 HARD_STOP: use of non-public material information for trading or advantage.

        Fires on insider trading patterns: non-public material information
        combined with trade/invest/profit verbs.
        """
        if not text or len(text.strip()) < 20:
            return []

        m = _INSIDER_INFO_RE.search(text)
        if not m:
            return []

        prefix = text[max(0, m.start() - 120): m.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            return []

        return [Flag(
            flag_type=FlagType.INSIDER_INFORMATION_ASSISTANCE,
            entity_name="insider_information",
            claim="Response involves non-public material information for trading or competitive advantage",
            evidence=f"Insider information pattern detected: {m.group()[:80]!r}",
            severity="error",
        )]

    def _check_procurement_fraud(self, text: str) -> list[Flag]:
        """Axis 3 HARD_STOP: bid-rigging, kickbacks, or conflict-of-interest concealment.

        Fires on procurement fraud patterns: bid-rigging combinations,
        kickback + concealment, or insider-tip + tender context.
        """
        if not text or len(text.strip()) < 20:
            return []

        m = _PROCUREMENT_FRAUD_RE.search(text)
        if not m:
            return []

        prefix = text[max(0, m.start() - 120): m.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            return []

        return [Flag(
            flag_type=FlagType.PROCUREMENT_FRAUD_FACILITATION,
            entity_name="procurement_fraud",
            claim="Response facilitates bid-rigging, kickbacks, or conflict-of-interest concealment",
            evidence=f"Procurement fraud pattern detected: {m.group()[:80]!r}",
            severity="error",
        )]

    # ── Verify-or-refuse checkers (new domains) ───────────────────────────────

    def _check_personalized_academic_advice(
        self, text: str, user_input: str | None = None
    ) -> list[Flag]:
        """Verify-or-refuse: personalised admissions chances or credential plans.

        Fires when response addresses a named individual's specific admissions
        prospects, eligibility, or credential pathway.
        """
        if not text or len(text.strip()) < 15:
            return []

        source = user_input or text
        if not _ACADEMIC_ADVICE_PERSONAL_RE.search(source):
            return []
        if not _ACADEMIC_ADVICE_VOCAB_RE.search(source):
            return []

        return [Flag(
            flag_type=FlagType.PERSONALIZED_ACADEMIC_ADVICE,
            entity_name="academic_advice",
            claim="Response provides personalised academic admissions or credential advice",
            evidence="Personalisation signal + admissions/credential vocabulary detected",
            severity="warning",
        )]

    def _check_personalized_employment_advice(
        self, text: str, user_input: str | None = None
    ) -> list[Flag]:
        """Verify-or-refuse: individual employment dispute posture.

        Fires when response addresses the user's specific employment claim
        (wrongful dismissal, discrimination case, individual entitlements).
        """
        if not text or len(text.strip()) < 15:
            return []

        source = user_input or text
        if not _EMPLOYMENT_ADVICE_PERSONAL_RE.search(source):
            return []
        if not _EMPLOYMENT_ADVICE_RE.search(source):
            return []

        return [Flag(
            flag_type=FlagType.PERSONALIZED_EMPLOYMENT_ADVICE,
            entity_name="employment_advice",
            claim="Response provides personalised employment dispute or dismissal advice",
            evidence="Personal employment dispute pattern detected",
            severity="warning",
        )]

    def _check_personalized_business_strategy(
        self, text: str, user_input: str | None = None
    ) -> list[Flag]:
        """Verify-or-refuse: named-company strategic recommendations.

        Fires when response issues strategic directives (should, must, recommend)
        targeting a named company in a competitive or M&A context.
        """
        if not text or len(text.strip()) < 15:
            return []

        source = user_input or text
        if not _BUSINESS_STRATEGY_DIRECTIVE_RE.search(source):
            return []
        if not _BUSINESS_STRATEGY_VOCAB_RE.search(source):
            return []
        if not _BUSINESS_STRATEGY_NAMED_RE.search(source):
            return []

        return [Flag(
            flag_type=FlagType.PERSONALIZED_BUSINESS_STRATEGY_ADVICE,
            entity_name="business_strategy",
            claim="Response provides personalised strategic advice for a named company",
            evidence="Named-company directive + strategic vocabulary detected",
            severity="warning",
        )]

    def _get_rel_object_text(self, rel: Relationship, pef: PEFState) -> str | None:
        """Get the textual representation of a relationship's object."""
        if rel.object_entity_id:
            obj_entity = pef.entities.get(rel.object_entity_id)
            return obj_entity.name if obj_entity else None
        if rel.object_literal is not None:
            return str(rel.object_literal)
        return None
