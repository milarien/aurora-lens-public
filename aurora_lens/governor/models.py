from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

class LensStatus(Enum):
    ADMIT = "ADMIT"
    ASK = "ASK"
    REFUSE = "REFUSE"
    STOP = "STOP"

class Domain(Enum):
    GENERAL = "general"
    AMBIGUITY = "ambiguity"
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCE = "finance"
    RESEARCH = "research"
    CRISIS = "crisis"      # Psychological safety / self-harm; P_STOP_TERMINAL + interaction_open=True
    EDUCATION = "education"
    WORKFORCE = "workforce"
    ENTERPRISE = "enterprise"
    # Hostile / illicit operational content (hazard ontology, illegal instruction).
    # Distinct from LEGAL (personalized legal advice / compliance corridors).
    HARMFUL = "harmful"

class AuthorityClass(Enum):
    GP = "GP"  # General Purpose
    DA = "DA"  # Domain Authorized
    HS = "HS"  # Human Supervised

class UserClass(Enum):
    GENERAL = "general"
    CLINICIAN = "clinician"
    AUDITOR = "auditor"
    LEGAL_PRO = "legal_pro"
    FINANCIAL_PRO = "financial_pro"
    RESEARCHER = "researcher"
    INTEGRATOR = "integrator"
    EDUCATOR = "educator"
    HR_PRO = "hr_pro"
    COMPLIANCE_OFF = "compliance_off"

class SpeechAct(Enum):
    # Pure linguistic moves
    EXPLAIN_BOUNDARY = "explain_boundary"
    STATE_INSUFFICIENCY = "state_insufficiency"
    REQUEST_DISAMBIGUATION = "request_disambiguation"
    REQUEST_MISSING_FACT = "request_missing_fact"
    OFFER_SAFE_NEXT_STEP = "offer_safe_next_step"
    OFFER_HANDOFF_SUMMARY = "offer_handoff_summary"
    NOTICE_ESCALATION = "notice_escalation"
    NOTICE_TERMINATION = "notice_termination"
    EXPOSE_AUDIT_BASIS = "expose_audit_basis"
    
    # Substantive/Admitted acts
    PROVIDE_SUBSTANTIVE_ANSWER = "provide_substantive_answer"
    PROVIDE_GENERAL_INFORMATION = "provide_general_information"
    
    # Forbidden acts (for invariants)
    INFER_ANSWER = "infer_answer"
    SUGGEST_PROBABLE_RESOLUTION = "suggest_probable_resolution"
    SOFTEN_INTO_ADVICE = "soften_into_advice"
    NARRATIVE_COMPLETION = "narrative_completion"
    COMPARATIVE_RANKING = "comparative_ranking"

class ProceduralAction(Enum):
    ESCALATE_TO_HUMAN = "escalate_to_human"
    GENERATE_SUMMARY = "generate_summary"
    NOTIFY_OPERATOR = "notify_operator"

class ForensicObligation(Enum):
    EMIT_FORENSIC_ENVELOPE = "emit_forensic_envelope"
    ATTACH_PEF_SNAPSHOT = "attach_pef_snapshot"

class DisclosureType(Enum):
    INFORMATIONAL_ONLY = "informational_only"
    NOT_LEGAL_ADVICE = "not_legal_advice"
    NOT_FINANCIAL_ADVICE = "not_financial_advice"
    NOT_EDUCATION_ADVICE = "not_education_advice"
    NOT_EMPLOYMENT_ADVICE = "not_employment_advice"
    NOT_BUSINESS_ADVICE = "not_business_advice"
    NONE = "none"

class OutputMode(Enum):
    FULL_RESPONSE = "full_response"
    CLARIFICATION_REQUEST = "clarification_request"
    REFUSAL_WITH_EXPLANATION = "refusal_with_explanation"
    REFUSAL_WITH_CLARIFICATION = "refusal_with_clarification"
    CONSTRAINED_RESPONSE = "constrained_response"
    PRO_REFUSAL = "pro_refusal"
    TERMINAL_STOP = "terminal_stop"
    FORENSIC_STOP = "forensic_stop"
    DEFAULT_REFUSAL = "default_refusal"

class ExposureLevel(Enum):
    MINIMAL = "minimal"
    MODERATE = "moderate"
    FULL = "full"

class ResolutionMode(Enum):
    EXACT = "exact"
    AUTHORITY_FALLBACK = "authority_fallback"
    DOMAIN_FALLBACK = "domain_fallback"
    GLOBAL_SAFE_FALLBACK = "global_safe_fallback"

class ContinuationCapability(Enum):
    NEUTRAL_TIMELINE = "neutral_timeline"
    MEDICAL_POST_REFUSAL_SAFE_FOLLOWUP = "medical_post_refusal_safe_followup"
    NEUTRAL_SUMMARY = "neutral_summary"
    LIST_DOCUMENTS = "list_documents"

class ContinuationPathway(Enum):
    P_ADMIT_STANDARD = "P_ADMIT_STANDARD"
    P_ASK_DISAMBIGUATE = "P_ASK_DISAMBIGUATE"
    P_ASK_MISSING_FACT = "P_ASK_MISSING_FACT"
    P_REFUSE_EXPLAIN_REDIRECT = "P_REFUSE_EXPLAIN_REDIRECT"
    P_REFUSE_ESCALATE_PRO = "P_REFUSE_ESCALATE_PRO"
    P_STOP_TERMINAL = "P_STOP_TERMINAL"
    P_STOP_FORENSIC = "P_STOP_FORENSIC"
    P_STOP_ESCALATE = "P_STOP_ESCALATE"
    P_STOP_ESCALATE_EMERGENCY = "P_STOP_ESCALATE_EMERGENCY"       # Physical medical emergency → immediate escalation
    P_STOP_SUPPORTIVE_DEESCALATE = "P_STOP_SUPPORTIVE_DEESCALATE" # Psychological crisis → supportive de-escalation
    P_STOP_REDIRECT_QUALIFIED = "P_STOP_REDIRECT_QUALIFIED"       # Refusal with lawful professional/resource redirect
    P_STOP_REFUSE_CLEAN = "P_STOP_REFUSE_CLEAN"                   # Refusal without redirect or workaround adjacency
    P_HANDOFF_SUMMARY = "P_HANDOFF_SUMMARY"

@dataclass(frozen=True)
class GovernorPolicy:
    domain: Domain
    authority_class: AuthorityClass
    lens_status: LensStatus
    user_class: UserClass = UserClass.GENERAL
    
    commitment_closed: bool = True
    interaction_open: bool = True
    
    allowed_speech_acts: List[SpeechAct] = field(default_factory=list)
    allowed_procedural_actions: List[ProceduralAction] = field(default_factory=list)
    allowed_continuations: List[ContinuationCapability] = field(default_factory=list)
    forbidden_speech_acts: List[SpeechAct] = field(default_factory=list)
    forensic_obligations: List[ForensicObligation] = field(default_factory=list)
    
    required_disclosures: List[DisclosureType] = field(default_factory=list)
    
    escalation_target: Optional[str] = None
    pathway_id: ContinuationPathway = ContinuationPathway.P_STOP_TERMINAL
    output_mode: OutputMode = OutputMode.DEFAULT_REFUSAL
    exposure_level: ExposureLevel = ExposureLevel.MINIMAL
    
    resolution_mode: ResolutionMode = ResolutionMode.EXACT

    def __post_init__(self):
        # CENTRAL INVARIANT:
        # Governor may constrain, explain, summarize, route, or escalate after Lens, 
        # but it may never widen epistemic commitment or recover a blocked determination by implication.
        
        # Invariant 1: No Epistemic Widening
        if self.lens_status in (LensStatus.REFUSE, LensStatus.STOP, LensStatus.ASK):
            if not self.commitment_closed:
                raise ValueError(
                    f"Epistemic widening violation: commitment_closed must be True when Lens status is {self.lens_status.value}. "
                    "Governor cannot recover a blocked determination."
                )

        # Invariant 2: Formalized Anti-Cheat
        if self.commitment_closed:
            substantive_acts = {
                SpeechAct.PROVIDE_SUBSTANTIVE_ANSWER,
                SpeechAct.INFER_ANSWER,
                SpeechAct.SUGGEST_PROBABLE_RESOLUTION,
                SpeechAct.SOFTEN_INTO_ADVICE,
                SpeechAct.NARRATIVE_COMPLETION,
                SpeechAct.COMPARATIVE_RANKING
            }
            for act in self.allowed_speech_acts:
                if act in substantive_acts:
                    raise ValueError(
                        f"Policy violation: {act.value} allowed while commitment is closed. "
                        "Governor must not assert, imply, recommend, rank, compare, or narratively complete prohibited determinations."
                    )

        # Invariant 3: Auditor corridor hardening is enforced in resolver.py (PR6).
        # PolicyProjection validation here does not widen forensic visibility.
        if self.user_class == UserClass.AUDITOR:
            pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "authority_class": self.authority_class.value,
            "lens_status": self.lens_status.value,
            "user_class": self.user_class.value,
            "commitment_closed": self.commitment_closed,
            "interaction_open": self.interaction_open,
            "allowed_speech_acts": [sa.value for sa in self.allowed_speech_acts],
            "allowed_procedural_actions": [pa.value for pa in self.allowed_procedural_actions],
            "allowed_continuations": [c.value for c in self.allowed_continuations],
            "forbidden_speech_acts": [sa.value for sa in self.forbidden_speech_acts],
            "forensic_obligations": [fo.value for fo in self.forensic_obligations],
            "required_disclosures": [rd.value for rd in self.required_disclosures],
            "escalation_target": self.escalation_target,
            "pathway_id": self.pathway_id.value,
            "output_mode": self.output_mode.value,
            "exposure_level": self.exposure_level.value,
            "resolution_mode": self.resolution_mode.value
        }