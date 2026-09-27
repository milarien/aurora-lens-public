import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from .models import (
    Domain,
    AuthorityClass,
    LensStatus,
    UserClass,
    GovernorPolicy,
    ContinuationPathway,
    SpeechAct,
    ProceduralAction,
    ForensicObligation,
    DisclosureType,
    OutputMode,
    ExposureLevel,
    ResolutionMode,
    ContinuationCapability,
)
from .continuation_matrix import CONTINUATION_MATRIX

_BUNDLED_MATRIX_PATH = Path(__file__).parent / "policy_matrix.json"


class PolicyResolver:
    """Deterministic continuation policy resolver.

    Resolution signature:
        resolve(domain, authority, status, user_class, policy_version)

    Fallback ladder (explicit, auditable):
        1. domain:authority:status:user_class  (EXACT)
        2. domain:authority:status             (AUTHORITY_FALLBACK when user_class != GENERAL)
        3. domain:GP:status                    (AUTHORITY_FALLBACK when authority != GP)
        4. general:GP:status                   (DOMAIN_FALLBACK)
        5. general:GP:STOP                     (GLOBAL_SAFE_FALLBACK)

    Policy source is recorded on every resolved policy via resolution_mode and
    is available for audit via active_version / policy_source properties.

    Args:
        matrix_path:    Absolute path to a policy matrix JSON file.
                        If None, the bundled matrix is used.
        policy_version: Logical version label (e.g. '1.0', '2024-Q2').
                        When matrix_path is None, the resolver first looks for
                        policy_matrix.<version>.json next to the bundled file.
                        If that versioned file does not exist, it falls back to
                        the bundled policy_matrix.json.
    """

    def __init__(
        self,
        matrix_path: Optional[Path] = None,
        policy_version: str = "1.0",
    ):
        self._requested_version = policy_version

        if matrix_path is not None:
            resolved_path = Path(matrix_path)
            self._policy_source = "operator"
        else:
            versioned = _BUNDLED_MATRIX_PATH.parent / f"policy_matrix.{policy_version}.json"
            if policy_version != "1.0" and versioned.exists():
                resolved_path = versioned
                self._policy_source = "bundled"
            else:
                resolved_path = _BUNDLED_MATRIX_PATH
                self._policy_source = "bundled"

        with open(resolved_path, "r", encoding="utf-8-sig") as f:
            self._matrix = json.load(f)

        self._active_version: str = self._matrix.get("_version", policy_version)

        if self._policy_source == "operator":
            self._validate_operator_matrix()

    @property
    def active_version(self) -> str:
        """The policy version actually loaded."""
        return self._active_version

    @property
    def policy_source(self) -> str:
        """'bundled' or 'operator' -- where the matrix was loaded from."""
        return self._policy_source

    def _validate_operator_matrix(self) -> None:
        """Validate all operator matrix entries against the override constitution.

        Called at load time when policy_source == 'operator'.  Raises
        OperatorOverrideError on the first key with constitutional violations,
        so malformed operator matrices are rejected before any policy is resolved.
        """
        from .operator_override import validate_override, OperatorOverrideError

        for key, op in self._matrix.items():
            if key.startswith("_") or not isinstance(op, dict):
                continue
            parts = key.split(":")
            if len(parts) < 3:
                continue
            try:
                domain = Domain(parts[0])
                authority = AuthorityClass(parts[1])
                status = LensStatus(parts[2])
                discriminator: Optional[str] = parts[3] if len(parts) > 3 else None
            except ValueError:
                # Unknown domain/authority/status — no canonical row to validate against.
                continue
            try:
                canonical_row = CONTINUATION_MATRIX.lookup(domain, authority, status, discriminator)
            except (ValueError, KeyError):
                continue
            violations = validate_override(canonical_row, op)
            if violations:
                raise OperatorOverrideError(key, violations)

    def resolve(
        self,
        domain: Domain,
        authority: AuthorityClass,
        status: LensStatus,
        user_class: UserClass = UserClass.GENERAL,
        reason_code: Optional[str] = None,
        policy_version: str = "1.0",
    ) -> GovernorPolicy:
        """
        Deterministic resolution of the continuation policy with explicit fallback tracking.
        Pathway = f(lens_status, domain, authority_class, user_type, policy_version)
        """

        # 0. Attempt Flag-Class Match (Domain + Authority + Status + reason_code)
        #    reason_code carries the primary flag type name (e.g. "SELF_HARM_INSTRUCTION").
        #    This level lets the policy matrix split behaviour for flags that share the
        #    same domain:authority:status corridor — e.g. medical:GP:STOP routes
        #    EMERGENCY_TRIAGE_GUIDANCE and SELF_HARM_INSTRUCTION to different pathways.
        if reason_code:
            key_flag = f"{domain.value}:{authority.value}:{status.value}:{reason_code}"
            if key_flag in self._matrix:
                return self._build_policy(key_flag, domain, authority, status, user_class, ResolutionMode.EXACT)

        # 1. Attempt Exact Match (Domain + Authority + Status + UserClass)
        key_exact = f"{domain.value}:{authority.value}:{status.value}:{user_class.value}"
        if key_exact in self._matrix:
            return self._build_policy(key_exact, domain, authority, status, user_class, ResolutionMode.EXACT)

        # 2. Attempt Authority Match (Domain + Authority + Status)
        key_auth = f"{domain.value}:{authority.value}:{status.value}"
        if key_auth in self._matrix:
            mode = ResolutionMode.AUTHORITY_FALLBACK if user_class != UserClass.GENERAL else ResolutionMode.EXACT
            return self._build_policy(key_auth, domain, authority, status, user_class, mode)

        # 3. Attempt Domain-GP Match (Domain + GP + Status)
        key_domain_gp = f"{domain.value}:GP:{status.value}"
        if key_domain_gp in self._matrix:
            return self._build_policy(key_domain_gp, domain, authority, status, user_class, ResolutionMode.AUTHORITY_FALLBACK)

        # 4. Attempt Global Status Match (General + GP + Status)
        key_general = f"general:GP:{status.value}"
        if key_general in self._matrix:
            return self._build_policy(key_general, domain, authority, status, user_class, ResolutionMode.DOMAIN_FALLBACK)

        # 5. Global Safe Fallback
        return self._build_policy("general:GP:STOP", domain, authority, status, user_class, ResolutionMode.GLOBAL_SAFE_FALLBACK)

    def _build_policy(
        self,
        key: str,
        domain: Domain,
        authority: AuthorityClass,
        status: LensStatus,
        user_class: UserClass,
        mode: ResolutionMode
    ) -> GovernorPolicy:
        data = self._matrix.get(key, self._matrix.get("general:GP:STOP", {}))

        # Continuation parameters (pathway_id, commitment_closed, interaction_open,
        # output_mode, forensic_obligations, escalation_target) are sourced from the
        # canonical ContinuationMatrix — the single authoritative table for lawful
        # continuation routing.  Speech-act permissions, procedural actions, required
        # disclosures, and exposure level remain in the JSON policy matrix.
        #
        # The discriminator is the 4th key component when present (a flag-type name
        # or user-class value).  ContinuationMatrix.lookup() applies the same
        # four-level fallback ladder as the resolver, keyed on this discriminator.
        parts = key.split(":")
        discriminator: Optional[str] = parts[3] if len(parts) > 3 else None
        row = CONTINUATION_MATRIX.lookup(domain, authority, status, discriminator)

        # Operator-supplied matrices may override individual continuation fields
        # by including them explicitly in their JSON.  This allows operators to
        # tighten policy (e.g. close commitment for ADMIT in regulated contexts)
        # without rewriting the full canonical matrix.  When no operator matrix
        # is loaded (bundled path), CONTINUATION_MATRIX is the sole authority.
        if self._policy_source == "operator" and key in self._matrix:
            op = self._matrix[key]
            commitment_closed = op.get("commitment_closed", row.commitment_closed)
            interaction_open  = op.get("interaction_open",  row.interaction_open)
            try:
                pathway_id = ContinuationPathway(op["pathway_id"]) if "pathway_id" in op else row.pathway_id
            except ValueError:
                pathway_id = row.pathway_id
            try:
                output_mode = OutputMode(op["output_mode"]) if "output_mode" in op else row.output_mode
            except ValueError:
                output_mode = row.output_mode
            escalation_target = op.get("escalation_target", row.escalation_target)
            raw_fo = op.get("forensic_obligations")
            if raw_fo is not None:
                forensic_obligations = [ForensicObligation(fo) for fo in raw_fo]
            else:
                forensic_obligations = list(row.forensic_obligations)
        else:
            commitment_closed    = row.commitment_closed
            interaction_open     = row.interaction_open
            pathway_id           = row.pathway_id
            output_mode          = row.output_mode
            escalation_target    = row.escalation_target
            forensic_obligations = list(row.forensic_obligations)

        allowed_speech_acts = [SpeechAct(sa) for sa in data.get("allowed_speech_acts", [])]
        allowed_procedural_actions = [
            ProceduralAction(pa) for pa in data.get("allowed_procedural_actions", [])
        ]
        allowed_continuations = [
            ContinuationCapability(c) for c in data.get("allowed_continuations", [])
        ]
        forbidden_speech_acts = [SpeechAct(sa) for sa in data.get("forbidden_speech_acts", [])]
        required_disclosures = [DisclosureType(rd) for rd in data.get("required_disclosures", [])]
        exposure_level = ExposureLevel(data.get("exposure_level", "minimal"))

        # PR6 boundary: AUDITOR routing is explicit-only. Forensic-visibility widening
        # is lawful only when policy resolved through a dedicated :auditor corridor key.
        # All other auditor requests are hard-capped to general visibility.
        if user_class == UserClass.AUDITOR:
            explicit_auditor_corridor = key.endswith(":auditor")
            if explicit_auditor_corridor and status != LensStatus.STOP:
                raise ValueError(
                    "AUDITOR override corridors are only permitted for STOP status."
                )
            if explicit_auditor_corridor:
                if SpeechAct.EXPOSE_AUDIT_BASIS not in allowed_speech_acts:
                    allowed_speech_acts.append(SpeechAct.EXPOSE_AUDIT_BASIS)
                if ForensicObligation.ATTACH_PEF_SNAPSHOT not in forensic_obligations:
                    forensic_obligations.append(ForensicObligation.ATTACH_PEF_SNAPSHOT)
                output_mode = OutputMode.FORENSIC_STOP
                exposure_level = ExposureLevel.FULL
            else:
                allowed_speech_acts = [
                    act for act in allowed_speech_acts if act != SpeechAct.EXPOSE_AUDIT_BASIS
                ]
                forensic_obligations = [
                    fo
                    for fo in forensic_obligations
                    if fo != ForensicObligation.ATTACH_PEF_SNAPSHOT
                ]
                if output_mode == OutputMode.FORENSIC_STOP:
                    output_mode = OutputMode.TERMINAL_STOP
                exposure_level = ExposureLevel.MINIMAL

        return GovernorPolicy(
            domain=domain,
            authority_class=authority,
            lens_status=status,
            user_class=user_class,
            commitment_closed=commitment_closed,
            interaction_open=interaction_open,
            pathway_id=pathway_id,
            output_mode=output_mode,
            escalation_target=escalation_target,
            forensic_obligations=forensic_obligations,
            allowed_speech_acts=allowed_speech_acts,
            allowed_procedural_actions=allowed_procedural_actions,
            allowed_continuations=allowed_continuations,
            forbidden_speech_acts=forbidden_speech_acts,
            required_disclosures=required_disclosures,
            exposure_level=exposure_level,
            resolution_mode=mode
        )


# Module-level singleton -- invalidated if called with non-default args.
_resolver: Optional[PolicyResolver] = None


def get_resolver() -> PolicyResolver:
    global _resolver
    if _resolver is None:
        _resolver = PolicyResolver()
    return _resolver


def resolve(
    domain: Domain,
    authority: AuthorityClass,
    status: LensStatus,
    user_class: UserClass = UserClass.GENERAL,
    reason_code: Optional[str] = None,
    policy_version: str = "1.0"
) -> GovernorPolicy:
    return get_resolver().resolve(domain, authority, status, user_class, reason_code, policy_version)
