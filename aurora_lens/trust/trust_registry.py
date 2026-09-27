"""Declared trust/authority registry for source admissibility."""

from __future__ import annotations

from aurora_lens.trust.source_id import normalize_source_id, source_id_key
from aurora_lens.trust.trust_policy import TrustEvaluationResult, evaluate_trust_admissibility
from aurora_lens.trust.trust_profile import TrustSourceProfile


def profiles_by_id(profiles: list[TrustSourceProfile]) -> dict[str, TrustSourceProfile]:
    out: dict[str, TrustSourceProfile] = {}
    for profile in profiles:
        normalized = normalize_source_id(profile.source_id)
        if normalized is None:
            continue
        out[source_id_key(normalized)] = profile
    return out


class TrustRegistry:
    """In-memory registry of declared trust source profiles."""

    def __init__(
        self,
        profiles: list[TrustSourceProfile] | None = None,
        *,
        registry_ref: str = "default",
    ) -> None:
        self._registry_ref = str(registry_ref or "").strip() or "default"
        self._profiles = profiles_by_id(profiles or [])

    @property
    def registry_ref(self) -> str:
        return self._registry_ref

    def register(self, profile: TrustSourceProfile) -> None:
        normalized = normalize_source_id(profile.source_id)
        if normalized is None:
            raise ValueError(f"Invalid source_id for trust profile: {profile.source_id!r}")
        self._profiles[source_id_key(normalized)] = profile

    def get_profile(self, source_id: str) -> TrustSourceProfile | None:
        normalized = normalize_source_id(source_id)
        if normalized is None:
            return None
        return self._profiles.get(source_id_key(normalized))

    def evaluate_source(
        self,
        source_id: str,
        *,
        task_domain: str,
        consequence_grade: str,
        trust_registry_ref: str,
    ) -> TrustEvaluationResult:
        """Evaluate trust for a source when contract registry ref matches this registry."""
        if str(trust_registry_ref or "").strip() != self._registry_ref:
            return TrustEvaluationResult(
                source_id=str(source_id or ""),
                admissible=True,
                registry_evaluated=False,
                reason="registry_ref_mismatch",
            )
        profile = self.get_profile(source_id)
        return evaluate_trust_admissibility(
            profile,
            source_id=source_id,
            task_domain=task_domain,
            consequence_grade=consequence_grade,
        )
