"""Declared provider profile schema — no inference, declare only."""

from __future__ import annotations

from dataclasses import dataclass

from aurora_lens.sovereign.provider_state import ProviderState, VALIDATED_CURRENT


@dataclass(frozen=True)
class ProviderProfile:
    """Minimum declared profile for a provider/model endpoint."""

    provider_id: str
    provider_name: str
    model_id: str
    endpoint_url: str
    hosting_jurisdiction: str
    data_boundary: str
    allowed_regions: tuple[str, ...] = ()
    allowed_data_classes: tuple[str, ...] = ("public", "internal", "restricted")
    permitted_domains: tuple[str, ...] = ("general",)
    max_consequence_grade: str = "low"
    supports_tools: bool = False
    supports_structured_output: bool = False
    context_window_tokens: int = 8192
    retention_policy: str = "provider_contract_required"
    validation_suite_id: str = "provider_regression_v1"
    last_validated_at: str = ""
    status: ProviderState = VALIDATED_CURRENT
    regression_result: str = ""
    report_hash: str | None = None
    failed_checks: tuple[str, ...] = ()
    observed_model_id: str | None = None

    def with_status(self, status: ProviderState) -> ProviderProfile:
        return ProviderProfile(
            provider_id=self.provider_id,
            provider_name=self.provider_name,
            model_id=self.model_id,
            endpoint_url=self.endpoint_url,
            hosting_jurisdiction=self.hosting_jurisdiction,
            data_boundary=self.data_boundary,
            allowed_regions=self.allowed_regions,
            allowed_data_classes=self.allowed_data_classes,
            permitted_domains=self.permitted_domains,
            max_consequence_grade=self.max_consequence_grade,
            supports_tools=self.supports_tools,
            supports_structured_output=self.supports_structured_output,
            context_window_tokens=self.context_window_tokens,
            retention_policy=self.retention_policy,
            validation_suite_id=self.validation_suite_id,
            last_validated_at=self.last_validated_at,
            status=status,
            regression_result=self.regression_result,
            report_hash=self.report_hash,
            failed_checks=self.failed_checks,
            observed_model_id=self.observed_model_id,
        )


def profiles_by_id(profiles: list[ProviderProfile]) -> dict[str, ProviderProfile]:
    return {p.provider_id: p for p in profiles}
