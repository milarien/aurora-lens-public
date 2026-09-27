"""LensConfig — configuration for the aurora-lens pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aurora_lens.adapters.base import LLMAdapter

if TYPE_CHECKING:
    from aurora_lens.interpret.base import ExtractionBackend
    from aurora_lens.govern.bridge import GovernanceBridge
    from aurora_lens.sovereign.provider_registry import SovereignProviderRegistry
    from aurora_lens.trust.trust_registry import TrustRegistry

from aurora_lens.state_native_engine.protocol import StateNativeEngine


@dataclass
class LensConfig:
    """Configuration for the Lens pipeline."""

    adapter: LLMAdapter                 # Required: the LLM adapter to use

    # Extraction backend (defaults to SpacyBackend if None)
    extraction_backend: ExtractionBackend | None = None

    # Governance bridge (defaults to BuiltinBridge with DEFAULT_STRICT)
    governance_bridge: GovernanceBridge | None = None

    # Path for governance audit log (JSONL)
    audit_log_path: str | None = None

    # Whether to auto-update PEF state from user input
    auto_interpret: bool = True

    # Whether to verify LLM responses (post-LLM checker). Request-side
    # :meth:`~aurora_lens.verify.checker.Checker.check_blocked_act_request` in
    # :class:`~aurora_lens.lens.Lens` runs regardless of this flag.
    auto_verify: bool = True

    # Whether to include PEF context in LLM system prompt
    inject_pef_context: bool = True

    # Maximum conversation history turns to send to LLM
    max_history_turns: int = 10

    # spaCy model name (when extraction_backend is SpacyBackend default)
    spacy_model: str = "en_core_web_sm"

    # Stream accumulator cap in bytes (0 = no cap)
    max_stream_bytes: int = 512 * 1024  # 512 KiB

    # Two-plane output: when False (default), flags/rationale/governance_note stay in audit
    # log only and are never returned in the client response body. Set True only for
    # operator tooling or internal debugging.
    include_operator_detail: bool = False

    # Whether process_stream() emits safe ProgressSignal events ("progress" kind)
    # during the buffer→verify→release cycle.  Off by default.
    # Progress events contain no content, no claim, and no governance metadata —
    # only a lifecycle phase label ("streaming" | "verifying" | "releasing").
    stream_emit_progress: bool = False

    # Programmatic override: when True, forces the retrieval-aware referent path on (unless
    # env ``AURORA_LENS_RAG_RETRIEVAL_AWARE_REFERENTS`` is force-off). The proxy normally
    # leaves this False and relies on :mod:`aurora_lens.rag_activation` (request_metadata
    # and/or ``Context:`` … ``Question:`` shape, plus env force-on/off).
    rag_retrieval_aware_referents: bool = False

    # When True, :meth:`Lens.process` may delegate strict location ``QUERY`` turns
    # to ``state_native_engine`` before the LLM. If ``state_native_engine`` is None
    # but this flag is True, :class:`~aurora_lens.state_native_engine.default_engine.DefaultStateNativeEngine`
    # is used. Streaming path is unchanged in v1.
    enable_state_native_delegation: bool = False
    state_native_engine: StateNativeEngine | None = None

    #: Optional Sovereign Provider Registry for provider-route admissibility.
    #: When ``request_metadata.provider_route`` is present, Lens requires this registry
    #: to evaluate failover and declare bridge metadata before adapter execution.
    sovereign_provider_registry: SovereignProviderRegistry | None = None

    #: When True with a configured registry, requests missing ``provider_route`` metadata
    #: are blocked before adapter execution (no silent sovereign bypass).
    sovereign_enforce_provider_route: bool = False

    #: Optional Trust Registry for Track C ``source_untrusted`` generation.
    #: When a relationship carries a complete trust contract with
    #: ``trust_evaluation_required``, Lens evaluates declared source profiles
    #: and may emit ``source_untrusted`` deterministically.
    trust_registry: TrustRegistry | None = None

    #: When True, state-native counted possession transfers mutate only from **HIGH**
    #: confidence frames (spaCy-tight parses); LOW regex adapters see builds but apply path abstains.
    require_high_confidence_possession_transfer: bool = False

