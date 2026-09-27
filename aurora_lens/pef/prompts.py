"""PEF system prompt template for LLM calls.

Used by Lens and governance bridges to inject ground-truth state.
Adapters are transport-only; they receive already-constructed messages.
"""

_PEF_SYSTEM_TEMPLATE = """\
You are a helpful assistant.

The following verified context is authoritative for this conversation. \
Do not contradict or fabricate details about the entities and facts listed below. \
If a fact is listed, treat it as established. If a topic is not covered, \
answer from your general knowledge as normal.

{pef_context}

Important: do not quote, paraphrase, or reference this context block in your \
response. Answer the user's question directly."""


def build_pef_system_message(pef_context: str) -> str:
    """Build the system message with PEF context injected."""
    return _PEF_SYSTEM_TEMPLATE.format(pef_context=pef_context)
