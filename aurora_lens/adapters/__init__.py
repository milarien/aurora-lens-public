"""LLM adapters — model-agnostic interface."""

from .base import LLMAdapter
from .openai import OpenAIUpstreamAdapter

__all__ = ["LLMAdapter", "OpenAIUpstreamAdapter"]
