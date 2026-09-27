"""Corpus plane — file-based record registry and chunk storage.

Lens-independent: no imports from ``aurora_lens.lens``, PEF, proxy, or RAG admission.

Product Phase 3 entrypoint: ``propose_document_ingestion`` (ingestion proposes; Lens disposes).
"""

from aurora_lens.corpus.document_ingestion import (
    ingest_file,
    propose_document_ingestion,
)
from aurora_lens.corpus.establishment_failure_proposal import EstablishmentFailureProposal
from aurora_lens.corpus.ingestion_report import (
    CandidateRecord,
    IngestionReport,
)
from aurora_lens.corpus.ingest_metadata import IngestMetadata
from aurora_lens.corpus.intake_translator import (
    preview_gate_evaluation,
    translate_candidate_record,
    translate_plain_english_intake,
    translate_structured_intake,
)
from aurora_lens.corpus.models import CorpusChunk, CorpusRecord, make_chunk_id
from aurora_lens.corpus.registry import CorpusRegistry, default_corpus_root
from aurora_lens.corpus.retrieve import (
    assemble_rag_message,
    assemble_request_metadata,
    retrieve_chunks,
)

__all__ = [
    "CandidateRecord",
    "CorpusChunk",
    "CorpusRecord",
    "CorpusRegistry",
    "EstablishmentFailureProposal",
    "IngestionReport",
    "IngestMetadata",
    "assemble_rag_message",
    "assemble_request_metadata",
    "default_corpus_root",
    "ingest_file",
    "make_chunk_id",
    "preview_gate_evaluation",
    "propose_document_ingestion",
    "retrieve_chunks",
    "translate_candidate_record",
    "translate_plain_english_intake",
    "translate_structured_intake",
]
