from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple

# --- Core RAG Pipeline Models ---

class SearchPolicy(BaseModel):
    strategy: str = "hybrid"
    k: int = 5
    rerank_depth: int = 20
    min_score_threshold: float = 0.7

class QueryContext(BaseModel):
    original_query: str
    rewritten_query: Optional[str] = None
    chat_history: List[Dict[str, str]] = []
    policy: SearchPolicy = Field(default_factory=SearchPolicy)

class RetrievedDoc(BaseModel):
    doc_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float

class RetrievalResult(BaseModel):
    docs: List[RetrievedDoc]
    query_context: QueryContext

class ScoredRetrievalResult(BaseModel):
    retrieval_result: RetrievalResult
    relevance_score: float
    diversity_score: float
    is_sufficient: bool

class DraftAnswer(BaseModel):
    content: str
    used_doc_ids: List[str]

class Critique(BaseModel):
    has_hallucinations: bool
    unsupported_claims: List[str] = Field(default_factory=list)
    revision_suggestions: str

class Citation(BaseModel):
    doc_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FinalAnswer(BaseModel):
    content: str
    citations: List[Citation] = Field(default_factory=list)
    confidence_score: float = 0.0

# --- API Endpoint Models ---

class SetActiveDocsRequest(BaseModel):
    document_names: List[str]
    provider: str
    embedding_model: str
    google_api_key: Optional[str] = None

class ChatRequest(BaseModel):
    provider: str
    question: str
    chat_model: str
    embedding_model: str
    router_model: Optional[str] = None
    system_prompt: Optional[str] = None
    google_api_key: Optional[str] = None
    retrieval_k: Optional[int] = 5
    chat_history: List[Tuple[str, str]] = []
    mode: str = "auto"
    include_citations: bool = True

class DocumentSource(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    answer: str
    source_documents: List[DocumentSource]
    citations: List[str] = []
    used_search_queries: List[str] = []

class OllamaModel(BaseModel):
    name: str
    modified_at: str
    size: int
    digest: str
    details: dict

class OllamaTagsResponse(BaseModel):
    models: List[OllamaModel]

class DeleteRequest(BaseModel):
    doc_name: str
