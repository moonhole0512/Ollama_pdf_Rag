from typing import Dict, Any, Optional

# --- LangChain Imports ---
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from sentence_transformers import CrossEncoder

# --- Internal Module Imports ---
# Use a local import to break the circular dependency cycle
from backend.retrieval.hybrid_retriever import set_cross_encoder_model as set_retriever_cross_encoder

# --- Global State ---
model_cache: Dict[str, Any] = {}
retriever_session: Dict[str, Any] = {
    "retrievers": None,
    "all_parent_docs": [],
    "document_structure": {},
    "embeddings": None,
}
progress_data: Dict[str, Any] = {"current": 0, "total": 0, "status": "idle", "message": ""}


# --- Helper Functions (Model Loading) ---
def get_chat_model(provider: str, model_name: str, api_key: Optional[str] = None) -> BaseChatModel:
    cache_key = f"chat_{provider}_{model_name}"
    if cache_key in model_cache: return model_cache[cache_key]
    if provider == "google":
        model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)
    else:
        model = ChatOllama(model=model_name, temperature=0)
    model_cache[cache_key] = model
    return model

def get_embedding_model(provider: str, model_name: str, api_key: Optional[str] = None) -> Embeddings:
    cache_key = f"embedding_{provider}_{model_name}"
    if cache_key in model_cache: return model_cache[cache_key]
    if provider == "google":
        model = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    else:
        model = OllamaEmbeddings(model=model_name)
    model_cache[cache_key] = model
    return model

def get_cross_encoder_model():
    cache_key = "cross_encoder"
    if cache_key in model_cache: return model_cache[cache_key]
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    model_cache[cache_key] = model
    set_retriever_cross_encoder(model)
    return model
