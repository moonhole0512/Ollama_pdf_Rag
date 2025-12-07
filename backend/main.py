import os
import requests
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any, Optional
from operator import itemgetter

# --- LangChain Imports (Consolidated and Corrected) ---
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain import hub
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from tqdm import tqdm


# --- Data Models ---
class ChatRequest(BaseModel):
    provider: str
    question: str
    chat_model: str
    embedding_model: str
    system_prompt: Optional[str] = None
    google_api_key: Optional[str] = None
    retrieval_k: Optional[int] = 20 # Note: This k is now used by sub-retrievers
    chat_history: List[Tuple[str, str]] = []

class DocumentSource(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    answer: str
    source_documents: List[DocumentSource]

class OllamaModel(BaseModel):
    name: str
    modified_at: str
    size: int
    digest: str
    details: dict

class OllamaTagsResponse(BaseModel):
    models: List[OllamaModel]

# --- FastAPI App Initialization ---
app = FastAPI()
retriever: Optional[Runnable] = None
model_cache: Dict[str, Any] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
def get_chat_model(provider: str, model_name: str, api_key: Optional[str] = None) -> BaseChatModel:
    cache_key = f"chat_{provider}_{model_name}"
    if cache_key in model_cache: return model_cache[cache_key]

    if provider == "google":
        if not api_key: raise ValueError("Google API Key is required for Gemini models.")
        model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    elif provider == "ollama":
        model = ChatOllama(model=model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    model_cache[cache_key] = model
    return model

def get_embedding_model(provider: str, model_name: str, api_key: Optional[str] = None) -> Embeddings:
    cache_key = f"embedding_{provider}_{model_name}"
    if cache_key in model_cache: return model_cache[cache_key]
    
    if provider == "google":
        if not api_key: raise ValueError("Google API Key is required for Gemini embedding models.")
        model = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    elif provider == "ollama":
        model = OllamaEmbeddings(model=model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    model_cache[cache_key] = model
    return model

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# --- API Endpoints ---
@app.get("/api/ollama/models", response_model=OllamaTagsResponse)
async def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not connect to Ollama: {e}")

@app.post("/api/upload")
async def upload_pdf(
    provider: str = Form(...),
    embedding_model: str = Form(...),
    file: UploadFile = File(...),
    google_api_key: Optional[str] = Form(None)
):
    global retriever
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        if not docs:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not extract any pages from the PDF.")

        # --- Hybrid Search (Ensemble Retriever) Setup ---
        print("Processing and embedding PDF for Hybrid Search...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(tqdm(docs, desc="Splitting documents"))

        # Initialize Keyword (BM25) Retriever
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 5 # Retrieve top 5 based on keywords

        # Initialize Vector (FAISS) Retriever
        embeddings = get_embedding_model(provider, embedding_model, google_api_key)
        faiss_vectorstore = FAISS.from_documents(splits, embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 based on semantics

        # Initialize Ensemble Retriever
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.75, 0.25]  # Give more weight to keyword search
        )
        
        print("PDF processing complete. Hybrid retriever is ready.")
        return {"status": "success", "filename": file.filename, "message": "PDF processed and Hybrid retriever is ready."}
    except Exception as e:
        import traceback
        print("--- ERROR DURING PDF PROCESSING ---")
        print(traceback.format_exc())
        print("------------------------------------")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process PDF: {e}")
    finally:
        os.unlink(tmp_file_path)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    global retriever
    if retriever is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload a PDF first.")

    try:
        llm = get_chat_model(chat_request.provider, chat_request.chat_model, chat_request.google_api_key)
        
        base_prompt = hub.pull("rlm/rag-prompt")
        if chat_request.system_prompt and chat_request.system_prompt.strip():
            new_system_template = chat_request.system_prompt + "\n\n" + base_prompt.messages[0].prompt.template
            base_prompt.messages[0] = SystemMessagePromptTemplate.from_template(new_system_template)
        prompt = base_prompt
        
        # The EnsembleRetriever is used directly
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )
        
        rag_chain_with_source = (
            setup_and_retrieval
            | RunnableParallel(
                answer=(prompt | llm | StrOutputParser()),
                source_documents=(itemgetter("context")),
              )
        )
        
        result = rag_chain_with_source.invoke(chat_request.question)
        
        # Sanitize source_documents
        sanitized_sources = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in result["source_documents"]
        ]

        return {"answer": result["answer"], "source_documents": sanitized_sources}
    except Exception as e:
        # Log the full exception for debugging
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get response from LLM: {e}")

# --- Static File Serving ---
class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            raise ex

if os.path.exists("frontend/dist"):
    app.mount("/", SPAStaticFiles(directory="frontend/dist", html=True), name="spa")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)