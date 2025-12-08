import os
import requests
import shutil
import tempfile
import json
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any, Optional
from operator import itemgetter
from urllib.parse import quote_plus, unquote_plus

# --- LangChain Imports ---
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

# --- Constants ---
DB_PATH = "backend/db"

# --- Data Models ---
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
    system_prompt: Optional[str] = None
    google_api_key: Optional[str] = None
    retrieval_k: Optional[int] = 5
    chat_history: List[Tuple[str, str]] = []

class DocumentSource(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    answer: str
    source_documents: List[DocumentSource]

# --- FastAPI App Initialization ---
app = FastAPI()
retriever: Optional[Runnable] = None
model_cache: Dict[str, Any] = {}
os.makedirs(DB_PATH, exist_ok=True)

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
        model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    else: # ollama
        model = ChatOllama(model=model_name)
    model_cache[cache_key] = model
    return model

def get_embedding_model(provider: str, model_name: str, api_key: Optional[str] = None) -> Embeddings:
    cache_key = f"embedding_{provider}_{model_name}"
    if cache_key in model_cache: return model_cache[cache_key]
    if provider == "google":
        model = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    else: # ollama
        model = OllamaEmbeddings(model=model_name)
    model_cache[cache_key] = model
    return model

def docs_to_json(docs: List[Document]) -> List[Dict]:
    return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]

def docs_from_json(json_data: List[Dict]) -> List[Document]:
    return [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in json_data]

class OllamaModel(BaseModel):
    name: str
    modified_at: str
    size: int
    digest: str
    details: dict

class OllamaTagsResponse(BaseModel):
    models: List[OllamaModel]


# --- API Endpoints ---
@app.get("/api/ollama/models", response_model=OllamaTagsResponse)
async def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("--- ERROR FETCHING OLLAMA MODELS ---")
        print(traceback.format_exc())
        print("------------------------------------")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not connect to Ollama: {e}")

@app.get("/api/documents")
async def get_documents():
    if not os.path.exists(DB_PATH):
        return []
    return [unquote_plus(d) for d in os.listdir(DB_PATH) if os.path.isdir(os.path.join(DB_PATH, d))]

@app.post("/api/upload")
async def upload_pdf(
    provider: str = Form(...),
    embedding_model: str = Form(...),
    file: UploadFile = File(...),
    google_api_key: Optional[str] = Form(None)
):
    encoded_filename = quote_plus(file.filename)
    doc_dir = os.path.join(DB_PATH, encoded_filename)
    
    if os.path.exists(doc_dir):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Document '{file.filename}' already exists.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        if not docs:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not extract text from the PDF.")

        print(f"Processing '{file.filename}' for persistent storage...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(tqdm(docs, desc="Splitting documents"))

        os.makedirs(doc_dir, exist_ok=True)
        
        # Save documents (splits) to json
        with open(os.path.join(doc_dir, "docs.json"), "w", encoding="utf-8") as f:
            json.dump(docs_to_json(splits), f, ensure_ascii=False, indent=2)
            
        # Create and save FAISS index
        embeddings = get_embedding_model(provider, embedding_model, google_api_key)
        faiss_vectorstore = FAISS.from_documents(splits, embeddings)
        faiss_vectorstore.save_local(os.path.join(doc_dir, "faiss_index"))
        
        print(f"Successfully processed and saved '{file.filename}'.")
        return {"status": "success", "filename": file.filename, "message": f"Document '{file.filename}' processed and saved."}

    except Exception as e:
        print("--- ERROR DURING PDF UPLOAD ---")
        print(traceback.format_exc())
        if os.path.exists(doc_dir):
            shutil.rmtree(doc_dir) # Clean up partial processing
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process PDF: {e}")
    finally:
        os.unlink(tmp_file_path)

@app.post("/api/set-active-documents")
async def set_active_documents(req: SetActiveDocsRequest):
    global retriever
    if not req.document_names:
        retriever = None
        return {"message": "No documents selected. Retriever is cleared."}

    try:
        print(f"Loading and merging documents: {req.document_names}")
        all_splits = []
        faiss_stores = []
        
        embeddings = get_embedding_model(req.provider, req.embedding_model, req.google_api_key)

        for doc_name in tqdm(req.document_names, desc="Loading documents"):
            encoded_doc_name = quote_plus(doc_name)
            doc_dir = os.path.join(DB_PATH, encoded_doc_name)
            if not os.path.exists(doc_dir):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{doc_name}' not found.")
            
            # Load docs from json
            with open(os.path.join(doc_dir, "docs.json"), "r", encoding="utf-8") as f:
                all_splits.extend(docs_from_json(json.load(f)))

            # Load FAISS index
            faiss_stores.append(FAISS.load_local(os.path.join(doc_dir, "faiss_index"), embeddings, allow_dangerous_deserialization=True))

        # Merge FAISS stores
        merged_faiss = faiss_stores[0]
        if len(faiss_stores) > 1:
            for i in range(1, len(faiss_stores)):
                merged_faiss.merge_from(faiss_stores[i])
        
        # Initialize Keyword (BM25) Retriever on all docs
        bm25_retriever = BM25Retriever.from_documents(all_splits)
        
        # Initialize Vector (FAISS) Retriever from merged store
        faiss_retriever = merged_faiss.as_retriever()

        # Initialize Ensemble Retriever
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.75, 0.25]
        )
        print("Active RAG session updated with selected documents.")
        return {"message": f"Retriever activated for: {', '.join(req.document_names)}"}

    except Exception as e:
        print("--- ERROR SETTING ACTIVE DOCUMENTS ---")
        print(traceback.format_exc())
        retriever = None # Clear retriever on error
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to set active documents: {e}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    global retriever
    if retriever is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No active RAG session. Please select documents first.")

    try:
        # Set K for sub-retrievers if they have a 'k' attribute
        for sub_retriever in retriever.retrievers:
            if hasattr(sub_retriever, 'k'):
                sub_retriever.k = chat_request.retrieval_k

        llm = get_chat_model(chat_request.provider, chat_request.chat_model, chat_request.google_api_key)
        
        base_prompt = hub.pull("rlm/rag-prompt")
        prompt = base_prompt
        
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
        
        return {"answer": result["answer"], "source_documents": [doc.dict() for doc in result["source_documents"]]}

    except Exception as e:
        print("--- ERROR DURING CHAT ---")
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get response from LLM: {e}")

# --- Static File Serving ---
if os.path.exists("frontend/dist"):
    class SPAStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope):
            try:
                return await super().get_response(path, scope)
            except HTTPException as ex:
                if ex.status_code == 404:
                    return await super().get_response("index.html", scope)
                raise ex
    app.mount("/", SPAStaticFiles(directory="frontend/dist", html=True), name="spa")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)