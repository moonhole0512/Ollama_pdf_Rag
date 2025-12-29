import os
import requests
import shutil
import tempfile
import json
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any, Optional
from urllib.parse import quote_plus, unquote_plus

# --- LangChain Imports ---
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ParentDocumentRetriever
from tqdm import tqdm

# --- Internal Module Imports ---
from backend.orchestration.orchestrator import Orchestrator
from backend.indexing.pdf_parser import (
    load_pdf_with_pdfplumber,
    analyze_document_structure,
    concept_aware_chunking,
)
from backend.indexing.vector_store import docs_to_json, docs_from_json
from backend.core.model_manager import (
    get_embedding_model,
    retriever_session,
    progress_data,
)
from backend.core.data_models import (
    SetActiveDocsRequest,
    ChatRequest,
    ChatResponse,
    OllamaModel,
    OllamaTagsResponse,
    DeleteRequest
)


# --- Constants ---
DB_PATH = "backend/db"

# --- FastAPI App Initialization ---
app = FastAPI()
os.makedirs(DB_PATH, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== API Endpoints ====================

@app.get("/api/progress")
async def get_progress_status():
    return progress_data

@app.get("/api/ollama/models", response_model=OllamaTagsResponse)
async def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"--- ERROR FETCHING OLLAMA MODELS ---\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Could not connect to Ollama: {e}")

@app.get("/api/documents")
async def get_documents():
    if not os.path.exists(DB_PATH):
        return []
    return [unquote_plus(d) for d in os.listdir(DB_PATH) if os.path.isdir(os.path.join(DB_PATH, d))]

@app.delete("/api/documents")
async def delete_document(req: DeleteRequest):
    doc_name = req.doc_name
    if not doc_name or ".." in doc_name or "/" in doc_name or "\"" in doc_name:
        raise HTTPException(status_code=400, detail="Invalid document name.")
    
    try:
        encoded_doc_name = quote_plus(doc_name)
        doc_dir = os.path.join(DB_PATH, encoded_doc_name)
        
        if not os.path.abspath(doc_dir).startswith(os.path.abspath(DB_PATH)):
            raise HTTPException(status_code=400, detail="Invalid document path.")

        if os.path.isdir(doc_dir):
            shutil.rmtree(doc_dir)
            return {"status": "success", "message": f"Document '{doc_name}' deleted successfully."}
        else:
            raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found.")
    except Exception as e:
        print(f"--- ERROR DELETING DOCUMENT '{doc_name}' ---\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")

def process_pdf_background(
    tmp_file_path: str, 
    filename: str, 
    provider: str, 
    embedding_model: str, 
    google_api_key: Optional[str]
):
    global progress_data
    encoded_filename = quote_plus(filename)
    doc_dir = os.path.join(DB_PATH, encoded_filename)

    try:
        progress_data.update({"status": "loading", "message": "Loading PDF..."})
        docs, doc_type = load_pdf_with_pdfplumber(tmp_file_path)
        if not docs:
            raise Exception("Could not extract text from the PDF.")

        print(f"Analyzing document structure...")
        doc_structure = analyze_document_structure(docs, doc_type)

        progress_data.update({"status": "chunking", "message": "Splitting document..."})
        parent_docs, child_docs = concept_aware_chunking(docs, doc_structure, doc_type)

        os.makedirs(doc_dir, exist_ok=True)
        
        with open(os.path.join(doc_dir, "docs.json"), "w", encoding="utf-8") as f:
            json.dump({
                "doc_type": doc_type,
                "parent_docs": docs_to_json(parent_docs),
                "child_docs": docs_to_json(child_docs),
                "structure": doc_structure
            }, f, ensure_ascii=False, indent=2)
            
        progress_data.update({"status": "embedding", "total": len(child_docs), "message": "Generating embeddings..."})
        embeddings = get_embedding_model(provider, embedding_model, google_api_key)

        faiss_vectorstore = None
        if child_docs:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            doc_texts = [doc.page_content for doc in child_docs]
            text_embeddings = [None] * len(doc_texts)
            max_workers = min(32, (os.cpu_count() or 1) * 4)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = { executor.submit(embeddings.embed_query, text): i for i, text in enumerate(doc_texts) }
                progress_data["current"] = 0
                for future in tqdm(as_completed(future_to_index), total=len(doc_texts), desc="Generating Embeddings"):
                    index = future_to_index[future]
                    try:
                        text_embeddings[index] = (doc_texts[index], future.result())
                    except Exception as exc:
                        print(f'Document {index} generated an exception: {exc}')
                    progress_data["current"] += 1
            
            successful_embeddings = [item for item in text_embeddings if item is not None]
            successful_metadatas = [ child_docs[i].metadata for i, item in enumerate(text_embeddings) if item is not None ]

            if not successful_embeddings:
                raise Exception("Failed to generate any document embeddings.")
            
            progress_data.update({"status": "indexing", "message": "Creating FAISS index..."})
            faiss_vectorstore = FAISS.from_embeddings(
                text_embeddings=successful_embeddings,
                embedding=embeddings,
                metadatas=successful_metadatas
            )
        
        if faiss_vectorstore:
            faiss_vectorstore.save_local(os.path.join(doc_dir, "faiss_index"))

        progress_data.update({"status": "completed", "message": "Processing complete."})

    except Exception as e:
        print(f"--- ERROR DURING PDF BACKGROUND PROCESSING ---\n{traceback.format_exc()}")
        if os.path.exists(doc_dir):
            shutil.rmtree(doc_dir)
        progress_data.update({"status": "error", "message": str(e)})
    finally:
        os.unlink(tmp_file_path)

@app.post("/api/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    provider: str = Form(...), 
    embedding_model: str = Form(...), 
    file: UploadFile = File(...), 
    google_api_key: Optional[str] = Form(None)
):
    global progress_data
    progress_data = {"current": 0, "total": 0, "status": "starting", "message": "Initiating upload..."}

    if os.path.exists(os.path.join(DB_PATH, quote_plus(file.filename))):
        raise HTTPException(status_code=409, detail=f"Document '{file.filename}' already exists.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_file_path = tmp_file.name

    background_tasks.add_task(
        process_pdf_background, 
        tmp_file_path, file.filename, provider, embedding_model, google_api_key
    )
    
    return {"status": "processing_started", "filename": file.filename}

@app.post("/api/set-active-documents")
async def set_active_documents(req: SetActiveDocsRequest):
    global retriever_session
    if not req.document_names:
        retriever_session = {"retrievers": None, "all_parent_docs": [], "document_structure": {}, "embeddings": None}
        return {"message": "No documents selected."}

    try:
        all_child_docs, all_parent_docs = [], []
        parent_doc_store = InMemoryStore()
        faiss_stores = []
        combined_structure = {"toc_pages": [], "chapters": {}, "sections": {}}
        
        embeddings = get_embedding_model(req.provider, req.embedding_model, req.google_api_key)
        parent_id_offset = 0

        for doc_name in tqdm(req.document_names, desc="Loading Documents"):
            encoded_doc_name = quote_plus(doc_name)
            doc_dir = os.path.join(DB_PATH, encoded_doc_name)
            if not os.path.exists(doc_dir):
                raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found.")
            
            with open(os.path.join(doc_dir, "docs.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
                child_docs = docs_from_json(data["child_docs"])
                parent_docs = docs_from_json(data["parent_docs"])
                doc_structure = data.get("structure", {})
                
                for c in child_docs: c.metadata["parent_id"] += parent_id_offset
                parent_ids = [str(i + parent_id_offset) for i in range(len(parent_docs))]
                parent_doc_store.mset(list(zip(parent_ids, parent_docs)))
                all_child_docs.extend(child_docs)
                all_parent_docs.extend(parent_docs)
                parent_id_offset += len(parent_docs)
                
                for key in ["toc_pages", "chapters", "sections"]:
                    if key in doc_structure:
                        if isinstance(doc_structure[key], list):
                            combined_structure[key].extend(doc_structure[key])
                        elif isinstance(doc_structure[key], dict):
                            combined_structure[key].update(doc_structure[key])

            faiss_stores.append(FAISS.load_local(
                os.path.join(doc_dir, "faiss_index"), 
                embeddings, 
                allow_dangerous_deserialization=True
            ))

        merged_faiss = faiss_stores[0]
        if len(faiss_stores) > 1:
            for store in faiss_stores[1:]:
                merged_faiss.merge_from(store)
        
        parent_retriever = ParentDocumentRetriever(vectorstore=merged_faiss, docstore=parent_doc_store, child_splitter=RecursiveCharacterTextSplitter(chunk_size=400), parent_id_field="parent_id")
        bm25_retriever = BM25Retriever.from_documents(all_parent_docs)
        
        retriever_session.update({
            "retrievers": {"bm25": bm25_retriever, "parent": parent_retriever},
            "all_parent_docs": all_parent_docs,
            "document_structure": combined_structure,
            "embeddings": embeddings
        })
        
        return {"message": f"Retriever activated for: {', '.join(req.document_names)}"}

    except Exception as e:
        print(f"--- ERROR SETTING ACTIVE DOCS ---\n{traceback.format_exc()}")
        retriever_session = {"retrievers": None, "all_parent_docs": [], "document_structure": {}, "embeddings": None}
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    global retriever_session
    if not retriever_session.get("retrievers") or not retriever_session.get("embeddings"):
        raise HTTPException(status_code=400, detail="No active RAG session. Please set documents first.")

    try:
        orchestrator = Orchestrator(chat_request)
        response = orchestrator.execute_pipeline()
        return response
    except Exception as e:
        print(f"--- ERROR IN CHAT ENDPOINT ---\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

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
