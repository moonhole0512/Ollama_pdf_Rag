import os
import requests
import shutil
import tempfile
import json
import traceback
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any, Optional
from urllib.parse import quote_plus, unquote_plus
from difflib import SequenceMatcher

# --- LangChain Imports ---
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.storage import InMemoryStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain import hub
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from langchain.chains.summarize import load_summarize_chain

from backend.pdf_document_loader import load_pdf_with_pdfplumber

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

# --- FastAPI App Initialization ---
app = FastAPI()
os.makedirs(DB_PATH, exist_ok=True)

model_cache: Dict[str, Any] = {}
retriever_session: Dict[str, Any] = {
    "retriever": None,
    "all_parent_docs": [],
    "document_structure": {},
}
progress_data = {"current": 0, "total": 0, "status": "idle", "message": ""}

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
    return model

def docs_to_json(docs: List[Document]) -> List[Dict]:
    return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]

def docs_from_json(json_data: List[Dict]) -> List[Document]:
    return [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in json_data]

def find_consecutive_groups(page_list: List[int], max_gap: int = 2) -> List[List[int]]:
    """연속된 페이지 그룹 찾기 (gap 허용)"""
    if not page_list:
        return []
    
    page_list = sorted(set(page_list))
    groups = []
    current_group = [page_list[0]]
    
    for page in page_list[1:]:
        if page - current_group[-1] <= max_gap:
            current_group.append(page)
        else:
            groups.append(current_group)
            current_group = [page]
    groups.append(current_group)
    
    return groups

def analyze_document_structure(docs: List[Document], doc_type: str) -> Dict[str, Any]:
    if doc_type == "paper":
        return analyze_paper_structure(docs)
    else:
        return analyze_book_structure(docs)

def analyze_paper_structure(docs: List[Document]) -> Dict[str, Any]:
    """Analyzes the structure of a research paper."""
    structure = {"sections": {}}
    section_pattern = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][\w\s:]+)")

    for doc in docs:
        content = doc.page_content
        page_num = doc.metadata.get("page")
        for line in content.split('\n'):
            match = section_pattern.match(line)
            if match:
                section_number = match.group(1)
                section_title = match.group(2).strip()
                if section_number not in structure["sections"]:
                    structure["sections"][section_number] = {
                        "title": section_title,
                        "start_page": page_num,
                    }
                    print(f"  📄 Found paper section {section_number}: '{section_title}' at page {page_num}")

    return structure

def analyze_book_structure(docs: List[Document]) -> Dict[str, Any]:
    """Analyzes the structure of a book, focusing on TOC and chapters."""
    structure = {
        "toc_pages": [],
        "chapters": {},
        "preface_pages": [],
    }
    
    toc_keywords = ["차례", "목차", "contents", "table of contents", "목 차", "차 례"]
    
    chapter_list_patterns = [
        r'(?:^|\n)(\d{1,2})[\.장\s]+([^\n\.]{5,80})',
        r'(?:^|\n)제?(\d{1,2})장[:\s]+([^\n\.]{5,80})',
        r'(?:^|\n)Chapter\s+(\d{1,2})[:\s]+([^\n\.]{5,80})',
    ]
    
    chapter_start_patterns = [
        r'^(\d{1,2})\s+([^\d\n]{10,80})$',
        r'^제?(\d{1,2})장[\s:]+([^\n]{5,80})$',
        r'^Chapter\s+(\d{1,2})[\s:]+([^\n]{5,80})$',
    ]
    
    toc_candidate_pages = []
    toc_chapter_count = {}
    chapter_info = {}
    
    # 1단계: 목차 페이지 후보 찾기
    for doc in docs:
        page_num = doc.metadata.get('page', 0)
        content = doc.page_content.strip()
        content_lower = content.lower()
        
        if page_num <= 50:
            has_toc_keyword = any(kw in content_lower for kw in toc_keywords)
            
            chapter_matches = 0
            for pattern in chapter_list_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                chapter_matches += len(matches)
            
            if has_toc_keyword or chapter_matches >= 2:
                toc_candidate_pages.append(page_num)
                toc_chapter_count[page_num] = chapter_matches
    
    # 2단계: 연속된 목차 페이지 그룹핑
    if toc_candidate_pages:
        groups = find_consecutive_groups(toc_candidate_pages, max_gap=2)
        
        best_group = []
        max_chapters = 0
        
        for group in groups:
            total_chapters = sum(toc_chapter_count.get(p, 0) for p in group)
            if total_chapters > max_chapters:
                max_chapters = total_chapters
                best_group = group
        
        if best_group:
            start_page = min(best_group)
            end_page = max(best_group)
            structure["toc_pages"] = list(range(start_page, end_page + 1))
    
    # 3단계: 본문에서 챕터 시작 위치 찾기
    for doc in docs:
        page_num = doc.metadata.get('page', 0)
        content = doc.page_content.strip()
        
        if page_num in structure["toc_pages"]:
            continue
        
        lines = content.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if len(line) < 5:
                continue
            
            for pattern in chapter_start_patterns:
                match = re.match(pattern, line)
                if match:
                    chapter_num = match.group(1).lstrip('0')
                    title = match.group(2).strip()[:80]
                    
                    if chapter_num not in chapter_info:
                        chapter_info[chapter_num] = {
                            "start_page": page_num,
                            "title": title,
                            "end_page": None
                        }
                    break
    
    # 4단계: 챕터 끝 페이지 계산
    valid_chapters = [(k, v) for k, v in chapter_info.items() if k.isdigit() and k]
    sorted_chapters = sorted(valid_chapters, key=lambda x: int(x[0]))
    for i, (ch_num, ch_data) in enumerate(sorted_chapters):
        if i < len(sorted_chapters) - 1:
            next_start = sorted_chapters[i + 1][1]["start_page"]
            ch_data["end_page"] = next_start - 1
        else:
            ch_data["end_page"] = max(doc.metadata.get('page', 0) for doc in docs)
    
    structure["chapters"] = {k: v for k, v in chapter_info.items()}
    
    if structure["chapters"]:
        first_chapter_page = min(ch["start_page"] for ch in structure["chapters"].values())
        structure["preface_pages"] = list(range(1, first_chapter_page))
    
    return structure

def concept_aware_chunking(
    docs: List[Document], 
    doc_structure: Dict[str, Any], 
    doc_type: str
) -> Tuple[List[Document], List[Document]]:
    """개념 단위를 보존하는 향상된 청킹"""
    
    # 개념 경계 패턴 정의
    concept_patterns = [
        r'\n\s*[\'\""](.{5,50})[\'\""]',  # 인용 제목
        r'\n\s*(\d+\.\s+.{5,50})',        # 번호 제목
        r'\n\s*([가-힣A-Z].{5,50})\s*\n', # 단독 제목 라인
        r'\n\s*■\s*(.{5,50})',            # 불릿 제목
    ]
    
    # Parent 청크 - 개념 경계 고려
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=[
            "\n\n\n",
            "\n\n",
            "\n",
            ". ",
            "。",
        ]
    )
    
    # Child 청크 - 더 세밀하게
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "。",
            " ",
        ]
    )

    parent_docs = parent_splitter.split_documents(docs)
    
    child_docs = []
    doc_id_counter = 0
    
    if doc_type == 'book':
        sorted_sections = sorted(doc_structure.get('chapters', {}).items(), key=lambda item: item[1]['start_page'])
    elif doc_type == 'paper':
        def sort_key(item):
            try:
                return [int(x) for x in item[0].split('.')]
            except ValueError:
                return [999]
        sorted_sections = sorted(doc_structure.get('sections', {}).items(), key=sort_key)
    else:
        sorted_sections = []

    for p_doc in parent_docs:
        page_num = p_doc.metadata.get('page')
        if page_num is None:
            continue

        current_section_title = "N/A"
        current_section_hierarchy = "N/A"

        for sec_num, sec_info in sorted_sections:
            if sec_info['start_page'] <= page_num:
                sec_end_page = sec_info.get('end_page')
                if sec_end_page is None or page_num <= sec_end_page:
                    current_section_title = sec_info['title']
                    current_section_hierarchy = sec_num
            else:
                break

        # 개념 제목 추출
        concept_title = "N/A"
        for pattern in concept_patterns:
            match = re.search(pattern, p_doc.page_content)
            if match:
                concept_title = match.group(1).strip()
                break

        p_doc.metadata["chapter_title"] = current_section_title
        p_doc.metadata["section_hierarchy"] = current_section_hierarchy
        p_doc.metadata["concept_title"] = concept_title
        
        _child_docs = child_splitter.split_documents([p_doc])
        for _c_doc in _child_docs:
            _c_doc.metadata["parent_id"] = doc_id_counter
            _c_doc.metadata["chapter_title"] = current_section_title
            _c_doc.metadata["section_hierarchy"] = current_section_hierarchy
            _c_doc.metadata["concept_title"] = concept_title
            
        child_docs.extend(_child_docs)
        doc_id_counter += 1
        
    return parent_docs, child_docs

def fuzzy_similarity(s1: str, s2: str) -> float:
    """Fuzzy 문자열 유사도 계산 (0.0 ~ 1.0)"""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def preprocess_query(query: str) -> str:
    """쿼리 전처리: 불필요한 문자 제거, 핵심 키워드 추출"""
    # 중복 따옴표 제거
    query = re.sub(r'["""\'\']{2,}', '', query)
    query = re.sub(r'["\']', '', query)
    
    # 불필요한 조사 제거 (선택적)
    # query = re.sub(r'(에\s*대해서?|에\s*관해서?|에\s*대한|에\s*관한)', '', query)
    
    return query.strip()

def calculate_chunk_importance(doc: Document, query: str) -> float:
    """청크의 중요도 점수 계산 (향상된 버전)"""
    
    content = doc.page_content.lower()
    query_lower = query.lower()
    query_terms = query_lower.split()
    
    # 1. 정확한 쿼리 매칭 (최우선)
    if query_lower in content:
        exact_match_bonus = 50.0
    else:
        exact_match_bonus = 0.0
    
    # 2. Fuzzy 매칭 (OCR 오류 대응)
    max_fuzzy_score = 0.0
    content_chunks = content.split('\n')
    for chunk in content_chunks:
        if len(chunk) > 10:  # 너무 짧은 라인 제외
            fuzzy_score = fuzzy_similarity(query_lower, chunk)
            if fuzzy_score > max_fuzzy_score:
                max_fuzzy_score = fuzzy_score
    
    fuzzy_bonus = max_fuzzy_score * 30.0 if max_fuzzy_score > 0.6 else 0.0
    
    # 3. 키워드 빈도
    keyword_count = sum(content.count(term) for term in query_terms)
    
    # 4. 키워드 밀도
    keyword_density = keyword_count / len(content) if len(content) > 0 else 0
    
    # 5. 개념 제목 매칭 (향상)
    concept_title = doc.metadata.get("concept_title", "").lower()
    title_match_score = 0.0
    
    if concept_title and concept_title != "n/a":
        # 정확한 매칭
        if query_lower in concept_title or concept_title in query_lower:
            title_match_score = 40.0
        else:
            # Fuzzy 매칭
            fuzzy_title_score = fuzzy_similarity(query_lower, concept_title)
            if fuzzy_title_score > 0.5:
                title_match_score = fuzzy_title_score * 30.0
    
    # 6. 챕터 제목 매칭
    chapter_title = doc.metadata.get("chapter_title", "").lower()
    chapter_match_score = 0.0
    
    if chapter_title and chapter_title != "n/a":
        for term in query_terms:
            if term in chapter_title:
                chapter_match_score += 5.0
    
    # 7. 페이지 위치 보너스 (개념 정의는 보통 앞부분)
    position_bonus = 1.0
    page_num = doc.metadata.get("page", 999)
    if page_num < 100:
        position_bonus = 1.1
    elif page_num < 200:
        position_bonus = 1.05
    
    # 종합 점수
    importance = (
        exact_match_bonus +
        fuzzy_bonus +
        title_match_score +
        chapter_match_score +
        keyword_density * 100 +
        keyword_count * 2
    ) * position_bonus
    
    return importance

def get_adaptive_k(query_info: Dict[str, Any], query: str) -> Dict[str, int]:
    """쿼리 특성에 따라 검색 깊이를 동적으로 조정"""
    
    # 기본값
    initial_k = 40  # 20 -> 40 (더 넓게 검색)
    final_k = 10     # 5 -> 10 (더 많은 후보 유지)
    
    query_lower = query.lower()
    
    # 1. 특정 개념/용어 검색 (정확한 매칭 필요)
    concept_indicators = [
        "이란", "이라는", "무엇", "뭐", "정의", "의미",
        "전략", "원리", "법칙", "효과", "방법", "기법"
    ]
    if any(indicator in query_lower for indicator in concept_indicators):
        initial_k = 50  # 더 넓게 검색
        final_k = 12     # 더 많은 후보 유지
    
    # 2. 비교/나열 질문 (여러 문서 필요)
    comparison_indicators = [
        "모두", "전부", "리스트", "나열", "비교", "차이", "종류"
    ]
    if any(indicator in query_lower for indicator in comparison_indicators):
        initial_k = 40
        final_k = 15
    
    # 3. 목차/구조 질문 (타겟팅된 검색)
    if query_info["type"] == "table_of_contents_lookup":
        initial_k = 10
        final_k = 10
    
    # 4. 요약 질문
    if query_info["type"] == "chapter_summary":
        initial_k = 5
        final_k = 5
    
    return {
        "initial_k": initial_k,
        "final_k": final_k,
        "use_broad_search": initial_k > 20
    }

def get_adaptive_weights(query: str, query_info: Dict[str, Any]) -> Tuple[float, float]:
    """쿼리 특성에 따라 BM25/Dense 가중치 동적 조정 (OCR 고려)"""
    
    query_lower = query.lower()
    
    # OCR 품질이 낮을 것으로 예상되는 경우 Dense 검색 강화
    # 1. 특수 문자나 따옴표가 포함된 경우
    if '"' in query or "'" in query or '「' in query or '」' in query:
        return (0.3, 0.7)  # Dense 우선
    
    # 2. 긴 구문 검색 (OCR 오류 가능성)
    if len(query.split()) > 5:
        return (0.4, 0.6)  # Dense 우선
    
    # 3. 정확한 키워드 매칭이 중요한 경우
    keyword_indicators = [
        "이란", "이라는", "전략", "법칙", "원리",
        "효과", "방법", "기법", "정의"
    ]
    if any(ind in query_lower for ind in keyword_indicators):
        # 하지만 OCR을 고려하여 Dense도 비중 있게
        return (0.4, 0.6)  # 이전 0.7, 0.3에서 변경
    
    # 4. 의미적 이해가 중요한 경우 (Dense 우선)
    semantic_indicators = [
        "왜", "어떻게", "설명", "이유", "과정",
        "관계", "영향", "차이"
    ]
    if any(ind in query_lower for ind in semantic_indicators):
        return (0.3, 0.7)
    
    # 5. 개념 정의의 경우 균형
    if query_info.get("type") == "concept_definition":
        return (0.4, 0.6)  # Dense 약간 우선 (OCR 대응)
    
    # 6. 기본값 (균형)
    return (0.4, 0.6)  # 이전 0.5, 0.5에서 변경

from typing import List, Tuple, Dict, Any, Optional, Literal

# --- LLM-based Query Router ---
class IntelligentRouterOutput(BaseModel):
    """LLM이 반환할 라우팅 및 쿼리 재작성 결과 모델"""
    intent: Literal[
        "concept_definition", 
        "table_of_contents_lookup", 
        "chapter_summary", 
        "general_information_retrieval"
    ]
    rewritten_query: str
    expanded_queries: List[str]
    
    
LLM_ROUTER_SYSTEM_PROMPT = """You are an expert query analyzer and rewriter for a Retrieval-Augmented Generation (RAG) system.
Your task is to understand the user's query, classify its intent, and rewrite it for optimal retrieval.

**1. De-contextualize:**
If the query contains pronouns like 'that', 'this', 'it', '이거', '저거', '그거', use the provided chat history to resolve them and create a self-contained, complete question.
- Example (History: "What is the 'reciprocity' principle?", User: "Tell me more about it.") -> Rewritten: "Tell me more about the 'reciprocity' principle."

**2. Classify Intent:**
Categorize the rewritten query into one of the following intents:
- `concept_definition`: Asks for the definition, explanation, or meaning of a specific term, concept, principle, or strategy. (e.g., "What is cognitive dissonance?", "설득의 6가지 원칙이란?")
- `table_of_contents_lookup`: Asks for the table of contents, structure, or list of chapters. (e.g., "Show me the table of contents.", "목차 보여줘.")
- `chapter_summary`: Asks to summarize a specific chapter. (e.g., "Summarize chapter 3.", "3장 요약해줘.")
- `general_information_retrieval`: All other questions that seek specific information, examples, or general knowledge from the document. This is the default.

**3. Rewrite and Expand:**
- **`rewritten_query`**: Create a clear, concise, and keyword-rich version of the de-contextualized query. This should be the best possible query for a search engine.
- **`expanded_queries`**: Generate 3 additional, diverse search queries based on the original question to improve search recall. These should explore different phrasings, synonyms, or related aspects.

**Output Format:**
You MUST respond with a single, valid JSON object that adheres to the `IntelligentRouterOutput` schema. Do not add any text before or after the JSON.
Example JSON:
{{
  "intent": "concept_definition",
  "rewritten_query": "Definition and examples of the commitment and consistency principle",
  "expanded_queries": [
    "commitment and consistency principle explained",
    "How does the commitment and consistency rule work?",
    "사회적 증거의 원칙"
  ]
}}
"""


def intelligent_query_router(
    query: str, 
    chat_history: List[Tuple[str, str]],
    llm: BaseChatModel
) -> Dict[str, Any]:
    """LLM을 사용한 지능형 쿼리 분류 및 재작성 (수동 JSON 파싱)"""
    
    history_str = "\n".join([f"Human: {h}\nAI: {a}" for h, a in chat_history])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", LLM_ROUTER_SYSTEM_PROMPT),
        ("human", "Chat History:\n---\n{history}\n---\n\nUser Query: \"{query}\"")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        print("🧠 Calling LLM Router...")
        response_str = chain.invoke({
            "history": history_str,
            "query": query
        })
        
        # LLM 응답에서 JSON 부분만 추출
        match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if not match:
            raise ValueError("LLM did not return a valid JSON object.")
        
        json_str = match.group(0)
        
        # JSON 파싱 및 Pydantic 모델로 검증
        response_data = json.loads(json_str)
        response = IntelligentRouterOutput.model_validate(response_data)

        # 결과를 기존 형식에 맞게 변환
        output = {
            "type": response.intent,
            "rewritten_query": response.rewritten_query,
            "search_queries": [response.rewritten_query] + response.expanded_queries,
        }
        
        # 특수 타입에 필요한 정보 추가
        if response.intent == "concept_definition":
            output["concept"] = response.rewritten_query.replace("Definition and examples of the", "").strip()
        elif response.intent == "chapter_summary":
            match_num = re.search(r'\d+', response.rewritten_query)
            if match_num:
                output["chapter_num"] = match_num.group(0)

        # fallback 함수가 rewritten_query를 반환하지 않으므로 추가
        if "rewritten_query" not in output:
            output["rewritten_query"] = query

        return output
        
    except Exception as e:
        print(f"--- LLM ROUTER ERROR ---")
        print(f"Error: {e}")
        print("Falling back to legacy rule-based classification.")
        return classify_query_advanced_fallback(query, chat_history)


def classify_query_advanced_fallback(
    query: str, 
    chat_history: List[Tuple[str, str]]
) -> Dict[str, Any]:
    """Fallback: 향상된 쿼리 분류 및 확장 (기존 로직)"""
    
    # 쿼리 전처리
    query = preprocess_query(query)
    rewritten_query = query # fallback에서는 재작성 기능 없으므로 그대로 사용
    query_lower = query.lower()
    
    # 1. 개념 정의 질문 감지
    definition_patterns = [
        r'(.+?)(이란|란|이라는|라는|은 무엇|는 무엇|이 뭐|가 뭐)',
        r'(.+?)(에 대해|에대해).+(설명|말해|알려)',
        r'(전략|원리|법칙|효과|방법|기법).+(뭐|무엇)',
    ]
    for pattern in definition_patterns:
        match = re.search(pattern, query_lower)
        if match:
            concept = match.group(1).strip()
            concept_clean = re.sub(r'\s+(에|의|를|을|가|이|은|는)$', '', concept)
            return {
                "type": "concept_definition", "concept": concept_clean,
                "rewritten_query": rewritten_query,
                "search_queries": [query, concept_clean, f'{concept_clean} 전략', f'{concept_clean} 방법']
            }
    
    # 2. 목차 질문
    toc_patterns = [r'목차', r'차례', r'구성', r'table of contents', r'toc']
    if any(re.search(p, query_lower) for p in toc_patterns):
        return {"type": "table_of_contents_lookup", "rewritten_query": rewritten_query, "search_queries": [query]}
    
    # 3. 챕터 요약
    summary_match = re.search(r'(summarize|요약)\s*(?:chapter|장)?\s*(\d{1,2})', query_lower)
    if summary_match:
        return {
            "type": "chapter_summary", "chapter_num": summary_match.group(2),
            "rewritten_query": rewritten_query, "search_queries": [query]
        }
    
    # 4. 일반 주제 질문
    return {
        "type": "general_information_retrieval",
        "rewritten_query": rewritten_query,
        "search_queries": [query, f'{query} 설명', f'{query} 예시']
    }


HYDE_PROMPT = """You are a helpful assistant. The user will ask a question.
Your task is to write a short, one-paragraph, hypothetical answer to the question.
This answer will be used to find similar documents.
Focus on capturing the key concepts and terminology. Do not say you don't know the answer.
Be concise and clear.

User question: {question}
Hypothetical answer:"""

def generate_hypothetical_answer(query: str, llm: BaseChatModel) -> str:
    """HyDE: LLM을 사용하여 질문에 대한 가상 답변 생성"""
    prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": query})


def multi_stage_retrieval(
    query_info: Dict[str, Any],
    retrievers: Dict[str, Any],
    all_parent_docs: List[Document],
    chat_request: ChatRequest,
    llm: BaseChatModel,
    embeddings: Embeddings
) -> List[Document]:
    """다단계 검색 파이프라인 (HyDE + Reranking)"""
    
    original_query = chat_request.question
    rewritten_query = query_info["rewritten_query"]
    
    # Stage 1: 적응형 K값 및 가중치 결정
    adaptive_k = get_adaptive_k(query_info, rewritten_query)
    bm25_weight, dense_weight = get_adaptive_weights(rewritten_query, query_info)
    
    print(f"\n🔍 Adaptive Search Config:")
    print(f"   Query Type: {query_info['type']}")
    print(f"   Rewritten Query: {rewritten_query}")
    print(f"   K: {adaptive_k['final_k']} (Initial: {adaptive_k['initial_k']})")
    print(f"   Weights: BM25={bm25_weight:.2f}, Dense={dense_weight:.2f}")

    # Stage 2: HyDE (Hypothetical Document Embeddings)
    hypothetical_answer = generate_hypothetical_answer(rewritten_query, llm)
    print(f"   🧠 HyDE Answer: {hypothetical_answer[:100]}...")
    hyde_embedding = embeddings.embed_query(hypothetical_answer)

    # Stage 3: 하이브리드 검색 (BM25 + HyDE-based Dense)
    # BM25 검색
    bm25_retriever = retrievers['bm25']
    bm25_docs = bm25_retriever.get_relevant_documents(rewritten_query)
    
    # Dense 검색 (ParentDocumentRetriever의 vectorstore 직접 사용)
    parent_retriever = retrievers['parent']
    dense_docs_with_scores = parent_retriever.vectorstore.similarity_search_with_score_by_vector(
        hyde_embedding, k=adaptive_k['initial_k']
    )
    
    # ParentDocumentRetriever를 통해 부모 문서 가져오기
    dense_parent_doc_ids = [doc.metadata['parent_id'] for doc, score in dense_docs_with_scores]
    dense_parent_docs = parent_retriever.docstore.mget(dense_parent_doc_ids)

    # Stage 4: 결과 병합 및 가중치 적용
    combined_results = {}
    for i, doc in enumerate(bm25_docs):
        combined_results[doc.page_content] = combined_results.get(doc.page_content, 0) + bm25_weight * (1 / (i + 1))
        
    for i, doc in enumerate(dense_parent_docs):
        if doc: # docstore.mget은 None을 반환할 수 있음
             combined_results[doc.page_content] = combined_results.get(doc.page_content, 0) + dense_weight * (1 / (i + 1))

    # 점수 기준으로 정렬
    sorted_docs_content = sorted(combined_results.keys(), key=lambda k: combined_results[k], reverse=True)
    
    # 원본 Document 객체 찾기
    doc_map = {doc.page_content: doc for doc in all_parent_docs}
    initial_retrieved_docs = [doc_map[content] for content in sorted_docs_content if content in doc_map]

    if not initial_retrieved_docs:
        print("   ⚠️ No documents retrieved from hybrid search.")
        return []

    print(f"   📚 Hybrid retrieved (before rerank): {len(initial_retrieved_docs)}")

    # Stage 5: Cross-Encoder Reranking
    cross_encoder = get_cross_encoder_model()
    rerank_pairs = [[rewritten_query, doc.page_content] for doc in initial_retrieved_docs[:50]] # Rerank 상위 50개만
    
    if rerank_pairs:
        ce_scores = cross_encoder.predict(rerank_pairs)
    else:
        ce_scores = []
        
    docs_with_scores = []
    for doc, ce_score in zip(rerank_pairs, ce_scores):
        # 구조적 중요도 점수 반영
        importance = calculate_chunk_importance(initial_retrieved_docs[len(docs_with_scores)], rewritten_query)
        # 최종 점수 = Cross-Encoder 점수 + 구조적 중요도
        final_score = float(ce_score) + (importance / 100.0)
        docs_with_scores.append((initial_retrieved_docs[len(docs_with_scores)], final_score, importance, ce_score))

    docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    final_docs = [doc for doc, _, _, _ in docs_with_scores[:adaptive_k['final_k']]]

    # Stage 6: 디버깅 정보 출력
    print(f"\n📊 Top Reranked Results:")
    for i, (doc, score, imp, ce) in enumerate(docs_with_scores[:10], 1):
        page = doc.metadata.get('page', '?')
        sec_hier = doc.metadata.get('section_hierarchy', 'N/A')
        preview = doc.page_content[:60].replace('\n', ' ')
        print(f"   {i:2}. P{page:3} | Score: {score:.3f} (CE: {ce:.3f}, Imp: {imp:.1f}) | Sec: {sec_hier} | {preview}...")

    return final_docs

def get_targeted_documents(
    query_info: Dict[str, Any],
    all_parent_docs: List[Document],
    doc_structure: Dict[str, Any]
) -> List[Document]:
    """쿼리 타입에 따라 정확한 문서 추출"""
    
    if query_info["type"] == "toc":
        toc_pages = doc_structure.get("toc_pages", [])
        
        if toc_pages:
            print(f"🎯 목차 페이지 추출: {toc_pages}")
            target_docs = [
                doc for doc in all_parent_docs 
                if doc.metadata.get('page', 0) in toc_pages
            ]
            
            if target_docs:
                target_docs.sort(key=lambda d: d.metadata.get('page', 0))
                print(f"  ✅ {len(target_docs)}개 문서 추출 완료")
                return target_docs
        
        print("⚠️ 목차 자동 감지 실패, 초반 30페이지 검색")
        target_docs = [
            doc for doc in all_parent_docs 
            if doc.metadata.get('page', 999) <= 30
        ]
        target_docs.sort(key=lambda d: d.metadata.get('page', 0))
        return target_docs
    
    elif query_info["type"] == "chapter_summary":
        ch_num = query_info["chapter_num"]
        chapters = doc_structure.get("chapters", {})
        
        if ch_num in chapters:
            ch_info = chapters[ch_num]
            start, end = ch_info["start_page"], ch_info["end_page"]
            print(f"🎯 {ch_num}장 페이지 범위: {start}-{end}")
            
            target_docs = [
                doc for doc in all_parent_docs
                if start <= doc.metadata.get('page', 0) <= end
            ]
            target_docs.sort(key=lambda d: d.metadata.get('page', 0))
            print(f"  ✅ {len(target_docs)}개 문서 추출 완료")
            return target_docs
        else:
            print(f"⚠️ {ch_num}장 위치 미발견")
    
    return all_parent_docs

def summarize_with_map_reduce(docs: List[Document], llm: BaseChatModel, chat_request: ChatRequest):
    """Summarizes a large document using the Map-Reduce strategy."""
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary

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

@app.post("/api/upload")
async def upload_pdf(
    provider: str = Form(...),
    embedding_model: str = Form(...),
    file: UploadFile = File(...),
    google_api_key: Optional[str] = Form(None)
):
    global progress_data
    progress_data = {"current": 0, "total": 0, "status": "starting", "message": "Upload started..."}

    encoded_filename = quote_plus(file.filename)
    doc_dir = os.path.join(DB_PATH, encoded_filename)
    
    if os.path.exists(doc_dir):
        raise HTTPException(status_code=409, detail=f"Document '{file.filename}' already exists.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_file_path = tmp_file.name

    try:
        progress_data.update({"status": "loading", "message": "Loading PDF with pdfplumber..."})
        docs, doc_type = load_pdf_with_pdfplumber(tmp_file_path)
        if not docs:
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

        print(f"\n📄 Document Type: {doc_type}")
        print(f"📚 '{file.filename}' 구조 분석 중...")
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
            import numpy as np

            doc_texts = [doc.page_content for doc in child_docs]
            text_embeddings = [None] * len(doc_texts)
            
            max_workers = min(32, (os.cpu_count() or 1) * 4)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(embeddings.embed_query, text): i for i, text in enumerate(doc_texts)
                }
                
                progress_data["current"] = 0
                for future in tqdm(as_completed(future_to_index), total=len(doc_texts), desc="Generating Embeddings (Parallel)"):
                    index = future_to_index[future]
                    try:
                        vector = future.result()
                        text_embeddings[index] = (doc_texts[index], vector)
                    except Exception as exc:
                        print(f'   - Document {index} generated an exception: {exc}')
                    
                    progress_data["current"] += 1
            
            successful_embeddings = [item for item in text_embeddings if item is not None]
            successful_metadatas = [
                child_docs[i].metadata for i, item in enumerate(text_embeddings) if item is not None
            ]

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
        else:
            empty_index = FAISS.from_texts([" "], embeddings)
            empty_index.save_local(os.path.join(doc_dir, "faiss_index"))

        print(f"\n✅ '{file.filename}' 처리 완료")
        progress_data.update({"status": "completed", "message": "Processing complete."})
        return {"status": "success", "filename": file.filename}

    except Exception as e:
        print(f"--- ERROR DURING PDF UPLOAD ---\n{traceback.format_exc()}")
        if os.path.exists(doc_dir):
            shutil.rmtree(doc_dir)
        progress_data.update({"status": "error", "message": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")
    finally:
        os.unlink(tmp_file_path)

@app.post("/api/set-active-documents")
async def set_active_documents(req: SetActiveDocsRequest):
    global retriever_session
    if not req.document_names:
        retriever_session = {"retrievers": None, "all_parent_docs": [], "document_structure": {}}
        return {"message": "No documents selected."}

    try:
        print(f"\n📚 문서 로딩: {req.document_names}")
        all_child_docs, all_parent_docs = [], []
        parent_doc_store = InMemoryStore()
        faiss_stores = []
        combined_structure = {"toc_pages": [], "chapters": {}, "sections": {}}
        
        embeddings = get_embedding_model(req.provider, req.embedding_model, req.google_api_key)
        parent_id_offset = 0

        for doc_name in tqdm(req.document_names, desc="Loading"):
            encoded_doc_name = quote_plus(doc_name)
            doc_dir = os.path.join(DB_PATH, encoded_doc_name)
            if not os.path.exists(doc_dir):
                raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found.")
            
            with open(os.path.join(doc_dir, "docs.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
                child_docs = docs_from_json(data["child_docs"])
                parent_docs = docs_from_json(data["parent_docs"])
                doc_structure = data.get("structure", {})
                
                for c in child_docs: 
                    c.metadata["parent_id"] += parent_id_offset
                
                parent_ids = [str(i + parent_id_offset) for i in range(len(parent_docs))]
                parent_doc_store.mset(list(zip(parent_ids, parent_docs)))
                all_child_docs.extend(child_docs)
                all_parent_docs.extend(parent_docs)
                parent_id_offset += len(parent_docs)
                
                if doc_structure:
                    if 'toc_pages' in doc_structure:
                        combined_structure["toc_pages"].extend(doc_structure.get("toc_pages", []))
                    if 'chapters' in doc_structure:
                        combined_structure["chapters"].update(doc_structure.get("chapters", {}))
                    if 'sections' in doc_structure:
                        combined_structure["sections"].update(doc_structure.get("sections", {}))

            faiss_stores.append(FAISS.load_local(
                os.path.join(doc_dir, "faiss_index"), 
                embeddings, 
                allow_dangerous_deserialization=True
            ))

        merged_faiss = faiss_stores[0]
        if len(faiss_stores) > 1:
            for store in faiss_stores[1:]:
                merged_faiss.merge_from(store)
        
        parent_retriever = ParentDocumentRetriever(
            vectorstore=merged_faiss,
            docstore=parent_doc_store,
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100),
            parent_id_field="parent_id"
        )
        bm25_retriever = BM25Retriever.from_documents(all_parent_docs)
        
        retriever_session["retrievers"] = {
            "bm25": bm25_retriever,
            "parent": parent_retriever
        }
        retriever_session["all_parent_docs"] = all_parent_docs
        retriever_session["document_structure"] = combined_structure
        retriever_session["embeddings"] = embeddings # 임베딩 모델 저장
        
        print(f"✅ RAG 세션 활성화 완료")
        print(f"   Total parent docs: {len(all_parent_docs)}")
        print(f"   Total child docs: {len(all_child_docs)}")
        
        return {"message": f"Retriever activated: {', '.join(req.document_names)}"}

    except Exception as e:
        print(f"--- ERROR ---\n{traceback.format_exc()}")
        retriever_session = {"retrievers": None, "all_parent_docs": [], "document_structure": {}}
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    global retriever_session
    if not retriever_session.get("retrievers"):
        raise HTTPException(status_code=400, detail="No active RAG session.")

    try:
        # 1. 기본 LLM 및 임베딩 모델 로드
        llm = get_chat_model(
            chat_request.provider, 
            chat_request.chat_model, 
            chat_request.google_api_key
        )
        embeddings = retriever_session["embeddings"]
        
        # 2. 라우팅을 위한 경량 LLM 로드 (Ollama 사용)
        router_llm = None
        if chat_request.router_model and chat_request.router_model != "default":
            try:
                router_llm = get_chat_model("ollama", chat_request.router_model)
                print(f"🧠 Using selected router model: {chat_request.router_model}")
            except Exception as e:
                print(f"⚠️ Failed to load selected router model '{chat_request.router_model}'. Error: {e}")
        
        if not router_llm:
            try:
                router_llm = get_chat_model("ollama", "phi3") 
            except Exception:
                try:
                    router_llm = get_chat_model("ollama", "llama3")
                except Exception:
                    print("⚠️ Default router LLM (phi3, llama3) not found. Using main chat model for routing.")
                    router_llm = llm

        # 3. LLM 기반 지능형 쿼리 라우터 호출
        query_info = intelligent_query_router(
            chat_request.question, 
            chat_request.chat_history,
            router_llm
        )
        print(f"\n🧠 LLM Router Output:")
        print(f"   Intent: {query_info['type']}")
        if 'concept' in query_info:
            print(f"   Concept: {query_info.get('concept', 'N/A')}")
        print(f"   Search Queries: {query_info['search_queries']}")
        
        all_parent_docs = retriever_session["all_parent_docs"]
        doc_structure = retriever_session["document_structure"]
        
        # --- 특수 케이스 처리: 목차 조회 및 챕터 요약 ---
        if query_info["type"] in ["chapter_summary", "table_of_contents_lookup"]:
            temp_query_info = query_info.copy()
            if temp_query_info["type"] == "table_of_contents_lookup":
                temp_query_info["type"] = "toc"

            target_docs = get_targeted_documents(
                temp_query_info, 
                all_parent_docs, 
                doc_structure
            )
            
            if query_info.get("type") == "chapter_summary":
                total_content = "".join([doc.page_content for doc in target_docs])
                if len(total_content) > 28000:
                    print("📚 Large chapter detected, using Map-Reduce")
                    answer = summarize_with_map_reduce(target_docs, llm, chat_request)
                    return ChatResponse(
                        answer=answer, 
                        source_documents=[doc.dict() for doc in target_docs],
                        citations=[],
                        used_search_queries=query_info["search_queries"]
                    )
        
        else:
            # --- HyDE 기반 다단계 검색 파이프라인 ---
            retrievers = retriever_session["retrievers"]
            
            target_docs = multi_stage_retrieval(
                query_info=query_info,
                retrievers=retrievers,
                all_parent_docs=all_parent_docs,
                chat_request=chat_request,
                llm=llm,
                embeddings=embeddings
            )
        
        # --- 답변 생성 ---
        if not target_docs:
             answer = "죄송합니다, 관련 정보를 문서에서 찾을 수 없습니다."
             citations = []
        else:
            context_string = "\n\n".join([
                f"[Page {doc.metadata.get('page', '?')}, "
                f"Section {doc.metadata.get('section_hierarchy', 'N/A')}]\n"
                f"{doc.page_content}" 
                for doc in target_docs
            ])
            
            system_instruction = (
                "You are a technical expert assistant. Analyze the provided context carefully and answer the question.\n\n"
                "Instructions:\n"
                "- For concept definitions, provide a clear and precise explanation\n"
                "- Use examples from the context when available\n"
                "- Cite your sources using the format [Page X, Section Y]\n"
                "- If the exact answer is not in the context, state this clearly\n"
                "- For Korean queries, respond in Korean\n"
                "- Be concise but thorough\n"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_instruction),
                ("human", "Context:\n{context}\n\nQuestion: {question}\n\nProvide a detailed answer with citations.")
            ])
            
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({
                "context": context_string,
                "question": chat_request.question
            })
            citations = re.findall(r'\[Page \d+[^\]]*\]', answer)

        print(f"\n✅ Answer generated successfully")
        print(f"   Citations found: {len(citations)}")

        return ChatResponse(
            answer=answer,
            source_documents=[doc.dict() for doc in target_docs],
            citations=list(set(citations)),
            used_search_queries=query_info["search_queries"]
        )

    except Exception as e:
        print(f"--- ERROR ---\n{traceback.format_exc()}")
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