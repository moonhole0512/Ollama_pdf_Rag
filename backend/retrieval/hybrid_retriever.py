import re
from typing import List, Tuple, Dict, Any
from difflib import SequenceMatcher
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# This will be initialized from main.py to avoid circular model loading
cross_encoder = None

def set_cross_encoder_model(model):
    global cross_encoder
    cross_encoder = model

def fuzzy_similarity(s1: str, s2: str) -> float:
    """Fuzzy 문자열 유사도 계산 (0.0 ~ 1.0)"""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def preprocess_query(query: str) -> str:
    """쿼리 전처리: 불필요한 문자 제거, 핵심 키워드 추출"""
    # 중복 따옴표 제거
    query = re.sub(r'["\']{2,}', '', query)
    query = re.sub(r'["\']', '', query)
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
    initial_k = 40
    final_k = 10
    
    query_lower = query.lower()
    
    # 1. 특정 개념/용어 검색 (정확한 매칭 필요)
    concept_indicators = [
        "이란", "이라는", "무엇", "뭐", "정의", "의미",
        "전략", "원리", "법칙", "효과", "방법", "기법"
    ]
    if any(indicator in query_lower for indicator in concept_indicators):
        initial_k = 50
        final_k = 12
    
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
    
    if '"' in query or "'" in query or '「' in query or '」' in query:
        return (0.3, 0.7)
    
    if len(query.split()) > 5:
        return (0.4, 0.6)
    
    keyword_indicators = [
        "이란", "이라는", "전략", "법칙", "원리",
        "효과", "방법", "기법", "정의"
    ]
    if any(ind in query_lower for ind in keyword_indicators):
        return (0.4, 0.6)
    
    semantic_indicators = [
        "왜", "어떻게", "설명", "이유", "과정",
        "관계", "영향", "차이"
    ]
    if any(ind in query_lower for ind in semantic_indicators):
        return (0.3, 0.7)
    
    if query_info.get("type") == "concept_definition":
        return (0.4, 0.6)
    
    return (0.4, 0.6)

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
    original_query: str,
    llm: BaseChatModel,
    embeddings: Embeddings
) -> List[Document]:
    """다단계 검색 파이프라인 (HyDE + Reranking)"""
    
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
    bm25_retriever = retrievers['bm25']
    bm25_docs = bm25_retriever.get_relevant_documents(rewritten_query)
    
    parent_retriever = retrievers['parent']
    dense_docs_with_scores = parent_retriever.vectorstore.similarity_search_with_score_by_vector(
        hyde_embedding, k=adaptive_k['initial_k']
    )
    
    dense_parent_doc_ids = [doc.metadata['parent_id'] for doc, score in dense_docs_with_scores]
    dense_parent_docs = parent_retriever.docstore.mget(dense_parent_doc_ids)

    # Stage 4: 결과 병합 및 가중치 적용
    combined_results = {}
    for i, doc in enumerate(bm25_docs):
        combined_results[doc.page_content] = combined_results.get(doc.page_content, 0) + bm25_weight * (1 / (i + 1))
        
    for i, doc in enumerate(dense_parent_docs):
        if doc:
             combined_results[doc.page_content] = combined_results.get(doc.page_content, 0) + dense_weight * (1 / (i + 1))

    sorted_docs_content = sorted(combined_results.keys(), key=lambda k: combined_results[k], reverse=True)
    
    doc_map = {doc.page_content: doc for doc in all_parent_docs}
    initial_retrieved_docs = [doc_map[content] for content in sorted_docs_content if content in doc_map]

    if not initial_retrieved_docs:
        print("   ⚠️ No documents retrieved from hybrid search.")
        return []

    print(f"   📚 Hybrid retrieved (before rerank): {len(initial_retrieved_docs)}")

    # Stage 5: Cross-Encoder Reranking
    global cross_encoder
    if not cross_encoder:
        raise Exception("Cross-encoder model not set. Please call set_cross_encoder_model first.")

    rerank_pairs = [[rewritten_query, doc.page_content] for doc in initial_retrieved_docs[:50]]
    
    if rerank_pairs:
        ce_scores = cross_encoder.predict(rerank_pairs)
    else:
        ce_scores = []
        
    docs_with_scores = []
    for i, (doc, ce_score) in enumerate(zip(initial_retrieved_docs, ce_scores)):
        importance = calculate_chunk_importance(doc, rewritten_query)
        final_score = float(ce_score) + (importance / 100.0)
        doc.metadata['score'] = final_score # Add score to metadata for scorer
        docs_with_scores.append((doc, final_score, importance, ce_score))

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
    
    return []