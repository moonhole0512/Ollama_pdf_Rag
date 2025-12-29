import re
import json
from typing import List, Tuple, Dict, Any, Optional, Literal
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

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
    chapter_number: Optional[int] = None
    
    
LLM_ROUTER_SYSTEM_PROMPT = """You are an expert query analyzer and rewriter for a Retrieval-Augmented Generation (RAG) system.
Your task is to understand the user's query, classify its intent, rewrite it for optimal retrieval, and extract relevant entities.

**1. De-contextualize:**
If the query contains pronouns like 'that', 'this', 'it', '이거', '저거', '그거', use the provided chat history to resolve them and create a self-contained, complete question.
- Example (History: "What is the 'reciprocity' principle?", User: "Tell me more about it.") -> Rewritten: "Tell me more about the 'reciprocity' principle."

**2. Classify Intent & Extract Entities:**
Categorize the rewritten query into one of the following intents. If you extract an entity, place it in the corresponding field.
- `concept_definition`: Asks for the definition, explanation, or meaning of a specific term, concept, principle, or strategy. (e.g., "What is cognitive dissonance?", "설득의 6가지 원칙이란?")
- `table_of_contents_lookup`: Asks for the table of contents, structure, or list of chapters. (e.g., "Show me the table of contents.", "목차 보여줘.")
- `chapter_summary`: Asks to summarize a specific chapter. 
  - **You MUST extract the chapter number (e.g., from 'chapter 3', 'third chapter', '3장') as an integer and put it in the `chapter_number` field.**
- `general_information_retrieval`: All other questions that seek specific information, examples, or general knowledge from the document. This is the default.

**3. Rewrite and Expand:**
- **`rewritten_query`**: Create a clear, concise, and keyword-rich version of the de-contextualized query. This should be the best possible query for a search engine.
- **`expanded_queries`**: Generate 3 additional, diverse search queries based on the original question to improve search recall. These should explore different phrasings, synonyms, or related aspects.

**Output Format:**
You MUST respond with a single, valid JSON object that adheres to the `IntelligentRouterOutput` schema. Do not add any text before or after the JSON.
Example JSON for chapter summary:
{{
  "intent": "chapter_summary",
  "rewritten_query": "Summarize the third chapter about social proof",
  "expanded_queries": [
    "summary of chapter 3",
    "key points of the chapter on social proof",
    "main ideas from the third chapter"
  ],
  "chapter_number": 3
}}

Example JSON for concept definition:
{{
  "intent": "concept_definition",
  "rewritten_query": "Definition and examples of the commitment and consistency principle",
  "expanded_queries": [
    "commitment and consistency principle explained",
    "How does the commitment and consistency rule work?",
    "사회적 증거의 원칙"
  ],
  "chapter_number": null
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
            # LLM이 추출한 chapter_number를 우선 사용
            if response.chapter_number is not None:
                output["chapter_num"] = str(response.chapter_number)
            else:
                # Fallback: 원본 쿼리에서 숫자 직접 찾기
                match_num = re.search(r'\d+', query)
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
    from backend.retrieval.hybrid_retriever import preprocess_query # Avoid circular import
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

