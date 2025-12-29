import re
from typing import Dict, Any

# --- LangChain Imports ---
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Internal Module Imports ---
from backend.core.data_models import ChatRequest, ChatResponse, DocumentSource, RetrievalResult
from backend.core.model_manager import (
    get_chat_model,
    get_embedding_model,
    get_cross_encoder_model,
    retriever_session,
)
from backend.query_understanding.router import intelligent_query_router
from backend.retrieval.hybrid_retriever import multi_stage_retrieval, get_targeted_documents
from backend.generation.answer_generator import summarize_with_map_reduce
from backend.evaluation.retrieval_scorer import RetrievalScorer
from backend.evaluation.answer_verifier import AnswerVerifier

class Orchestrator:
    """
    Main orchestrator for the RAG pipeline.
    It coordinates the query understanding, retrieval, evaluation, and generation steps.
    """
    def __init__(self, chat_request: ChatRequest):
        self.chat_request = chat_request
        self.llm = get_chat_model(chat_request.provider, chat_request.chat_model, chat_request.google_api_key)
        self.embeddings = retriever_session["embeddings"]
        get_cross_encoder_model()  # Ensures the cross-encoder is loaded

        # Initialize shadow mode evaluators
        self.retrieval_scorer = RetrievalScorer(self.embeddings)
        self.answer_verifier = AnswerVerifier()

    def execute_pipeline(self) -> ChatResponse:
        """
        Executes the full RAG pipeline from query to final answer.
        """
        # --- Internal Imports for Pydantic Models ---
        from backend.core.data_models import QueryContext, RetrievedDoc

        # 1. Create initial QueryContext
        query_context = QueryContext(
            original_query=self.chat_request.question,
            chat_history=[{"role": "user" if i % 2 == 0 else "assistant", "content": msg} 
                          for i, msg_pair in enumerate(self.chat_request.chat_history) for msg in msg_pair],
        )

        # 2. Route Query
        router_llm = self.llm
        if self.chat_request.router_model and self.chat_request.router_model != "default":
            try:
                router_llm = get_chat_model("ollama", self.chat_request.router_model)
            except Exception:
                print(f"Could not load router model {self.chat_request.router_model}, falling back.")

        query_info = intelligent_query_router(
            self.chat_request.question, self.chat_request.chat_history, router_llm
        )
        print(f"\n🧠 LLM Router Output: {query_info}")
        
        # Update QueryContext with router results
        query_context.rewritten_query = query_info.get("rewritten_query", self.chat_request.question)


        all_parent_docs = retriever_session["all_parent_docs"]
        doc_structure = retriever_session["document_structure"]

        # 3. Retrieve Documents
        if query_info["type"] in ["chapter_summary", "table_of_contents_lookup"]:
            temp_query_info = query_info.copy()
            if temp_query_info["type"] == "table_of_contents_lookup":
                temp_query_info["type"] = "toc"
            target_docs = get_targeted_documents(temp_query_info, all_parent_docs, doc_structure)
        else:
            target_docs = multi_stage_retrieval(
                query_info=query_info,
                retrievers=retriever_session["retrievers"],
                all_parent_docs=all_parent_docs,
                original_query=self.chat_request.question,
                llm=self.llm,
                embeddings=self.embeddings,
            )

        # --- PHASE 2: Shadow Evaluation of Retrieval ---
        if target_docs:
            # Convert LangChain Documents to our Pydantic RetrievedDoc model
            retrieved_docs = [
                RetrievedDoc(
                    doc_id=doc.metadata.get("doc_id", f"doc_{i}"),
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=doc.metadata.get("score", 0.0)
                ) for i, doc in enumerate(target_docs)
            ]
            
            # Create the RetrievalResult object with the correct types
            retrieval_result_obj = RetrievalResult(docs=retrieved_docs, query_context=query_context)
            
            # This call just logs the scores for now
            self.retrieval_scorer.assess(retrieval_result_obj)
        # ---------------------------------------------

        # 4. Generate Answer
        if not target_docs:
            answer = "죄송합니다, 관련 정보를 문서에서 찾을 수 없습니다."
            final_answer_obj = ChatResponse(answer=answer, source_documents=[], citations=[], used_search_queries=query_info.get("search_queries", []))
        
        elif query_info.get("type") == "chapter_summary":
            answer = summarize_with_map_reduce(target_docs, self.llm, self.chat_request)
            final_answer_obj = ChatResponse(answer=answer, source_documents=[doc.dict() for doc in target_docs], used_search_queries=query_info.get("search_queries", []))

        else:
            context_string = "\n\n".join([f"[Page {doc.metadata.get('page', '?')}, Section {doc.metadata.get('section_hierarchy', 'N/A')}]\n{doc.page_content}" for doc in target_docs])
            
            system_instruction = "You are a helpful assistant. Please answer the user's question based on the context provided. Cite sources as [Page X, Section Y]."
            prompt = ChatPromptTemplate.from_messages([("system", system_instruction), ("human", "Context:\n{context}\n\nQuestion: {question}")])
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context_string, "question": self.chat_request.question})
            citations = re.findall(r'\[Page \d+[^\]]*\]', answer)

            final_answer_obj = ChatResponse(
                answer=answer,
                source_documents=[doc.dict() for doc in target_docs],
                citations=list(set(citations)),
                used_search_queries=query_info.get("search_queries", [])
            )

        # --- PHASE 2: Shadow Evaluation of Answer ---
        from backend.core.data_models import DraftAnswer
        draft_doc_ids = [doc.metadata.get('doc_id') for doc in target_docs if doc.metadata.get('doc_id') is not None]
        draft = DraftAnswer(content=final_answer_obj.answer, used_doc_ids=draft_doc_ids)
        # This call just logs the action for now
        self.answer_verifier.critique(draft, target_docs)
        # ------------------------------------------

        print(f"\n✅ Answer generated successfully. Citations found: {len(final_answer_obj.citations)}")
        return final_answer_obj
