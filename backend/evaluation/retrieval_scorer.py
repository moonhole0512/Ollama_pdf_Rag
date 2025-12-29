from typing import List
import numpy as np
from langchain_core.embeddings import Embeddings

from backend.core.data_models import RetrievedDoc, RetrievalResult, ScoredRetrievalResult


class RetrievalScorer:
    """Scores the quality of a set of retrieved documents."""

    def __init__(self, embeddings: Embeddings):
        if not embeddings:
            raise ValueError("Embeddings model must be provided to RetrievalScorer.")
        self.embeddings = embeddings

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates cosine similarity between two vectors."""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def score_relevance(self, docs: List[RetrievedDoc]) -> float:
        """
        Calculates relevance score based on the 'score' attribute from the retriever.
        Assumes higher scores are better.
        """
        if not docs:
            return 0.0
        
        scores = [doc.score for doc in docs if doc.score is not None]
        if not scores:
            return 0.0
            
        # Simple average of scores for now. Can be improved.
        avg_score = sum(scores) / len(scores)
        return float(avg_score)

    def score_diversity(self, docs: List[RetrievedDoc]) -> float:
        """
        Calculates diversity as 1 - average cosine similarity between document embeddings.
        A higher score (closer to 1) means more diverse documents.
        """
        if len(docs) < 2:
            return 1.0  # Maximum diversity if 0 or 1 doc

        doc_contents = [doc.content for doc in docs]
        try:
            doc_embeddings = self.embeddings.embed_documents(doc_contents)
        except Exception as e:
            print(f"Error embedding documents for diversity scoring: {e}")
            return 0.0 # Cannot calculate diversity

        similarities = []
        for i in range(len(doc_embeddings)):
            for j in range(i + 1, len(doc_embeddings)):
                similarity = self._cosine_similarity(np.array(doc_embeddings[i]), np.array(doc_embeddings[j]))
                similarities.append(similarity)
        
        if not similarities:
            return 1.0

        avg_similarity = sum(similarities) / len(similarities)
        diversity_score = 1.0 - avg_similarity
        return float(diversity_score)

    def assess(self, retrieval_result: RetrievalResult) -> ScoredRetrievalResult:
        """
        Main method to compute all scores and return a ScoredRetrievalResult.
        The `is_sufficient` flag is not yet intelligently used in Phase 2.
        """
        docs = retrieval_result.docs
        
        relevance = self.score_relevance(docs)
        diversity = self.score_diversity(docs)

        # Simple thresholding for now, will be driven by SearchPolicy later
        is_sufficient = relevance > 0.5 and diversity > 0.6 and len(docs) > 0

        print(f"📊 Retrieval Evaluation (Shadow Mode):")
        print(f"   - Relevance Score: {relevance:.3f}")
        print(f"   - Diversity Score: {diversity:.3f}")
        print(f"   - Sufficient: {is_sufficient}")

        return ScoredRetrievalResult(
            retrieval_result=retrieval_result,
            relevance_score=relevance,
            diversity_score=diversity,
            is_sufficient=is_sufficient
        )
