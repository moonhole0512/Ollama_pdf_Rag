from typing import List
from langchain_core.documents import Document

from backend.core.data_models import DraftAnswer, Critique


class AnswerVerifier:
    """
    Acts as the 'Answer Critic' to detect hallucinations and verify citations.
    In Phase 2, this is a placeholder (shadow mode).
    """

    def __init__(self):
        # In later phases, this could load a specific NLI or LLM model for verification.
        pass

    def critique(self, draft: DraftAnswer, docs: List[Document]) -> Critique:
        """
        Analyzes a draft answer against source documents for factual grounding.
        Currently runs in shadow mode, only logging its operation.
        """
        print(f"🕵️  Answer Verifier (Shadow Mode):")
        print(f"   - Received draft to verify: '{draft.content[:100]}...'")
        print(f"   - Number of source documents: {len(docs)}")
        
        # In a real implementation, this would involve sentence-by-sentence checks.
        # For now, we return a default "all good" response.
        default_critique = Critique(
            has_hallucinations=False,
            unsupported_claims=[],
            revision_suggestions="No issues found in shadow mode."
        )
        
        return default_critique
