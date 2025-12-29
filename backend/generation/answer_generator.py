from langchain.chains.summarize import load_summarize_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document
from typing import List

# A placeholder for the future ChatRequest data model
class ChatRequest:
    pass

def summarize_with_map_reduce(docs: List[Document], llm: BaseChatModel, chat_request: ChatRequest):
    """Summarizes a large document using the Map-Reduce strategy."""
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary
