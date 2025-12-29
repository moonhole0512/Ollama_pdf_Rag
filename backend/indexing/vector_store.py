from langchain_core.documents import Document
from typing import List, Dict

def docs_to_json(docs: List[Document]) -> List[Dict]:
    return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]

def docs_from_json(json_data: List[Dict]) -> List[Document]:
    return [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in json_data]
