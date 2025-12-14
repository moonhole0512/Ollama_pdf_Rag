import os
import re
from typing import List, Tuple, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
import pdfplumber
from langchain_core.documents import Document

def load_pdf_with_pdfplumber(file_path: str) -> Tuple[List[Document], str]:
    """Loads a PDF using pdfplumber, preserves structure, and detects document type."""
    documents = []
    doc_type = "book"  # Default to book
    
    with pdfplumber.open(file_path) as pdf:
        first_pages_text = ""
        for i, page in enumerate(pdf.pages):
            if i < 2:
                first_pages_text += page.extract_text(x_tolerance=2, keep_blank_chars=True) or ""
            
            documents.append(Document(
                page_content=page.extract_text(x_tolerance=2, keep_blank_chars=True) or "",
                metadata={"source": os.path.basename(file_path), "page": page.page_number}
            ))

        if "abstract" in first_pages_text.lower() and "references" in first_pages_text.lower():
            doc_type = "paper"

    return documents, doc_type
