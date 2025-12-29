import os
import re
from typing import List, Tuple, Dict, Any
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
