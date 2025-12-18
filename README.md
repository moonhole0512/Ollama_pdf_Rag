# PDF RAG (Retrieval-Augmented Generation) Chat

이 프로젝트는 PDF 문서의 내용을 기반으로 질문하고 답할 수 있는 RAG(검색 증강 생성) 챗봇 애플리케이션입니다. 사용자는 PDF를 업로드하여 임베딩하고, 웹 인터페이스를 통해 자연어 질문을 던져 문서 기반의 답변을 얻을 수 있습니다.

## ✨ 주요 기능

- **📄 PDF 업로드 및 처리**: PDF 문서를 업로드하면 자동으로 텍스트를 추출하고, 의미 단위로 청킹(Chunking)하여 벡터 데이터베이스에 저장합니다.
- **🧠 다중 LLM 제공자 지원**:
  - **Ollama**: 로컬에서 실행되는 LLM (Llama3, Phi-3 등)을 지원합니다.
  - **Google Gemini**: Google의 Gemini Pro 및 Flash 모델을 지원합니다.
- **🚀 지능형 쿼리 라우팅**: 사용자의 질문 의도('개념 정의', '목차 조회' 등)를 LLM이 동적으로 파악하여, 각기 다른 검색 전략을 수행합니다. 프론트엔드에서 라우팅에 사용할 모델을 직접 선택할 수 있습니다.
- **🔍 고급 RAG 파이프라인**:
  - **하이브리드 검색**: 키워드 기반의 BM25와 의미 기반의 Dense 검색을 함께 사용하여 검색 정확도를 높입니다.
  - **HyDE (Hypothetical Document Embeddings)**: 질문에 대한 '가상 답변'을 생성하고, 그 임베딩을 사용하여 관련 문서를 찾아 검색 품질을 향상시킵니다.
  - **리랭킹 (Reranking)**: 초기 검색된 결과를 Cross-Encoder 모델을 사용하여 재정렬함으로써 답변과 가장 관련성 높은 문서를 선별합니다.
- **📚 문서 구조 분석**: 책의 목차, 챕터 등 구조 정보를 자동으로 분석하여 검색 및 요약에 활용합니다.
- **💬 대화형 채팅 인터페이스**: 깔끔하고 직관적인 UI를 통해 문서의 내용에 대해 대화할 수 있습니다.
- **🗑️ 문서 삭제**: 분석 완료된 문서를 UI에서 직접 삭제할 수 있습니다.

## 🛠️ 기술 스택

- **백엔드**: Python, FastAPI, LangChain
- **프론트엔드**: React, TypeScript, Vite, Material-UI
- **벡터 데이터베이스**: FAISS (로컬 저장)

## ⚙️ 시작하기

### 사전 요구사항

- Python 3.11 이상
- Node.js 및 npm (또는 yarn)
- [Ollama](https://ollama.com/) (로컬 모델을 사용하려는 경우)

### 설치

1.  **저장소 복제:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **백엔드 종속성 설치:**
    ```bash
    pip install -r backend/requirements.txt
    ```

3.  **프론트엔드 종속성 설치:**
    ```bash
    cd frontend
    npm install
    cd .. 
    ```

4.  **(선택사항) Ollama 모델 다운로드:**
    최적의 경험을 위해 추천 모델들을 미리 다운로드 받으세요.
    ```bash
    ollama pull llama3   # 채팅 모델
    ollama pull phi3     # 라우터 모델
    ollama pull bge-m3   # 임베딩 모델
    ```

### 실행

1.  **백엔드 서버 시작:**
    프로젝트 루트 디렉터리에서 다음 명령을 실행합니다.
    ```bash
    python backend/main.py
    ```
    서버가 시작되면 프론트엔드도 자동으로 함께 제공됩니다.

2.  **애플리케이션 접속:**
    웹 브라우저를 열고 `http://localhost:8000` 로 접속하세요.

## 📖 간단 사용법

1.  **모델 설정**:
    - 좌측 상단에서 `Ollama` 또는 `Google Gemini` 제공자를 선택합니다.
    - (Ollama 선택 시) 채팅 모델, **라우터 모델**, 임베딩 모델을 선택합니다.
    - (Gemini 선택 시) Google API 키를 입력하고 모델을 선택합니다.

2.  **PDF 업로드**:
    - 'PDF Upload' 섹션으로 파일을 드래그하거나 클릭하여 PDF를 선택합니다.
    - 'Process PDF' 버튼을 눌러 분석을 시작합니다. 하단의 진행률 표시줄을 통해 진행 상황을 확인할 수 있습니다.

3.  **RAG 세션 시작**:
    - 분석이 완료되면 'Select Documents for RAG' 목록에 문서가 나타납니다.
    - 채팅에 사용할 문서를 하나 이상 체크합니다.
    - 'Start RAG Session' 버튼을 누릅니다.

4.  **채팅 시작**:
    - 우측 채팅창에 문서와 관련된 질문을 입력하고 'Send' 버튼을 누르거나 Enter 키를 입력합니다.
    - AI가 문서를 기반으로 답변과 출처를 제공합니다.

5.  **(선택사항) 문서 삭제**:
    - 'Select Documents for RAG' 목록에서 각 문서 이름 옆의 휴지통 아이콘을 클릭하여 저장된 문서를 영구적으로 삭제할 수 있습니다.
