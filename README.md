# DocuMind: Knowledge-Base Search Engine

## Project Overview

DocuMind is a Retrieval-Augmented Generation (RAG) system designed as a knowledge-base search engine. It enables users to upload multiple documents (PDF, DOCX, TXT), process them into text chunks, generate vector embeddings, and store them in a vector database. Users can then query the system in natural language, retrieve relevant document chunks via similarity search, and receive concise, accurate answers synthesized by a Large Language Model (LLM) such as GPT-4 or Llama 3.

The project aims to provide an efficient way to search and query large volumes of unstructured text data, making it ideal for knowledge management, research, or any scenario requiring document-based Q&A.

## Key Features

- **Document Upload and Ingestion**: Supports PDF, DOCX, and TXT files. Extracts text content and handles errors for unsupported formats.
- **Text Processing**: Splits extracted text into manageable chunks (200-500 tokens) with configurable overlap for better context retention.
- **Embedding Generation**: Creates vector embeddings using Sentence Transformers (e.g., `all-MiniLM-L6-v2`) or OpenAI's `text-embedding-3-small`.
- **Vector Database Storage**: Stores embeddings in FAISS or ChromaDB for fast similarity search.
- **Query and Retrieval**: Accepts natural language queries, performs vector similarity search to retrieve top-k relevant chunks.
- **Answer Synthesis**: Uses an LLM to generate succinct answers based on retrieved chunks, with optional source citations.
- **API Backend**: Built with FastAPI for endpoints to upload documents, query the system, and check status.
- **Optional Frontend**: Streamlit-based UI for easy file uploads, query input, and result display.
- **Demo and Testing**: Includes a demo video showcasing end-to-end usage and instructions for setup and testing.

## Project Structure

```
DocuMind-RAG/
│
├── backend/
│   ├── app.py                  # FastAPI backend API with endpoints for upload, query, and status
│   ├── ingestion.py            # Document ingestion logic for extracting text from PDFs, DOCX, and TXT
│   ├── embeddings.py           # Text chunking and embedding generation
│   ├── retrieval.py            # Vector database initialization, storage, and query
│   ├── synthesis.py            # LLM-based answer generation
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   ├── app.py                  # Streamlit UI for file upload and query interface
│   └── components/             # Additional UI components (if needed)
│
├── data/
│   ├── uploaded_docs/          # Directory for user-uploaded files
│   └── vector_store/           # Directory for vector embeddings storage
│
├── demo/
│   └── demo_video.mp4          # Demo video showing project usage
│
└── README.md                   # This file - project documentation
```

## Implementation Steps

The project is built in 7 sequential steps, each producing a modular Python module with proper error handling, logging, and docstrings.

1. **Document Ingestion** (`backend/ingestion.py`):
   - Extract text from PDF using PyPDF2 or pdfminer.six.
   - Extract text from DOCX using python-docx.
   - Handle TXT files directly.
   - Implement `ingest_documents(file_list)` to process files based on extension and return a list of `{file_name, content}` dictionaries.
   - Log errors for failed or unsupported files.

2. **Text Chunking & Embeddings** (`backend/embeddings.py`):
   - Split text into chunks of 200-500 tokens with overlap.
   - Generate embeddings using Sentence Transformers or OpenAI.
   - Functions:
     - `chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]`: Splits text into chunks with specified size and overlap.
     - `generate_embeddings(chunks: List[str], method: str = "sentence-transformers") -> np.ndarray`: Generates embeddings for chunks using the specified method.

3. **Vector Database & Retrieval** (`backend/retrieval.py`):
   - Initialize FAISS or ChromaDB vector store.
   - Add embeddings to the database with persistence (save/load from `data/vector_store/`).
   - Query for top-k similar chunks.
   - Functions:
     - `initialize_vector_db(db_type: str = "faiss") -> object`: Initializes and loads/saves the vector store.
     - `add_embeddings_to_db(chunks: List[str], embeddings: np.ndarray, db: object)`: Adds chunks and embeddings to the database.
     - `query_vector_db(query_embedding: np.ndarray, db: object, top_k: int = 5) -> List[str]`: Retrieves top-k similar chunks.

4. **LLM Answer Synthesis** (`backend/synthesis.py`):
   - Generate answers using an LLM (e.g., GPT-4).
   - Combine retrieved chunks into a single context string (e.g., join with newlines or separators).
   - Prompt format: "You are an assistant answering based on provided documents. Using these excerpts, answer the user's question succinctly and accurately."
   - Function: `generate_answer(query: str, retrieved_chunks: List[str], model: str = "gpt-4") -> str`: Synthesizes answer with optional citations.

5. **Backend API** (`backend/app.py`):
   - FastAPI endpoints: `POST /upload` (ingest docs and store embeddings), `POST /query` (process query and return answer), `GET /status` (health check).
   - Include CORS for frontend integration and proper JSON responses.

6. **Optional Frontend** (`frontend/app.py`):
   - Streamlit UI with file uploader (`st.file_uploader`), query input (`st.text_input`), submit button (`st.button`), and result display (`st.write`).
   - Loading indicators (`st.spinner`) and user-friendly instructions.

7. **Testing & Demo**:
   - Test each module independently.
   - Record a demo video of uploading documents, querying, and viewing answers.
   - Update README.md with setup instructions, example usage, and screenshots.

## Tech Stack

- **Language**: Python 3.10+
- **Backend Framework**: FastAPI (or Flask as alternative)
- **Frontend Framework**: Streamlit (or React as alternative)
- **LLM**: GPT-4, Llama 3, or Mistral (via API or local inference)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`) or OpenAI (`text-embedding-3-small`)
- **Vector Database**: FAISS or ChromaDB
- **Document Parsing**:
  - PDF: PyPDF2 or pdfminer.six
  - DOCX: python-docx
  - TXT: Built-in Python file handling
- **Deployment**: Localhost (development), Render, or Hugging Face Spaces (production)

## Dependencies (requirements.txt)

Key Python packages with suggested versions for reproducibility:

**Backend Dependencies:**
```
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.4.2
python-multipart==0.0.6
typing-extensions==4.8.0
pypdf2==3.0.1
python-docx==1.1.0
sentence-transformers==2.2.2
faiss-cpu==1.9.0.post1
chromadb==0.4.18
openai==1.3.0
numpy==1.24.3
requests==2.31.0
huggingface-hub==0.19.0
```

**Frontend Dependencies:**
```
streamlit==1.28.1
requests==2.31.0
```

Install with: `pip install -r backend/requirements.txt`

## Coding Guidelines

- **Modularity**: Each module should contain focused functions with clear docstrings and comments.
- **Error Handling**: Implement try-except blocks for file operations, API calls, and LLM interactions. Log errors appropriately.
- **Logging**: Use Python's `logging` module for important steps and debugging.
- **Reusability**: Design modules to be importable and testable independently.
- **Testing**: Test each module before integration; include example usage in comments.
- **Best Practices**: Follow PEP 8 for code style, ensure compatibility with the tech stack, and optimize for performance where possible.

## Deliverables

- Complete `backend/` modules: `ingestion.py`, `embeddings.py`, `retrieval.py`, `synthesis.py`, `app.py`, and `requirements.txt`.
- Optional `frontend/` module: `app.py` with Streamlit UI.
- Data directories: `data/uploaded_docs/` and `data/vector_store/`.
- Demo: `demo/demo_video.mp4` showcasing end-to-end functionality.
- Documentation: This `README.md` with overview, setup, structure, and usage examples.

## Setup and Installation

DocuMind offers several installation options depending on your needs and system capabilities.

### Installation Options

1. **Standard Installation (Recommended)**
   - Install core dependencies first, then advanced dependencies
   - Provides full functionality with optimal performance

2. **Minimal Installation**
   - Installs only essential packages with minimal build requirements
   - Faster installation but some features may use fallback methods

3. **Manual Installation**
   - For users who want to customize the installation process
   - Good for troubleshooting or advanced users

### Windows Installation (Standard Method)

1. Clone or navigate to the project directory.
2. Double-click on `install.bat` to install core dependencies.
3. After core dependencies are installed, run `install_advanced.bat` to install additional dependencies.
   - This step is optional but recommended for full functionality.
4. Once installation is complete, run `verify_install.bat` to check that all packages are correctly installed.
5. Open two separate Command Prompt windows:
   - In the first window, run `start_backend.bat` to start the backend server.
   - In the second window, run `start_frontend.bat` to start the frontend UI.
6. Open a web browser and navigate to http://localhost:8501

### Windows Installation (Minimal Method)

If you're having trouble with the standard installation:

1. Navigate to the project directory.
2. Double-click on `install_minimal.bat` to install only essential dependencies.
3. Once installation is complete, open two separate Command Prompt windows:
   - In the first window, run `start_backend.bat` to start the backend server.
   - In the second window, run `start_frontend.bat` to start the frontend UI.
4. Open a web browser and navigate to http://localhost:8501

### Manual Installation

If you prefer to install manually or are using Linux/macOS:

1. Clone or navigate to the project directory.
2. Create a virtual environment:
   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install core dependencies:
   ```
   pip install --upgrade pip setuptools wheel
   pip install fastapi uvicorn python-docx pypdf2 numpy faiss-cpu requests streamlit
   ```
4. Start the backend server:
   ```
   uvicorn backend.app:app --reload
   ```
5. In a separate terminal, start the frontend:
   ```
   streamlit run frontend/app.py
   ```
6. Open a web browser and navigate to http://localhost:8501

### Troubleshooting

- **Missing setuptools error**: If you see an error about setuptools or backend unavailable, run `pip install --upgrade setuptools wheel` first, then try the installation again.
- **Visual Studio Build Tools Error**: If you encounter an error related to building packages that need C++ compilation, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the C++ development workload.
- **'streamlit' not found**: If you get a "streamlit not found" error, make sure you've activated the virtual environment and run `pip install streamlit`.
- **Import errors**: If you see import errors when running the application, check that all required packages are installed correctly in your virtual environment.
- **Fallback modes**: The application is designed to fall back to simpler methods if advanced dependencies aren't available. If you see warnings about falling back to different embedding methods, this is normal and the application will still work, but with potentially reduced accuracy.

## Example Usage

- Upload a PDF: Send a POST request to `/upload` with file data.
- Query: POST to `/query` with a JSON body like `{"query": "What is machine learning?"}`.
- Response: JSON with the synthesized answer and optional sources.

For more details, refer to the demo video or individual module docstrings.

## Contributing

This is a modular project; contributions should focus on improving specific modules, adding features (e.g., more file types), or optimizing performance. Ensure changes align with the coding guidelines.
