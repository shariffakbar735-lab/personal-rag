# LLM AXIS — Local PDF RAG

A simple Retrieval-Augmented Generation (RAG) application built with Python, LangChain, ChromaDB and Ollama. It allows users to ask questions about PDF documents using a locally running language model.

## Architecture

1. **PDF Extraction:** PyMuPDF extracts text from PDF documents.
2. **Text Splitting:** LangChain splits extracted text into chunks.
3. **Embeddings:** Ollama generates embeddings using `nomic-embed-text`.
4. **Vector Storage:** ChromaDB stores document embeddings.
5. **Retrieval:** Retrieves the top 3 relevant chunks for a query.
6. **Generation:** Llama 3.2 (1B) generates answers using the retrieved context.

## Tech Stack

- Python
- LangChain
- ChromaDB
- Ollama
- Llama 3.2 (1B)
- nomic-embed-text
- PyMuPDF

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/shariffakbar735-lab/llm-axis.git
cd llm-axis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama models

Install [Ollama](https://ollama.com/) and download the required models:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

Ensure Ollama is running locally.

### 5. Add your PDF

Place your PDF document in the project directory, or update the `pdf_path` variable in `llm.py`.

### 6. Run the application

```bash
python llm.py
```

Type `exit` to end the conversation.

## Configuration

- LLM: `llama3.2:1b`
- Embedding Model: `nomic-embed-text`
- Chunk Size: 512
- Chunk Overlap: 64
- Retrieved Documents: 3
- Vector Database: ChromaDB
- Persistence Directory: `./cprs_db`

## Project Status

Experimental local RAG implementation. Under development.