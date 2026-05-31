# InterviewHelper - Personal Interview RAG System

## Overview

A personal interview assistant that stores and retrieves interview data using Retrieval-Augmented Generation (RAG). Built with **ChromaDB** for vector storage and **Ollama** for local LLM inference.

---

## Architecture

### Core Components

```
┌─────────────────┐      ┌──────────────┐      ┌─────────────┐      ┌──────────────┐
│   Interview      │────▶│  DataLoader   │────▶│  Embeddings │────▶│  ChromaDB     │
│   Text           │      │   (Chunking)  │      │   (Ollama)    │      │   (Vector DB) │
└─────────────────┘      └──────────────┘      └─────────────┘      └──────────────┘
                                               ↑                           │
                                          ┌─────────────┐                  │
Query → Embedding ←─────────────────────│  VectorStore │◀───────────────┤
                               (ChromaDB Query)   └─────────────┘
                                     ↓
                             ┌─────────────┐
                             │   LLM        │
                             │   (Ollama)    │
                             └─────────────┘
```

### Module Structure

```
src/interview_helper/
├── __init__.py           # Package exports
├── core.py               # Main InterviewHelper orchestrator
├── cli.py                # CLI interface (click-based)
├── data_loader.py        # Text preprocessing & chunking
├── vector_store.py       # ChromaDB wrapper
└── ollama_client.py      # Ollama API client
```

---

## Key Classes

### 1. `InterviewHelper` (core.py)
Main orchestrator class providing RAG functionality:

**Methods:**
- `add_interview(content, metadata, chunk_size)` - Stores interview in vector database
- `search(query, where_clause, top_k)` - Semantic search with metadata filtering
- `ask(question, company_name)` - Natural language Q&A using retrieved context
- `get_all_interviews()` - Retrieve all stored interviews

**Usage:**
```python
helper = InterviewHelper()
helper.add_interview(
    content="Interview details...",
    metadata={"company": "Google", "role": "SDE"}
)
answer = helper.ask("What was my performance like?")
```

### 2. `VectorStore` (vector_store.py)
ChromaDB wrapper for persistent vector storage:

**Methods:**
- `add_documents(documents, embeddings, metadatas)` - Index documents with embeddings
- `query(query_vector, where, top_k)` - Semantic similarity search
- `get_all(limit)` - Retrieve all documents

**Features:**
- Persistent ChromaDB storage at `./chroma_db`
- Metadata filtering (company, date, tags)
- Configurable chunk size for retrieval quality

### 3. `DataLoader` (data_loader.py)
Text preprocessing and chunking utilities:

**Methods:**
- `load()` - Split text into overlapping chunks
- `_clean_text()` - Text normalization
- `_chunk_text(size, overlap)` - Chunk with configurable overlap
- `extract_tags()` - Extract tags from metadata

**Chunking Strategy:**
```python
# Example: 500 char chunks with 30 char overlap
while start < len(text):
    chunk = text[start:start + chunk_size]
    chunks.append(chunk)
    start += chunk_size - chunk_overlap
```

### 4. `OllamaClient` (ollama_client.py)
Local LLM API client for embeddings and text generation:

**Methods:**
- `generate(prompt, model)` - Text completion using Llama3/other models
- `create_embedding(text, model)` - Generate vector embeddings
- `list_models()` - List available Ollama models

**API Endpoints:**
- `/api/generate` - Text generation
- `/api/embed` - Embedding generation
- `/api/tags` - Model listing

---

## Installation

```bash
pip install -e .           # Install in development mode
pip install -e ".[dev]"    # Include test dependencies
```

**Dependencies:**
- `ollama-api >=0.1.1`
- `chromadb >=0.4.13`
- `sentence-transformers >=2.2.0`
- `click >=8.1.7`
- `rich >=13.7.1`

---

## Usage

### CLI Mode
```bash
# Add interview
interview-helper add --company "Google" --role "SDE" --content "(interview text)"

# Ask question about specific company
interview-helper ask "What was my performance?" --company "Google"

# List all interviews
interview-helper list
```

### Python API

```python
from interview_helper.core import InterviewHelper

helper = InterviewHelper()

# Add interview
helper.add_interview(
    content="""
    Interview went well. I solved the array problem in O(n) time.
    The interviewer asked about edge cases and I handled them well.
    Asked good questions about team structure and work culture.
    """,
    metadata={
         "company": "Google",
         "role": "Software Engineer",
         "date": "2025-01-15",
         "tags": ["technical", "good-preparation"]
     }
)

# Search with filters
results = helper.search(
    query="array problem solution approach",
    where_clause={"company": "Google"},
    top_k=3
)

# Ask natural language questions
answer = helper.ask("How did I perform on the coding question?")
```

---

## Data Flow

1. **Input**: Raw interview text + metadata dictionary
2. **Chunking**: Text split into overlapping segments (500 chars, 30 overlap)
3. **Embedding**: Each chunk → vector embedding via Ollama
4. **Storage**: Embeddings + metadata → ChromaDB vector collection
5. **Retrieval**: Query → semantic similarity search → top-k matches
6. **Generation**: Context + question → LLM response

---

## Configuration

### Ollama Setup
```bash
# Start Ollama server (if not running)
ollama serve

# Pull embedding model (example: nomic-embed-text)
ollama pull nomic-embed-text

# Pull Llama3 for generation
ollama pull llama3
```

---

## RAG Pipeline Explained

### Retrieval Phase
- User query → text embedding via Ollama
- Semantic similarity search in ChromaDB
- Filter by metadata (company, date, etc.)
- Return top-k most relevant chunks

### Augmentation Phase
- Retrieved context combined with user question
- System prompt defines assistant role and constraints
- Context window construction for LLM

### Generation Phase
- Llama3 generates answer based on retrieved context
- Response includes citations from source material
- Maintains factual accuracy from interview records

---

## Extending the System

### Custom Chunking Strategy
```python
class MyDataLoader(DataLoader):
    def _chunk_text(self, text):
        # Custom chunking logic
        return [
             {"content": chunk1, "metadata": {...}},
             {"content": chunk2, "metadata": {...}}
         ]
```

### Different LLM Models
```python
from interview_helper.core import InterviewHelper
from interview_helper.ollama_client import OllamaClient

helper = InterviewHelper(ollama_client=OllamaClient())
answer = helper.ask("Question", company_name="Google")
# Configure OllamaClient to use different model
```

---

## Development

### Running Tests
```bash
pytest tests/ -v
pytest tests/ -v --cov=interview_helper
```

---

## Troubleshooting

### Embedding Generation Fails
- Ensure Ollama server is running: `ollama serve`
- Verify embedding model loaded: `ollama list`
- Check OLLAMA_URL configuration

### No Search Results
- Verify documents added correctly: `helper.get_all_interviews()`
- Check chunk metadata is properly indexed
- Review metadata keys in where_clause filter

### Low Retrieval Quality
- Increase `chunk_size` for better context preservation
- Adjust `chunk_overlap` (30 default) based on text patterns
- Consider using specialized embedding model

---

## Limitations

- **Local-only**: Requires Ollama server running locally
- **Memory**: Vector embeddings stored in RAM by ChromaDB
- **Language**: Best performance with English queries
- **Context Window**: Limited by LLM's max token capacity
