# Interview Helper

Personal Interview Assistant with RAG (Retrieval Augmented Generation) powered by Ollama and ChromaDB.

## 🎯 Overview

Interview Helper is a personal interview assistant that stores, organizes, and retrieves your interview information using AI technology. It uses:

- **Ollama** - For LLM inference (works with local models like llama3, mistral, etc.)
- **ChromaDB** - For vector storage and semantic search
- **RAG Architecture** - Combines retrieved context with AI for smart answers

## ✨ Features

### Core Functionality
- 📝 **Store Interviews** - Save interview experiences with metadata
- 🔍 **Search & Retrieve** - Find specific topics, questions, or discussions
- 📊 **Pattern Analysis** - Identify trends across interviews
- 💬 **Ask Questions** - Query your stored data using natural language

### Key Capabilities
1. **Chronological Storage**: All interviews stored with date, company, role, and location info
2. **Topic Search**: Find content by subject matter (technical questions, HR topics, etc.)
3. **Pattern Recognition**: See how your performance evolves across interviews
4. **Natural Language Queries**: Ask "How did I do in technical rounds?" and get insights

## 🚀 Quick Start

### Prerequisites
```bash
# Install Ollama (https://ollama.ai)
# Pull a model
ollama pull llama3  # or mistral, codellama, etc.
```

### Installation
```bash
pip install .
```

### Basic Usage
```bash
# Add an interview
interview-helper add "Your interview experience goes here" --company TechCorp --role "Senior Engineer" --date 2026-05-29 --tags "technical,system-design"

# Search for topics
interview-helper search "behavioral questions"

# Ask questions about your interviews
interview-helper ask "What are the common technical questions I faced?"
```

## 📖 Usage Examples

### Adding an Interview
```bash
interview-helper add \
  --company Google \
  --role "Software Engineer - L4" \
  --date 2026-05-15 \
  --location "Mountain View, CA" \
  "The interview process had multiple rounds including coding assessments, \
   system design discussions about scalable systems, and behavioral questions \
   about my past projects. The technical team was particularly interested in \
   my experience with distributed systems..."
```

### Searching Content
```bash
# Find all interviews at specific company
interview-helper search --company Google

# Search by tags
interview-helper search --tags "system-design"

# Full-text search for topics
interview-helper search "algorithm questions data structures"
```

### Pattern Analysis
```bash
# Get insights about your interview performance
interview-helper analyze --topic "technical"

# Compare across companies
interview-helper compare companies

# Trend analysis
interview-helper trend --metric "difficulty"
```

## 🛠️ Architecture

### Tech Stack
- **Backend**: Python with async/await support
- **Vector Database**: ChromaDB (local file-based)
- **LLM Interface**: Ollama API
- **Data Structures**: Pydantic models for type safety
- **CLI Framework**: Rich CLI

### Core Components
```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  CLI Interface  │────▶│ InterviewHelper  │────▶│  OllamaClient│
│                 │     │                  │     │              │
└─────────────────┘     ├──────────────────┘     └──────┬───────┘
                        └───────────────┬───────────────┘
                                         ▼
                                   ┌──────────────┐
                                   │  ChromaDB    │
                                   │  (Vector DB) │
                                   └──────────────┘
```

### Data Flow
1. **Input**: Text + metadata → `DataLoader`
2. **Processing**: Chunk text → `DocumentChunk`
3. **Embedding**: Generate vectors → `OllamaClient.create_embedding()`
4. **Storage**: Index in ChromaDB collection
5. **Retrieval**: Query → retrieve relevant chunks → RAG generation

## 📁 Project Structure

```
interview_helper/
├── __init__.py          # Package exports
├── config.py            # Configuration management
├── data_loader.py       # Text processing & chunking
├── ollama_client.py     # Ollama API client
└── core.py              # Main InterviewHelper class
```

## ⚙️ Configuration

### Environment Variables
```bash
# Ollama API endpoint (default: http://localhost:11434)
export OLLAMA_URL="http://your-server:11434"

# Default model to use
export OLLAMA_MODEL="llama3"

# Embedding model
export EMBEDDING_MODEL="all-minist:v6"
```

### Custom Configuration
```python
from interview_helper.config import AppConfig

config = AppConfig()
config.set_ollama_url("http://localhost:11434")
```

## 🔧 Development

### Setup
```bash
pip install -e .
pip install -r requirements-dev.txt
pytest
```

### Architecture Patterns Used
- **Dependency Injection**: OllamaClient injection for easy testing
- **Repository Pattern**: ChromaDB as vector storage repository
- **Command/Query Segregation**: Separate input and query logic
- **Chain of Responsibility**: Multi-step RAG pipeline

## 📊 Data Structure

### Interview Metadata
```json
{
  "title": "Senior Software Engineer Interview",
  "date": "2026-05-15",
  "company": "TechCorp",
  "role": "Staff Engineer",
  "location": "San Francisco, CA",
  "tags": ["technical", "system-design", "behavioral"],
  "content": "Full interview content..."
}
```

### Vector Chunks
Each interview is split into semantic chunks for better retrieval:
- ~500 characters per chunk
- 30-char overlap between chunks
- Tagged with metadata for filtering

## 🧪 Testing

```bash
# Run all tests
pytest

# Test with mock Ollama client
pytest --mock-ollama

# Coverage report
pytest --cov=interview_helper
```


### Code Style
- Use type hints
- Write docstrings
- Follow PEP 8
- Add tests for new features

## 📄 License

MIT License - Feel free to use this for personal development.

## 🔗 Technologies

- [Ollama](https://ollama.ai) - Local LLM inference server
- [ChromaDB](https://www.trychroma.com) - Vector database
- [Rich](https://rich.readthedocs.io/) - Beautiful CLI output
- [Pydantic](https://docs.pydantic.dev/) - Type validation

## 🙋 Support

For issues or questions, please create an issue in the repository.

---

**Built by Ajaiswal1818** | **Made with ❤️ for interview preparation**
