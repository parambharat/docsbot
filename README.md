# DocsBot

**A documentation chat agent built for the CoreWeave hackathon, 2025**

DocsBot is an AI-powered documentation assistant that provides context-aware answers across multiple documentation sources including Weights & Biases (wandb), Weave, and CoreWeave. Built with openai-agents-sdk and chromadb, it delivers responses from the documentation sources with proper citations.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![AI](https://img.shields.io/badge/AI-Multi--LLM-purple)
![Vector DB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)

## Key Features

### **Multi-Expert AI System**
- **Specialized Agents**: Dedicated experts for Wandb, Weave, and CoreWeave documentation
- **Multi-LLM Support**: Integrates OpenAI, Anthropic, and Cohere models
- **Intelligent Routing**: Automatically directs queries to the most relevant expert

### **Advanced Retrieval & Search**
- **Semantic Search**: ChromaDB vector database for intelligent document retrieval
- **Smart Reranking**: Advanced reranking algorithms for improved result relevance
- **Hierarchical Processing**: Tree-structured document organization with automatic summarization
- **Multi-Query Expansion**: Enhanced search through query reformulation

### **Comprehensive Knowledge Base**
- **Multi-Source Ingestion**: Automated crawling of documentation websites
- **Smart Processing**: HTML cleaning, markdown conversion, and content optimization
- **Rich Datasets**: Includes Q&A pairs, community discussions, and structured documentation
- **Real-time Updates**: Configurable pipelines for keeping documentation current

### **Multiple Interfaces**
- **Slack Integration**: Full-featured Slack bot with rich formatting
- **CLI Tool**: Command-line interface for development and testing
- **Chat API**: RESTful API for custom integrations

### **Quality & Reliability**
- **Citation System**: All answers include proper source citations
- **Session Management**: Maintains conversation context and history
- **Guardrails**: Built-in safety and quality controls
- **Monitoring**: Comprehensive logging and performance tracking

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│    DocsBot       │───▶│   Expert        │
│  (Slack/CLI)    │    │   Orchestrator   │    │    Agents       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Context &      │    │   Retrieval     │
                       │   Session Mgmt   │    │    System       │
                       └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   ChromaDB      │
                                               │  Vector Store   │
                                               └─────────────────┘
```

### Core Components

- **Expert Agents**: Specialized AI agents for each documentation domain
- **Retrieval System**: Vector search with reranking for optimal results  
- **Storage Layer**: ChromaDB for embeddings, SQLAlchemy for session management
- **Ingestion Pipeline**: Automated document processing and indexing
- **User Interfaces**: Multi-platform support (Slack, CLI, (TODO: API, discord))

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- UV package manager (recommended, get from https://docs.astral.sh/uv/getting-started/installation/) or pip - `pip install uv`

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/parambharat/docsbot.git
   cd docsbot
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Initialize the vector database**
   
   This will create a file called `data/chunked_documents.jsonl` after crawling the documentation sources. This file contains the chunked documents that will be used to store in the vector database

   ```bash
   python -m docsbot.ingestion.ingestion
   ```

   Next, we need to set up the vector database, this will create a vector database in the `data/chromadb` directory
   
   ```bash
   python -m docsbot.ingestion.vectorstore
   ```

### Basic Usage

#### CLI Interface
```bash
# Start interactive chat from the CLI
python -m docsbot.apps.cli --user <your-username>
```
or for the slack bot, you can use the following command:

#### Slack Bot
```bash
# Start the Slack bot
python -m docsbot.apps.slack
```

## Data Pipeline

### Document Processing Workflow

1. **Ingestion**: Crawl documentation websites using sitemaps
2. **Preprocessing**: Clean HTML, convert to markdown, fix encoding
3. **Parsing**: Use tree-sitter for hierarchical document structure
4. **Chunking**: Intelligent chunking with overlap and context preservation
5. **Embedding**: Generate vector embeddings for semantic search
6. **Storage**: Store in ChromaDB with metadata indexing


## 🗂️ Project Structure

```
docsbot/
├── src/docsbot/           # Main application code
│   ├── apps/              # User interfaces (CLI, Slack)
│   ├── chat/              # Chat system and expert agents
│   ├── ingestion/         # Document processing pipeline
│   ├── retriever/         # Search and retrieval system
│   └── storage/           # Database models and clients
├── scripts/               # Utility scripts and pipelines
├── notebooks/             # Jupyter notebooks for experimentation
├── data/                  # Datasets and processed documents
│   ├── chromadb/          # Vector database
│   ├── cache/             # LLM response cache
│   └── *.jsonl            # Processed datasets
└── pyproject.toml         # Project configuration
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CoreWeave**: For hosting the hackathon that inspired this project
- **Weights & Biases**: For the wandb and weave documentation

## 🔗 Links

- [CoreWeave Documentation](https://docs.coreweave.com)
- [Wandb Documentation](https://docs.wandb.ai)
- [Weave Documentation](https://weave-docs.wandb.ai)

---

**Built with ❤️ for the CoreWeave Hackathon**
