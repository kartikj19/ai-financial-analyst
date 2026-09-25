# 📊 AI Financial Analyst

A domain-focused multi-PDF Retrieval-Augmented Generation (RAG) application for financial reports. Upload annual reports, quarterly results or investor presentations and ask grounded questions with page-level source attribution.

## Features

- Multi-PDF upload and indexing
- Local Sentence Transformers embeddings
- FAISS semantic vector search
- Google Gemini for grounded answer generation
- Financial-domain system prompt and hallucination guardrails
- Source document + page attribution
- Conversational follow-up questions
- Document summary mode
- Configurable chunk size, overlap and retrieval depth
- Streamlit interface
- Basic automated test

## Architecture

```text
PDFs
  ↓
PyPDFLoader
  ↓
RecursiveCharacterTextSplitter
  ↓
Sentence Transformers Embeddings
  ↓
FAISS Vector Store
  ↓
Semantic Retrieval (Top-K)
  ↓
Financial Analyst Prompt + Context
  ↓
Google Gemini
  ↓
Answer + Sources
```

## Prerequisites

- Windows/macOS/Linux
- Python 3.11 or 3.12 recommended
- 8 GB RAM minimum; 16 GB recommended for smoother local embeddings
- Internet connection for the first model download and Gemini API calls
- Google Gemini API key

A dedicated GPU is **not required**. The embedding model can run on CPU; a GTX 1650 can optionally accelerate local ML workloads depending on the PyTorch installation.

## Windows setup

```powershell
cd ai-financial-analyst
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and set:

```env
GOOGLE_API_KEY=your_key_here
```

Then run:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit.

## Git setup

```bash
git init
git add .
git commit -m "Initial AI Financial Analyst RAG project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Never commit `.env` or API keys.

## How to use

1. Upload one or more PDFs.
2. Click **Index Documents**.
3. Ask questions such as:
   - What was revenue in FY2025?
   - How did operating margin change year over year?
   - What risks does management identify?
   - Summarize the company's guidance.
4. Expand **Sources** to inspect supporting pages.
5. Use **Summarize documents** for a high-level analyst-style overview.

## Resume-ready description

**AI Financial Analyst — RAG, LangChain, FAISS, Gemini, Streamlit**

- Built a domain-specific Retrieval-Augmented Generation application that indexes multiple financial PDFs using Sentence Transformer embeddings and FAISS semantic search.
- Developed a LangChain-based question-answering pipeline with Google Gemini, contextual conversation history, and source/page attribution for grounded responses.
- Implemented configurable chunking and Top-K retrieval plus financial-domain guardrails to reduce unsupported answers and preserve reporting periods, units and currencies.
- Delivered an interactive Streamlit interface for multi-document upload, semantic Q&A and document-level financial summaries.

Only claim metrics such as latency, accuracy or hallucination reduction after you actually measure them.

## Important attribution

This project was inspired by the educational `alejandro-ao/ask-multiple-pdfs` repository, but the implementation in this folder is a separate customized application with a financial domain, modern package layout, Gemini integration, local embeddings, source attribution and additional UI features.

## Limitations

- Scanned/image-only PDFs require OCR and are not fully supported by the default text extractor.
- Tables may not extract perfectly from complex reports.
- Gemini API usage can incur costs depending on the provider's current pricing and quota.
- This is a research assistant, not a financial adviser.
