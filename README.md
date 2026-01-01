# ⚖️ Domain-Grounded Smart Legal Insight Engine

A high-performance RAG (Retrieval-Augmented Generation) application designed to analyze legal contracts and answer questions with citation-backed accuracy.

**Enterprise-Ready Architecture:** Built with **React** (Glassmorphism UI), **FastAPI**, **Pinecone Vector Cloud**, and powered by **Meta Llama 3** (via Groq) for lightning-fast inference.

## 🚀 Features

- **Document Analysis**: Ingests and indexes real-world legal contracts (PDF/TXT) into Pinecone Cloud.
- **Smart Retrieval**: Uses **Pinecone Vector Database** for scalable, millisecond-latency searches.
- **AI-Powered Synthesis**: Integrates **Llama 3-70B** (via Groq) to generate lawyer-like tables, lists, and summaries.
- **Rich Text Answers**: Renders responses with Markdown (Bold, Italics, Tables).
- **Evidence Highlighting**: Automatically highlights key terms in source documents for rapid verification.
- **Deep Dive Mode**: Instantly read full contract texts in a secure modal.

## 🛠️ Tech Stack

- **Frontend**: React.js, Vite, React Markdown, CSS Modules (Glassmorphism).
- **Backend**: FastAPI (Python), Uvicorn.
- **AI/LLM**: **Groq API** (running Llama 3-70B Versatile).
- **Vector DB**: **Pinecone** (Serverless Cloud Vector Search).
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`) - Local.

## 📸 Screenshots

| **Dashboard Interface** | **AI-Generated Answer** |
|:---:|:---:|
| ![Dashboard](screenshots/dashboard.png) | ![LLM Answer](screenshots/llm-generated-ans.png) |
| **Context Retrieval (Pinecone/FAISS)** | **Backend Logs** |
| ![FAISS Match](screenshots/faiss.png) | ![Uvicorn Logs](screenshots/main%20logs.png) |

*(See more in the `screenshots/` folder)*

## ⚡ Quick Start

### Prerequisites
- Python 3.9+
- Node.js & npm
- [Groq API Key](https://console.groq.com/)
- [Pinecone API Key](https://www.pinecone.io/)

### 1. Clone the Repo
```bash
git clone https://github.com/parthkharade04/smart-legal-assistant.git
cd smart-legal-assistant
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
# Activate venv: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
pip install -r requirements.txt

# Set up your Environment Variables
# Create a .env file in /backend with:
# GROQ_API_KEY=your_key_here
# PINECONE_API_KEY=your_key_here
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 4. Run the App
- **Backend**: `uvicorn main:app --reload --port 8000`
- **Frontend**: Open `http://localhost:3000`

## 🧠 How It Works (Cloud RAG)

1. **Ingestion**: Uploaded contracts are chunked and embedded locally.
2. **Indexing**: Vectors are uploaded to **Pinecone Cloud (Index: legal-contracts)**.
3. **Retrieval**: When you ask a question, the system queries Pinecone for the Top 5 matches.
4. **Generation**: The retrieved context + user question is sent to **Groq (Llama 3)**, which synthesizes a grounded answer in milliseconds.

## 🛡️ License
MIT License.
