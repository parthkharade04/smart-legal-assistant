from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from groq import Groq
from rag_engine import LegalRAG

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------
# CONFIGURATION
# --------------------------
# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Legal AI Assistant API")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag = LegalRAG()

class QueryRequest(BaseModel):
    question: str

class Source(BaseModel):
    text: str
    source: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

@app.on_event("startup")
async def startup_event():
    # Check if we need to ingest documents into Pinecone
    docs_path = "../documents"
    if os.path.exists(docs_path):
        try:
            # Check if Pinecone has data
            stats = rag.index.describe_index_stats()
            count = stats.get('total_vector_count', 0)
            
            if count == 0:
                print(f"Pinecone index is empty (count={count}). Ingesting documents...")
                rag.ingest_documents(docs_path)
            else:
                print(f"Pinecone index ready. Contains {count} vectors.")
                
        except Exception as e:
            print(f"Error checking Pinecone status: {e}")
    else:
        print("Warning: '../documents' folder not found!")

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    # 1. Search for relevant clauses
    results = rag.search(request.question, top_k=5)
    
    if not results:
        return QueryResponse(answer="I couldn't find any relevant clauses in the contracts to answer your question.", sources=[])
    
    # 2. Prepare Context for LLM
    context_text = "\n\n".join([f"Source: {r['source']}\nContent: {r['text']}" for r in results])
    
    prompt = f"""You are an expert legal AI assistant. Use the following context from legal contracts to answer the user's question.
    
    CONTEXT:
    {context_text}
    
    USER QUESTION: 
    {request.question}
    
    INSTRUCTIONS:
    1. Answer strictly based on the provided context.
    2. Cite the specific contract names (Sources) where you found the information.
    3. If the answer is not in the context, say "I cannot find the answer in the provided documents."
    4. Keep the tone professional and concise.
    """
    
    # 3. Generate Answer with Groq (Llama 3)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful legal assistant."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        ai_answer = chat_completion.choices[0].message.content
    except Exception as e:
        ai_answer = f"Error calling Groq AI: {str(e)}"
    
    return QueryResponse(
        answer=ai_answer,  
        sources=[Source(text=r["text"], source=r["source"]) for r in results]
    )

@app.get("/document/{filename}")
async def get_document(filename: str):
    file_path = os.path.join("../documents", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return {"filename": filename, "content": content}

@app.get("/")
def read_root():
    return {"status": "Legal AI API is running with Groq Llama 3"}
