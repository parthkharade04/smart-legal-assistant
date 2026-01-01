from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
# from google import genai # Removed (Using Groq)
from groq import Groq
from rag_engine import LegalRAG
from dotenv import load_dotenv

load_dotenv()

# App Definition
app = FastAPI(title="Legal AI Assistant API")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Variables (Lazy Loading)
client = None
rag = None

@app.on_event("startup")
async def startup_event():
    global client, rag
    print("Starup: Initializing services...")
    
    # 1. Initialize Groq
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            client = Groq(api_key=api_key)
            print("Groq Client initialized.")
        else:
            print("Warning: GROQ_API_KEY not found.")
    except Exception as e:
        print(f"Failed to init Groq: {e}")

    # 2. Initialize RAG Engine
    try:
        rag = LegalRAG()
        print("RAG Engine initialized.")
        
        # Check Pinecone Stats
        try:
            stats = rag.index.describe_index_stats()
            print(f"Pinecone Stats: {stats}")
        except:
            print("Could not fetch Pinecone stats.")
            
    except Exception as e:
        print(f"Failed to init RAG Engine: {e}")
        # We do NOT raise exception here, so app can start and return 500 cleanly if needed.

class QueryRequest(BaseModel):
    question: str

class Source(BaseModel):
    text: str
    source: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not rag or not client:
         raise HTTPException(status_code=503, detail="AI Services not initialized. Check server logs.")

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
    # Security: Prevent traversing up directories
    if ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join("../documents", filename)
    if not os.path.exists(file_path):
        # On Cloud, documents might not exist locally, so we return a placeholder or 404
        # Ideally, we should fetch from S3 or database, but for now 404 is correct.
        raise HTTPException(status_code=404, detail="Document content not available in cloud deployment (Privacy Mode).")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return {"filename": filename, "content": content}

@app.get("/")
def read_root():
    return {"status": "Legal AI API is running with Groq + Pinecone"}
