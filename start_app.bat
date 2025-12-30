@echo off
echo ===================================================
echo   STARTING LEGAL AI ASSISTANT (Global Env)
echo ===================================================

echo 1. Installing lightweight dependencies (FastAPI, Uvicorn) if missing...
pip install fastapi uvicorn python-multipart sentence-transformers faiss-cpu langchain langchain-community

echo 2. Starting Backend API (FastAPI) on Port 8000...
start "LegalAI Backend" cmd /k "cd backend && uvicorn main:app --reload --port 8000"

echo 3. Starting Frontend UI (React) on Port 5173...
cd frontend
npm run dev
