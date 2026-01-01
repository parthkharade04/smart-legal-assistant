import os
import glob
from typing import List, Dict, Any
# from sentence_transformers import SentenceTransformer # Removed for Memory Efficiency
from fastembed import TextEmbedding
from pinecone import Pinecone
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LegalRAG:
    def __init__(self, index_name="legal-contracts", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the RAG engine with Pinecone Vector Database.
        """
        print("Initializing LegalRAG with Pinecone...")
        
        # 1. Initialize Embeddings Model (FastEmbed - Lightweight)
        print(f"Loading FastEmbed model: {model_name}...")
        self.model = TextEmbedding(model_name=model_name)
        
        # 2. Initialize Pinecone (Cloud DB)
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("Warning: PINECONE_API_KEY not found in .env settings (OK for CI/CD checks)")
            self.pc = None
            self.index = None
            return
            
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = self.pc.Index(index_name)
        
        # Check connection
        try:
            stats = self.index.describe_index_stats()
            print(f"Connected to Pinecone Index '{index_name}'. Stats: {stats}")
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}")

    def ingest_documents(self, doc_folder: str, batch_size=50):
        """
        Reads all .txt files, chunks them, embeds them, and uploads to Pinecone.
        """
        print(f"Scanning {doc_folder} for documents...")
        files = glob.glob(os.path.join(doc_folder, "*.txt"))
        
        if not files:
            print("No documents found.")
            return

        all_chunks = []
        all_ids = []
        all_metadata = []
        
        print(f"Found {len(files)} files. Starting processing...")
        
        # 1. Process files into chunks
        total_chunks = 0
        for i, file_path in enumerate(files):
            file_name = os.path.basename(file_path)
            
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            
            # Simple chunking by paragraphs or size
            chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 50]
            
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{file_name}_{idx}"
                chunk_id = chunk_id.encode("ascii", "ignore").decode()
                
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadata.append({
                    "text": chunk,
                    "source": file_name,
                    "chunk_id": idx
                })
                total_chunks += 1
                
        print(f"Total extracted chunks: {total_chunks}")
        
        # 2. Embed and Upsert in Batches to Pinecone
        for i in range(0, len(all_chunks), batch_size):
            batch_end = min(i + batch_size, len(all_chunks))
            
            batch_texts = all_chunks[i:batch_end]
            batch_ids = all_ids[i:batch_end]
            batch_meta = all_metadata[i:batch_end]
            
            # Generate Embeddings (FastEmbed API)
            # FastEmbed.embed returns a generator, convert to list
            embeddings = list(self.model.embed(batch_texts))
            # FastEmbed returns numpy arrays, convert to lists for Pinecone
            embeddings = [e.tolist() for e in embeddings]
            
            # Prepare vectors for Pinecone
            vectors_to_upsert = []
            for j in range(len(batch_texts)):
                vectors_to_upsert.append({
                    "id": batch_ids[j],
                    "values": embeddings[j],
                    "metadata": batch_meta[j]
                })
            
            # Upsert
            try:
                self.index.upsert(vectors=vectors_to_upsert)
                print(f"Uploaded batch {i}-{batch_end} to Pinecone.")
            except Exception as e:
                print(f"Error uploading batch: {e}")
                
            time.sleep(0.5)

        print("Ingestion Complete! Data is now in the Cloud.")

    def search(self, query: str, top_k=5) -> List[Dict]:
        """
        Embeds the query and searches Pinecone for similar vectors.
        """
        if not self.index:
            return []

        # 1. Embed Query (FastEmbed)
        # embed returns generator, get first item, convert to list
        query_embedding = list(self.model.embed([query]))[0].tolist()
        
        # 2. Query Pinecone
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # 3. Format Results
            formatted_results = []
            for match in results['matches']:
                formatted_results.append({
                    "text": match['metadata']['text'],
                    "source": match['metadata']['source'],
                    "score": match['score']
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Search failed: {e}")
            return []
            
    def load_index(self):
        """
        No-op for Cloud DB (it's always loaded).
        We just check stats.
        """
        print(f"Checking Pinecone Index '{self.index_name}' status...")
        try:
            if self.index:
                stats = self.index.describe_index_stats()
                print(f"Index contains {stats['total_vector_count']} vectors.")
        except:
            print("Could not fetch stats.")
