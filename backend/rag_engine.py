import os
import glob
import pickle
import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class LegalRAG:
    def __init__(self, index_path="vector.index", metadata_path="metadata.pkl", model_name="all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        
        # Load existing index if available
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load_index()

    def load_index(self):
        """Load FAISS index and metadata from disk."""
        print("Loading existing vector index...")
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            self.chunks = pickle.load(f)

    def ingest_documents(self, doc_folder: str):
        """Read text files, chunk them, and build FAISS index."""
        print(f"Ingesting documents from {doc_folder}...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        new_chunks = []
        # Support .txt for now (we cleaned the folder to only have .txt)
        files = glob.glob(os.path.join(doc_folder, "*.txt"))
        
        for file_path in files:
            file_name = os.path.basename(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Create chunks with metadata
            splits = text_splitter.split_text(text)
            for split in splits:
                new_chunks.append({
                    "text": split,
                    "source": file_name
                })
        
        if not new_chunks:
            print("No documents found to ingest.")
            return

        # Embed and Index
        print(f"Embedding {len(new_chunks)} chunks...")
        embeddings = self.model.encode([c["text"] for c in new_chunks])
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        self.chunks = new_chunks
        
        # Save to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print("Ingestion complete.")

    def search(self, query: str, top_k=3) -> List[Dict]:
        """Search the index for relevant clauses."""
        if not self.index:
            return []
            
        query_vec = self.model.encode([query])
        distances, ids = self.index.search(np.array(query_vec).astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(ids[0]):
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

if __name__ == "__main__":
    # Test run
    rag = LegalRAG()
    # Assuming documents are one level up in 'documents' since we run this from backend/ usually
    # But wait, our structure is root/documents and root/backend. 
    # So path should be ../documents
    rag.ingest_documents("../documents")
    print(rag.search("termination"))
