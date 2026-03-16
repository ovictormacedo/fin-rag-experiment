import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSService:
    def __init__(self):
        self.number_of_documents_to_retrieve = 5
        self.index = faiss.read_index("data/faiss_index")
        self.metadata = []
        with open("data/faiss_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def get_context(self, query):
        vec = self.embedder.encode([query.strip()], normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32)
        scores, indices = self.index.search(vec, min(self.number_of_documents_to_retrieve, self.index.ntotal))
        
        texts = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            print(f"  doc {i}: index={idx} score={float(score):.6f}")
            
            if float(score) < 0.4:
                continue

            if 0 <= idx < len(self.metadata) and self.metadata[idx].get("text"):
                texts.append(self.metadata[idx]["text"])

        return "\n\n".join(texts)