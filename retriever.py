import pickle
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

class MedicalHybridRetriever:
    def __init__(self, index_dir="data/processed/retriever"):
        # Load artifacts
        self.faiss_index = faiss.read_index(f"{index_dir}/dense.faiss")
        
        with open(f"{index_dir}/sparse.bm25", "rb") as f:
            self.bm25_index = pickle.load(f)
            
        with open(f"{index_dir}/documents.pkl", "rb") as f:
            self.documents = pickle.load(f)
            
        # Load Models
        self.bi_encoder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
        # Cross-encoder for final reranking
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _rrf(self, dense_ranks, sparse_ranks, k=60):
        """Reciprocal Rank Fusion"""
        rrf_scores = {}
        
        for rank, doc_idx in enumerate(dense_ranks):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
            
        for rank, doc_idx in enumerate(sparse_ranks):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
            
        # Sort by highest RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_idx for doc_idx, score in sorted_docs]

    def search(self, query, top_k=5):
        # 1. Dense Search
        query_emb = self.bi_encoder.encode([query])
        faiss.normalize_L2(query_emb)
        _, dense_indices = self.faiss_index.search(query_emb, 20)
        dense_ranks = dense_indices[0].tolist()

        # 2. Sparse Search
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25_index.get_scores(tokenized_query)
        sparse_ranks = sparse_scores.argsort()[::-1][:20].tolist()

        # 3. Fuse Ranks (RRF)
        fused_indices = self._rrf(dense_ranks, sparse_ranks)[:10]
        candidate_docs = [self.documents[i] for i in fused_indices]

        # 4. Cross-Encoder Reranking
        # Bi-encoders compress to a single vector. Cross-encoders look at the query 
        # and document together for deeper contextual attention.
        cross_inp = [[query, doc] for doc in candidate_docs]
        cross_scores = self.cross_encoder.predict(cross_inp)
        
        # Sort final results by cross-encoder score
        reranked_pairs = sorted(zip(cross_scores, candidate_docs), reverse=True)
        return [doc for score, doc in reranked_pairs[:top_k]]
