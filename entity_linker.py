import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class MedicalEntityLinker:
    def __init__(self, threshold=0.70):
        self.threshold = threshold

        # 1. Load the precomputed FAISS index & terms list
        self.index = faiss.read_index("data/processed/hpo_faiss.index")
        with open("data/processed/hpo_terms_list.json", "r") as f:
            self.terms_list = json.load(f)

        with open("data/processed/hpo_mapping.json", "r") as f:
            self.hpo_map = json.load(f)
            self.name_to_hpo = self.hpo_map["name_to_hpo"]
            self.hpo_to_idx = self.hpo_map["hpo_to_idx"]

        # 2. Load SapBERT for embedding incoming queries (lightweight on CPU)
        self.model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

    def link_symptom(self, user_symptom: str):
        """
        Maps any user phrase / typo to official HPO term and graph index.
        Returns: {hpo_id, graph_idx, matched_term, score} or None
        """
        cleaned_text = user_symptom.strip().lower()

        # Quick check: If exact match exists, return immediately (0 ms)
        if cleaned_text in self.name_to_hpo:
            hpo_id = self.name_to_hpo[cleaned_text]
            return {
                "hpo_id": hpo_id,
                "graph_idx": self.hpo_to_idx[hpo_id],
                "matched_term": cleaned_text,
                "score": 1.0,
            }

        # Semantic Vector Search via SapBERT (~5 ms)
        query_vec = self.model.encode([cleaned_text], normalize_embeddings=True)
        scores, indices = self.index.search(query_vec.astype(np.float32), k=1)

        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])
        matched_term = self.terms_list[best_idx]
        hpo_id = self.name_to_hpo[matched_term]

        # Safety Confidence Threshold
        if best_score >= self.threshold:
            return {
                "hpo_id": hpo_id,
                "graph_idx": self.hpo_to_idx[hpo_id],
                "matched_term": matched_term,
                "score": round(best_score, 3),
            }
        else:
            print(f"Warning: '{user_symptom}' rejected (similarity {best_score:.2f} < threshold {self.threshold})")
            return None


if __name__ == "__main__":
    linker = MedicalEntityLinker(threshold=0.70)
    test_cases = [
        "arachnodactyly",         # Exact match
        "arachnodactily",         # Typo
        "spider fingers",         # Colloquial synonym
        "sunken breastbone",      # Layman term
        "abnormal heart rhythm",  # Clinical phrase
        "totally random text 123" # Out-of-vocabulary / irrelevant
    ]

    print("\n--- Testing MedicalEntityLinker ---")
    for symp in test_cases:
        res = linker.link_symptom(symp)
        if res:
            print(f"'{symp}' ➔ Matched: '{res['matched_term']}' (HPO: {res['hpo_id']}, Score: {res['score']})")
        else:
            print(f"'{symp}' ➔ No match (Below threshold)")