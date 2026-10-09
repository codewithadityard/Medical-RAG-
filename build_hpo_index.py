import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_hpo_vector_index():
    print("Loading HPO mapping...")
    with open("data/processed/hpo_mapping.json", "r") as f:
        hpo_map = json.load(f)
        name_to_hpo = hpo_map["name_to_hpo"]

    # All unique symptom names and synonyms in the ontology (~30,000 terms)
    symptom_names = list(name_to_hpo.keys())
    print(f"Total HPO terms & synonyms to embed: {len(symptom_names)}")

    # Load free medical SapBERT model
    print("Loading SapBERT model (cambridgeltl/SapBERT-from-PubMedBERT-fulltext)...")
    model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

    # Generate embeddings (runs fast in batches)
    print("Embedding HPO terms...")
    embeddings = model.encode(
        symptom_names,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalizing allows Cosine Similarity via Inner Product
    )

    # Build FAISS Index (Cosine similarity using IndexFlatIP)
    dim = embeddings.shape[1]  # 768 dimensions
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # Save to disk
    os.makedirs("data/processed", exist_ok=True)
    faiss.write_index(index, "data/processed/hpo_faiss.index")

    # Save the ordered list of names corresponding to the FAISS index rows
    with open("data/processed/hpo_terms_list.json", "w") as f:
        json.dump(symptom_names, f)

    print("Successfully built and saved HPO FAISS index!")


if __name__ == "__main__":
    build_hpo_vector_index()