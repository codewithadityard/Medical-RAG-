import os
import pickle
import faiss
import xml.etree.ElementTree as ET
import pandas as pd 
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def build_hybrid_indexes():
    documents = []
    
    # 1. Load PubMed Case Reports
    print("Downloading PubMed Case Reports...")
    dataset = load_dataset("findzebra/case-reports", split="train")
    
    for row in dataset:
        abstract_text = " ".join(row['content'])
        full_text = f"Title: {row['title']}\nContent: {abstract_text}"
        documents.append(full_text)
    print(f"Loaded {len(dataset)} PubMed case reports.")

    # 2. Load Orphadata Clinical Definitions
    print("Parsing Orphadata XML...")
    try:
        tree = ET.parse("data/raw/en_product4.xml")
        root = tree.getroot()
        orpha_count = 0
        
        for disorder in root.findall('.//Disorder'):
            orpha_code = disorder.find('OrphaCode').text
            name = disorder.find('Name').text
            text_section = disorder.find('.//TextSection/Contents')
            
            if text_section is not None and text_section.text:
                content = text_section.text.strip()
                doc_string = f"Disease: {name} (ORPHA:{orpha_code}). Clinical Summary: {content}"
                documents.append(doc_string)
                orpha_count += 1
                
        print(f"Loaded {orpha_count} Orphadata clinical definitions.")
    except FileNotFoundError:
        print("Warning: data/raw/en_product4.xml not found. Skipping Orphadata integration.")

    print(f"\nTotal corpus size ready for embedding: {len(documents)} documents.")

    # 3. Generate Hybrid Embeddings
    print("Generating dense embeddings (PubMedBERT)...")
    encoder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    dense_embeddings = encoder.encode(documents, show_progress_bar=True, convert_to_numpy=True)
    
    dimension = dense_embeddings.shape[1]
    faiss.normalize_L2(dense_embeddings)
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(dense_embeddings)

    print("Building sparse keyword index (BM25)...")
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25_index = BM25Okapi(tokenized_docs)

    os.makedirs("data/processed/retriever", exist_ok=True)
    
    print("Saving indexes to disk...")
    faiss.write_index(faiss_index, "data/processed/retriever/dense.faiss")
    with open("data/processed/retriever/sparse.bm25", "wb") as f:
        pickle.dump(bm25_index, f)
    with open("data/processed/retriever/documents.pkl", "wb") as f:
        pickle.dump(documents, f)
        
    print("Hybrid Corpus build complete.")

if __name__ == "__main__":
    build_hybrid_indexes()