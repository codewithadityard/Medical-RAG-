import os
import json
import torch
import pandas as pd
import urllib.request
from tqdm import tqdm

HPOA_URL = "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa"
PROCESSED_DIR = "data/processed"

def build_disease_bipartite_graph():
    print("Downloading disease annotations (phenotype.hpoa)...")
    hpoa_path = os.path.join(PROCESSED_DIR, "phenotype.hpoa")
    if not os.path.exists(hpoa_path):
        urllib.request.urlretrieve(HPOA_URL, hpoa_path) 
    
    # Load the HPO mapping we created on Day 1
    mapping_path = os.path.join(PROCESSED_DIR, "hpo_mapping.json")
    with open(mapping_path, "r") as f:
        hpo_mapping = json.load(f)["hpo_to_idx"]

    print("Parsing annotations...")
    # The HPOA file has comment lines starting with '#', so we skip them
    df = pd.read_csv(hpoa_path, sep='\t', comment='#', low_memory=False)
    
    # Standardize column names (sometimes the header has a different format)
    # The columns we care about are typically index 0 (Disease ID) and index 3 (HPO ID)
    disease_col = df.columns[0]
    hpo_col = [col for col in df.columns if 'HPO' in col.upper()][0]

    # Filter out negative associations (e.g., "NOT tachycardia")
    if 'Qualifier' in df.columns:
        df = df[df['Qualifier'] != 'NOT']



    unique_diseases = df[disease_col].unique()
    disease_to_idx = {disease: idx for idx, disease in enumerate(unique_diseases)}
    idx_to_disease = {idx: disease for idx, disease in enumerate(unique_diseases)}

    disease_to_name={}
    with open("data/processed/phenotype.hpoa","r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts=line.strip().split("\t")
            if len(parts)>2:
                db_id=parts[0]
                disease_name=parts[1]
                disease_to_name[db_id]=disease_name

    print(f"Total length of disease_to_name dict:{len(disease_to_name)}")


    print(f"Found {len(unique_diseases)} unique diseases.")

    # Build the edges connecting diseases to their symptoms
    disease_indices = []
    hpo_indices = []

    missing_hpo_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Mapping edges"):
        disease_id = row[disease_col]
        hpo_id = row[hpo_col]

        # Ensure the HPO term actually exists in our Day 1 graph
        if hpo_id in hpo_mapping:
            disease_indices.append(disease_to_idx[disease_id])
            hpo_indices.append(hpo_mapping[hpo_id])
        else:
            missing_hpo_count += 1

    print(f"Skipped {missing_hpo_count} outdated HPO terms.")

    # Save as a bipartite edge index (Shape: [2, num_edges])
    # Row 0: Disease nodes, Row 1: HPO nodes
    bipartite_edge_index = torch.tensor([disease_indices, hpo_indices], dtype=torch.long)
    
    # Save the artifacts
    edges_path = os.path.join(PROCESSED_DIR, "disease_hpo_edges.pt")
    torch.save(bipartite_edge_index, edges_path)
    
    disease_mapping_path = os.path.join(PROCESSED_DIR, "disease_mapping.json")
    with open(disease_mapping_path, "w") as f:
        json.dump({"disease_to_idx": disease_to_idx, "idx_to_disease": idx_to_disease,"disease_to_name":disease_to_name}, f)

    print(f"Successfully saved bipartite edges to {edges_path}")
    print(f"Successfully saved disease mapping to {disease_mapping_path}")

if __name__ == "__main__":
    build_disease_bipartite_graph()