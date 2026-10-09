import os
import json
import torch
import pandas as pd
import urllib.request
from tqdm import tqdm

HPOA_URL = "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa"
PROCESSED_DIR = "data/processed"


def parse_frequency(freq_val):
    """Converts HPOA frequency fractions, percentages, or ontology codes to a [0.0, 1.0] float."""
    if pd.isna(freq_val):
        return 0.5
    s = str(freq_val).strip()
    if '/' in s:
        try:
            num, den = s.split('/')
            return float(num) / float(den) if float(den) > 0 else 0.5
        except (ValueError, ZeroDivisionError):
            return 0.5
    if '%' in s:
        try:
            return float(s.replace('%', '')) / 100.0
        except ValueError:
            return 0.5
    # Standard HPO frequency codes
    freq_map = {
        'HP:0040280': 1.0,   # Obligate (100%)
        'HP:0040281': 0.9,   # Very frequent (80-99%)
        'HP:0040282': 0.5,   # Frequent (30-79%)
        'HP:0040283': 0.15,  # Occasional (5-29%)
        'HP:0040284': 0.025  # Very rare (1-4%)
    }
    return freq_map.get(s, 0.5)


def parse_onset(onset_val):
    """Encodes clinical onset age as an ordinal float in [0.0, 1.0]."""
    if pd.isna(onset_val):
        return 0.5
    s = str(onset_val).strip()
    onset_map = {
        'HP:0003577': 0.1,  # Congenital / Antenatal
        'HP:0003623': 0.2,  # Neonatal
        'HP:0003593': 0.3,  # Infantile
        'HP:0011463': 0.5,  # Childhood
        'HP:0003621': 0.7,  # Juvenile
        'HP:0003581': 0.9,  # Adult
    }
    return onset_map.get(s, 0.5)


def build_disease_bipartite_graph():
    print("Downloading disease annotations (phenotype.hpoa)...")
    hpoa_path = os.path.join(PROCESSED_DIR, "phenotype.hpoa")
    if not os.path.exists(hpoa_path):
        urllib.request.urlretrieve(HPOA_URL, hpoa_path) 
    
    # Load the HPO mapping
    mapping_path = os.path.join(PROCESSED_DIR, "hpo_mapping.json")
    with open(mapping_path, "r") as f:
        hpo_mapping = json.load(f)["hpo_to_idx"]

    print("Parsing annotations...")
    df = pd.read_csv(hpoa_path, sep='\t', comment='#', low_memory=False)
    
    disease_col = df.columns[0]
    hpo_col = [col for col in df.columns if 'HPO' in col.upper()][0]

    # Filter out negative associations (e.g., "NOT tachycardia")
    if 'qualifier' in df.columns:
        df = df[df['qualifier'] != 'NOT']
    elif 'Qualifier' in df.columns:
        df = df[df['Qualifier'] != 'NOT']

    unique_diseases = df[disease_col].unique()
    disease_to_idx = {disease: idx for idx, disease in enumerate(unique_diseases)}
    idx_to_disease = {idx: disease for idx, disease in enumerate(unique_diseases)}

    disease_to_name = {}
    with open(hpoa_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) > 2:
                db_id = parts[0]
                disease_name = parts[1]
                disease_to_name[db_id] = disease_name

    print(f"Total length of disease_to_name dict: {len(disease_to_name)}")
    print(f"Found {len(unique_diseases)} unique diseases.")

    # Build the edges connecting diseases to their symptoms along with attributes
    disease_indices = []
    hpo_indices = []
    edge_freqs = []
    edge_onsets = []

    freq_col = 'frequency' if 'frequency' in df.columns else ('Frequency' if 'Frequency' in df.columns else None)
    onset_col = 'onset' if 'onset' in df.columns else ('Onset' if 'Onset' in df.columns else None)

    missing_hpo_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Mapping edges & attributes"):
        disease_id = row[disease_col]
        hpo_id = row[hpo_col]

        if hpo_id in hpo_mapping:
            disease_indices.append(disease_to_idx[disease_id])
            hpo_indices.append(hpo_mapping[hpo_id])
            edge_freqs.append(parse_frequency(row[freq_col] if freq_col else None))
            edge_onsets.append(parse_onset(row[onset_col] if onset_col else None))
        else:
            missing_hpo_count += 1

    print(f"Skipped {missing_hpo_count} outdated HPO terms.")

    # Save as bipartite edge index [2, num_edges]
    bipartite_edge_index = torch.tensor([disease_indices, hpo_indices], dtype=torch.long)
    edges_path = os.path.join(PROCESSED_DIR, "disease_hpo_edges.pt")
    torch.save(bipartite_edge_index, edges_path)

    # Save edge attributes [num_edges, 2] (Frequency, Onset)
    edge_attr = torch.tensor(list(zip(edge_freqs, edge_onsets)), dtype=torch.float)
    attr_path = os.path.join(PROCESSED_DIR, "disease_hpo_edge_attr.pt")
    torch.save(edge_attr, attr_path)

    disease_mapping_path = os.path.join(PROCESSED_DIR, "disease_mapping.json")
    with open(disease_mapping_path, "w") as f:
        json.dump({"disease_to_idx": disease_to_idx, "idx_to_disease": idx_to_disease, "disease_to_name": disease_to_name}, f)

    print(f"Successfully saved bipartite edges to {edges_path}")
    print(f"Successfully saved edge attributes [freq, onset] to {attr_path}")
    print(f"Successfully saved disease mapping to {disease_mapping_path}")


if __name__ == "__main__":
    build_disease_bipartite_graph()