import os
import json
import urllib.request
import torch
import obonet
import networkx as nx
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer

# Configuration
HPO_OBO_URL = "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo"
PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw"


def parse_hpo_to_pyg():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    obo_path = os.path.join(RAW_DIR, "hp.obo")
    if not os.path.exists(obo_path):
        print("Downloading HPO .obo file...")
        urllib.request.urlretrieve(HPO_OBO_URL, obo_path)

    print("Parsing HPO .obo file...")
    graph = obonet.read_obo(obo_path)
    print(f"Loaded HPO graph: {len(graph)} nodes and {graph.number_of_edges()} edges.")

    # PyG requires integer node indices (0 to N-1)
    nodes = list(graph.nodes())
    hpo_to_idx = {hpo_id: idx for idx, hpo_id in enumerate(nodes)}
    idx_to_hpo = {idx: hpo_id for idx, hpo_id in enumerate(nodes)}

    # Map names and synonyms to HPO IDs
    name_to_hpo = {}
    for hpo_id, node_data in graph.nodes(data=True):
        if 'name' in node_data:
            name_to_hpo[node_data['name'].lower()] = hpo_id

        if 'synonym' in node_data:
            for syn in node_data['synonym']:
                clean_syn = syn.split('"')[1].lower() if '"' in syn else syn.lower()
                name_to_hpo[clean_syn] = hpo_id

    # Extract edges ('is_a' taxonomic relationships)
    edge_sources = []
    edge_targets = []
    for u, v, key in graph.edges(keys=True):
        if key == 'is_a':
            edge_sources.append(hpo_to_idx[u])
            edge_targets.append(hpo_to_idx[v])

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

    # Generate 768-dimensional SapBERT semantic embeddings for each HPO term definition
    print("Generating 768-dimensional SapBERT semantic embeddings for HPO nodes...")
    texts_to_encode = []
    for hpo_id in nodes:
        node_data = graph.nodes[hpo_id]
        name = node_data.get('name', '')
        defn = node_data.get('def', '')
        clean_def = defn.split('"')[1] if defn and '"' in defn else (defn or '')
        # Keep clean concise definition text
        clean_def = clean_def[:200]
        desc = f"{name}: {clean_def}".strip(": ")
        texts_to_encode.append(desc if desc else hpo_id)

    model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model.max_seq_length = 64  # Fast, optimal sequence length for medical entity representation

    embeddings = model.encode(
        texts_to_encode,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    x = torch.tensor(embeddings, dtype=torch.float)

    # Construct the PyTorch Geometric Data object with semantic embeddings
    pyg_graph = Data(x=x, edge_index=edge_index)

    # Save artifacts
    graph_path = os.path.join(PROCESSED_DIR, "hpo_pyg_graph.pt")
    torch.save(pyg_graph, graph_path)

    mapping_path = os.path.join(PROCESSED_DIR, "hpo_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({"hpo_to_idx": hpo_to_idx, "idx_to_hpo": idx_to_hpo, "name_to_hpo": name_to_hpo}, f)

    print(f"Successfully saved PyG graph with 768-dim SapBERT features to {graph_path}")
    print(f"Successfully saved node mapping to {mapping_path}")


if __name__ == "__main__":
    parse_hpo_to_pyg()
