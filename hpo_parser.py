import os
import json

import torch

import obonet

import networkx as nx

from torch_geometric.data import Data



# Configuration

HPO_OBO_URL = "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo"

PROCESSED_DIR = "data/processed"



def parse_hpo_to_pyg():

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Downloading and parsing HPO .obo file (this may take a minute)...")

    

    # obonet natively parses the standard ontology format into a NetworkX MultiDiGraph

    graph = obonet.read_obo(HPO_OBO_URL)

    print(f"Loaded HPO graph: {len(graph)} nodes and {graph.number_of_edges()} edges.")



    # PyG requires integer node indices (0 to N-1). We must maintain a mapping 

    # to translate between PyG indices and real HPO IDs (e.g., HP:0001627)

    nodes = list(graph.nodes())

    hpo_to_idx = {hpo_id: idx for idx, hpo_id in enumerate(nodes)}

    idx_to_hpo = {idx: hpo_id for idx, hpo_id in enumerate(nodes)}

    # converstion of english --> string ids
   # conversion of english --> string ids (Directly from the obonet graph)
    name_to_hpo = {}
    
    for hpo_id, node_data in graph.nodes(data=True):
        # Map the primary official name
        if 'name' in node_data:
            name_to_hpo[node_data['name'].lower()] = hpo_id
            
        # Map all alternate synonyms
        if 'synonym' in node_data:
            for syn in node_data['synonym']:
                # obonet leaves synonyms formatted like: '"Spider fingers" EXACT []'
                # We split by the quote mark to extract just the text
                clean_syn = syn.split('"')[1].lower()
                name_to_hpo[clean_syn] = hpo_id



    # Extract edges ('is_a' relationships)

    edge_sources = []

    edge_targets = []



    for u, v, key in graph.edges(keys=True):

        if key == 'is_a':

            # u is the child, v is the parent. 

            # We map the string IDs to our new integer indices.

            edge_sources.append(hpo_to_idx[u])

            edge_targets.append(hpo_to_idx[v])



    # PyG expects edge_index of shape [2, num_edges]

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)



    # For Day 1, initialize dummy node features (a column of 1s).

    # In Week 2, we will replace this by running the text description of each 

    # HPO node through our PubMedBERT encoder to get dense semantic features.

    num_nodes = len(nodes)

    x = torch.ones((num_nodes, 1), dtype=torch.float)



    # Construct the PyTorch Geometric Data object

    pyg_graph = Data(x=x, edge_index=edge_index)



    # Save artifacts

    graph_path = os.path.join(PROCESSED_DIR, "hpo_pyg_graph.pt")

    torch.save(pyg_graph, graph_path)



    mapping_path = os.path.join(PROCESSED_DIR, "hpo_mapping.json")

    with open(mapping_path, "w") as f:

        json.dump({"hpo_to_idx": hpo_to_idx, "idx_to_hpo": idx_to_hpo,"name_to_hpo":name_to_hpo}, f)



    print(f"Successfully saved PyG graph to {graph_path}")

    print(f"Successfully saved node mapping to {mapping_path}")

    



if __name__ == "__main__":

    parse_hpo_to_pyg() 

