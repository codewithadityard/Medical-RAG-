import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from gnn import DiagnosticGNN


def load_graph_data(device):
    """Loads the graph tensors and clinical edge attributes into PyTorch dictionaries."""
    # 1. Load the raw tensors
    hpo_graph = torch.load("data/processed/hpo_pyg_graph.pt")
    bipartite_edges = torch.load("data/processed/disease_hpo_edges.pt")
    edge_attr = torch.load("data/processed/disease_hpo_edge_attr.pt")

    # 2. Map them to the specific HGT edge relations
    edge_index_dict = {
        ('symptom', 'is_a', 'symptom'): hpo_graph.edge_index.to(device),
        ('disease', 'has', 'symptom'): bipartite_edges.to(device),
        ('symptom', 'rev_has', 'disease'): bipartite_edges.flip([0]).to(device)
    }

    edge_attr_dict = {
        ('disease', 'has', 'symptom'): edge_attr.to(device)
    }

    num_symptoms = hpo_graph.x.shape[0]
    num_diseases = bipartite_edges[0].max().item() + 1

    return edge_index_dict, edge_attr_dict, num_symptoms, num_diseases, bipartite_edges


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Load data
    edge_index_dict, edge_attr_dict, num_symptoms, num_diseases, bipartite_edges = load_graph_data(device)

    # Initialize model and optimizer
    model = DiagnosticGNN(num_diseases, num_symptoms, hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Pre-calculate which symptoms belong to which disease for fast sampling
    disease_to_symptoms = {}
    for d_idx, s_idx in zip(bipartite_edges[0].tolist(), bipartite_edges[1].tolist()):
        disease_to_symptoms.setdefault(d_idx, []).append(s_idx)

    epochs = 10
    batch_size = 128

    print(f"Starting Heterogeneous Graph Transformer training ({epochs} epochs)...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Full Graph Forward Pass with clinical edge attributes
        updated_node_embeddings = model(edge_index_dict, edge_attr_dict)
        
        # 2. Synthetic Patient Generation (Batching)
        target_diseases = random.choices(list(disease_to_symptoms.keys()), k=batch_size)
        target_tensor = torch.tensor(target_diseases, dtype=torch.long).to(device)
        d_norm = F.normalize(updated_node_embeddings['disease'], p=2, dim=-1)

        batch_scores = []
        for disease_idx in target_diseases:
            true_symptoms = disease_to_symptoms[disease_idx]
            
            # Simulate realistic patient presentation (keeping 40% to 80% of true symptoms)
            keep_ratio = random.uniform(0.4, 0.8)
            num_to_keep = max(1, int(len(true_symptoms) * keep_ratio))
            patient_symptoms = random.sample(true_symptoms, num_to_keep)
            patient_tensor = torch.tensor(patient_symptoms, dtype=torch.long).to(device)

            patient_vecs = updated_node_embeddings['symptom'][patient_tensor]
            patient_profile = patient_vecs.mean(dim=0, keepdim=True)
            patient_norm = F.normalize(patient_profile, p=2, dim=-1)
            
            # Temperature-scaled cosine similarity for stable multiclass cross-entropy
            scores = torch.matmul(patient_norm, d_norm.t()) / 0.05
            batch_scores.append(scores)

        # 3. Calculate Loss and Backpropagate
        logits = torch.cat(batch_scores, dim=0) 
        loss = criterion(logits, target_tensor)
        loss.backward()
        optimizer.step()

        top5 = torch.topk(logits, 5, dim=1).indices
        top1_acc = (top5[:, 0] == target_tensor).float().mean().item() * 100
        top5_acc = (top5 == target_tensor.unsqueeze(1)).any(dim=1).float().mean().item() * 100

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {loss.item():.4f} | Batch Top-1: {top1_acc:.1f}% | Batch Top-5: {top5_acc:.1f}%")

    print("\nTraining complete! Saving model...")
    os.makedirs("data/models", exist_ok=True)
    torch.save(model.state_dict(), "data/models/gnn_ranker.pth")
    print("Model saved to data/models/gnn_ranker.pth")


if __name__ == "__main__":
    train()