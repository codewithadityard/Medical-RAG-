import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# We will import the model we wrote previously
from gnn import DiagnosticGNN

def load_graph_data(device):
    """Loads the artifacts from Day 1 and Day 2 into PyTorch dictionaries."""
    # 1. Load the raw tensors
    hpo_graph = torch.load("data/processed/hpo_pyg_graph.pt")
    bipartite_edges = torch.load("data/processed/disease_hpo_edges.pt")

    # 2. Map them to the specific HeteroConv edge names we defined
    edge_index_dict = {
        ('symptom', 'is_a', 'symptom'): hpo_graph.edge_index.to(device),
        ('disease', 'has', 'symptom'): bipartite_edges.to(device),
        # The crucial reverse edge we discussed (flipping row 0 and 1)
        ('symptom', 'rev_has', 'disease'): bipartite_edges.flip([0]).to(device)
    }

    # Count nodes for the Embedding layers
    num_symptoms = hpo_graph.x.shape[0]
    num_diseases = bipartite_edges[0].max().item() + 1

    return edge_index_dict, num_symptoms, num_diseases, bipartite_edges

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Load data
    edge_index_dict, num_symptoms, num_diseases, bipartite_edges = load_graph_data(device)

    # Initialize model and optimizer
    model = DiagnosticGNN(num_diseases, num_symptoms, hidden_dim=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Pre-calculate which symptoms belong to which disease for fast sampling
    disease_to_symptoms = {}
    for d_idx, s_idx in zip(bipartite_edges[0].tolist(), bipartite_edges[1].tolist()):
        if d_idx not in disease_to_symptoms:
            disease_to_symptoms[d_idx] = []
        disease_to_symptoms[d_idx].append(s_idx)

    epochs = 50
    batch_size = 256

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        # 1. Full Graph Forward Pass: Update ALL node embeddings
        optimizer.zero_grad()
        updated_node_embeddings = model(edge_index_dict)
        
        # 2. Synthetic Patient Generation (Batching)
        # Randomly select a batch of target diseases
        target_diseases = random.choices(list(disease_to_symptoms.keys()), k=batch_size)
        target_tensor = torch.tensor(target_diseases, dtype=torch.long).to(device)

        batch_scores = []
        for disease_idx in target_diseases:
            true_symptoms = disease_to_symptoms[disease_idx]
            
            # Simulate a patient by keeping only 40% to 80% of the true symptoms
            keep_ratio = random.uniform(0.4, 0.8)
            num_to_keep = max(1, int(len(true_symptoms) * keep_ratio))
            patient_symptoms = random.sample(true_symptoms, num_to_keep)
            patient_tensor = torch.tensor(patient_symptoms, dtype=torch.long).to(device)

            # 3. Score this synthetic patient against ALL diseases
            # (Reusing the predict_diagnosis logic we wrote)
            patient_vecs = updated_node_embeddings['symptom'][patient_tensor]
            patient_profile = patient_vecs.mean(dim=0, keepdim=True)
            
            # Dot product against all diseases
            scores = torch.matmul(patient_profile, updated_node_embeddings['disease'].t())
            batch_scores.append(scores)

        # 4. Calculate Loss and Backpropagate
        # Stack scores into shape [batch_size, num_diseases]
        logits = torch.cat(batch_scores, dim=0) 
        
        # Cross Entropy pushes the score of the target_disease higher than the rest
        loss = criterion(logits, target_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f}")

    print("Training complete! Saving model...")
    os.makedirs("data/models", exist_ok=True)
    torch.save(model.state_dict(), "data/models/gnn_ranker.pth")

if __name__ == "__main__":
    train()