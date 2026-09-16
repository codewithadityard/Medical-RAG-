import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv

class DiagnosticGNN(nn.Module):
    def __init__(self, num_diseases, num_symptoms, hidden_dim=512):
        super().__init__()
        
        # 1. Trainable Node Embeddings
        # We start with random embeddings and let the graph learn their optimal shapes
        self.disease_emb = nn.Embedding(num_diseases, hidden_dim)
        self.symptom_emb = nn.Embedding(num_symptoms, hidden_dim)

        # 2. First Message Passing Layer (1-hop neighbors)
        self.conv1 = HeteroConv({
            # Symptoms share info with parent/child symptoms
            ('symptom', 'is_a', 'symptom'): SAGEConv(hidden_dim, hidden_dim),
            # Diseases pull info from their symptoms
            ('disease', 'has', 'symptom'): SAGEConv(hidden_dim, hidden_dim),
            # Symptoms pull info from the diseases they belong to
            ('symptom', 'rev_has', 'disease'): SAGEConv(hidden_dim, hidden_dim),
        }, aggr='mean')

        # 3. Second Message Passing Layer (2-hop neighbors)
        self.conv2 = HeteroConv({
            ('symptom', 'is_a', 'symptom'): SAGEConv(hidden_dim, hidden_dim),
            ('disease', 'has', 'symptom'): SAGEConv(hidden_dim, hidden_dim),
            ('symptom', 'rev_has', 'disease'): SAGEConv(hidden_dim, hidden_dim),
        }, aggr='mean')

    def forward(self, edge_index_dict):
        # Initialize the feature dictionaries
        x_dict = {
            'disease': self.disease_emb.weight,
            'symptom': self.symptom_emb.weight
        }

        # Pass messages through layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Pass messages through layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        
        # Return the context-aware embeddings
        return x_dict

    def predict_diagnosis(self, patient_symptom_indices, x_dict):
        """
        Ranks diseases based on a patient's observed symptoms.
        """
        # Get the updated embeddings for the patient's specific symptoms
        patient_symptoms = x_dict['symptom'][patient_symptom_indices]
        
        # Create a single patient profile vector by averaging their symptoms
        patient_profile = patient_symptoms.mean(dim=0, keepdim=True)
        
        # Compute dot product between patient profile and all diseases
        disease_embeddings = x_dict['disease']
        scores = torch.matmul(patient_profile, disease_embeddings.t())
        
        return scores.squeeze(0)