import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class DiagnosticGNN(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) for Clinical Diagnosis.
    Integrates 768-dimensional SapBERT semantic node embeddings,
    multi-head heterogeneous cross-attention, and clinical edge attributes (frequency & onset).
    """
    def __init__(self, num_diseases, num_symptoms, hidden_dim=256, symptom_in_dim=768, edge_dim=2, pretrained_symptom_x=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_diseases = num_diseases
        self.num_symptoms = num_symptoms

        # 1. Semantic Node Projections
        if pretrained_symptom_x is None and os.path.exists("data/processed/hpo_pyg_graph.pt"):
            try:
                hpo_data = torch.load("data/processed/hpo_pyg_graph.pt", map_location='cpu')
                if hasattr(hpo_data, 'x') and hpo_data.x is not None and hpo_data.x.shape[-1] == symptom_in_dim:
                    pretrained_symptom_x = hpo_data.x
            except Exception:
                pass

        if pretrained_symptom_x is not None:
            self.register_buffer("symptom_features", pretrained_symptom_x)
            self.symptom_proj = nn.Sequential(
                nn.Linear(symptom_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            self.symptom_emb = None
        else:
            self.symptom_emb = nn.Embedding(num_symptoms, hidden_dim)
            self.symptom_proj = None

        # 2. Edge Attribute Projection (Clinical Frequency & Onset Age)
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU()
        )

        # 3. Heterogeneous Graph Transformer (HGT) Layers
        self.metadata = (
            ['disease', 'symptom'],
            [
                ('symptom', 'is_a', 'symptom'),
                ('disease', 'has', 'symptom'),
                ('symptom', 'rev_has', 'disease')
            ]
        )

        self.conv1 = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=self.metadata,
            heads=2
        )
        self.conv2 = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=self.metadata,
            heads=2
        )

        self.norm_d1 = nn.LayerNorm(hidden_dim)
        self.norm_s1 = nn.LayerNorm(hidden_dim)
        self.norm_d2 = nn.LayerNorm(hidden_dim)
        self.norm_s2 = nn.LayerNorm(hidden_dim)

    def forward(self, edge_index_dict, edge_attr_dict=None):
        # 1. Initialize node representations
        if hasattr(self, "symptom_features") and self.symptom_features is not None and self.symptom_proj is not None:
            s_x = self.symptom_proj(self.symptom_features)
        elif self.symptom_emb is not None:
            s_x = self.symptom_emb.weight
        else:
            raise ValueError("No symptom representations available.")

        # Disease nodes aggregate from their symptom representations, modulated by clinical edge attributes
        d_idx = edge_index_dict[('disease', 'has', 'symptom')][0]
        s_idx = edge_index_dict[('disease', 'has', 'symptom')][1]

        if edge_attr_dict is not None and ('disease', 'has', 'symptom') in edge_attr_dict:
            ea = edge_attr_dict[('disease', 'has', 'symptom')]
            freq = ea[:, 0:1]
            e_emb = self.edge_proj(ea)
            s_msg = s_x[s_idx] * freq + 0.05 * e_emb
        else:
            freq = torch.ones(s_idx.shape[0], 1, device=s_x.device)
            s_msg = s_x[s_idx]

        d_x = torch.zeros(self.num_diseases, self.hidden_dim, device=s_x.device)
        counts = torch.zeros(self.num_diseases, 1, device=s_x.device)
        d_x.index_add_(0, d_idx, s_msg)
        counts.index_add_(0, d_idx, freq)
        d_x = d_x / torch.clamp(counts, min=1e-5)

        x_dict = {'disease': d_x, 'symptom': s_x}

        # 2. First HGT Layer + Residual + LayerNorm
        h1 = self.conv1(x_dict, edge_index_dict)
        x_dict['disease'] = self.norm_d1(0.2 * h1['disease'] + d_x)
        x_dict['symptom'] = self.norm_s1(0.2 * h1['symptom'] + s_x)

        # 3. Second HGT Layer + Residual + LayerNorm
        h2 = self.conv2(x_dict, edge_index_dict)
        x_dict['disease'] = self.norm_d2(0.2 * h2['disease'] + x_dict['disease'])
        x_dict['symptom'] = self.norm_s2(0.2 * h2['symptom'] + x_dict['symptom'])

        return x_dict

    def predict_diagnosis(self, patient_symptom_indices, x_dict):
        """
        Ranks diseases based on a patient's observed symptoms using cosine similarity.
        """
        patient_symptoms = x_dict['symptom'][patient_symptom_indices]
        patient_profile = patient_symptoms.mean(dim=0, keepdim=True)
        patient_norm = F.normalize(patient_profile, p=2, dim=-1)
        disease_norm = F.normalize(x_dict['disease'], p=2, dim=-1)
        scores = torch.matmul(patient_norm, disease_norm.t())
        return scores.squeeze(0)