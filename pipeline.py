import torch
import json
from gnn import DiagnosticGNN
from generator import MedicalGenerator
from retriever import MedicalHybridRetriever
from verifier import FaithfulnessVerifier
from query_transformation import MedicalQueryTransformer
from entity_linker import MedicalEntityLinker
from multimodal import BiomedCLIPPhenotypeClassifier

class DiagnosticPipeline:
    def __init__(self, device='cpu'):
        self.device = device
        print("Initializing Diagnostic Engine...")
        
        # 1. Load Mappings
        with open("data/processed/hpo_mapping.json", "r") as f:
            self.hpo_map = json.load(f)
            self.hpo_to_idx = self.hpo_map["hpo_to_idx"]
            self.name_to_hpo = self.hpo_map["name_to_hpo"]
            
        with open("data/processed/disease_mapping.json", "r") as f:
            self.disease_map = json.load(f)
            self.idx_to_disease = {int(k): v for k, v in self.disease_map["idx_to_disease"].items()}
            self.disease_to_name = self.disease_map["disease_to_name"]

        # 2. Load Graph Data for GNN context
        self.edge_index_dict = self._load_edges()
        num_symptoms = len(self.hpo_to_idx)
        num_diseases = len(self.idx_to_disease)

        # 3. Load Trained GNN (Heterogeneous Graph Transformer)
        self.gnn = DiagnosticGNN(num_diseases, num_symptoms, hidden_dim=256).to(self.device)
        self.gnn.load_state_dict(torch.load("data/models/gnn_ranker.pth", map_location=self.device))
        self.gnn.eval()

        # 4. Load Engine Components
        self.retriever = MedicalHybridRetriever()
        self.generator = MedicalGenerator()
        self.verifier = FaithfulnessVerifier(device=self.device)
        self.query_transformer = MedicalQueryTransformer()
        self.entity_linker = MedicalEntityLinker(threshold=0.70)
        self.multimodal_classifier = None

    def _get_multimodal_classifier(self):
        if self.multimodal_classifier is None:
            self.multimodal_classifier = BiomedCLIPPhenotypeClassifier(device=self.device)
        return self.multimodal_classifier

    def _load_edges(self):
        bipartite_edges = torch.load("data/processed/disease_hpo_edges.pt").to(self.device)
        hpo_graph = torch.load("data/processed/hpo_pyg_graph.pt").to(self.device)
        edge_attr = torch.load("data/processed/disease_hpo_edge_attr.pt").to(self.device)
        self.edge_attr_dict = {
            ('disease', 'has', 'symptom'): edge_attr
        }
        return {
            ('symptom', 'is_a', 'symptom'): hpo_graph.edge_index,
            ('disease', 'has', 'symptom'): bipartite_edges,
            ('symptom', 'rev_has', 'disease'): bipartite_edges.flip([0])
        }

    def run(self, patient_symptoms_list=None, image_input=None):
        print(f"\n--- New Patient Case ---")

        if patient_symptoms_list is None:
            patient_symptoms_list = []
        elif isinstance(patient_symptoms_list, str):
            patient_symptoms_list = [s.strip() for s in patient_symptoms_list.split(",") if s.strip()]
        else:
            patient_symptoms_list = list(patient_symptoms_list)

        visual_detections = []
        if image_input is not None:
            print("\nAnalyzing clinical image using Microsoft BiomedCLIP...")
            classifier = self._get_multimodal_classifier()
            visual_detections = classifier.classify_image(image_input, top_k=2)
            for d in visual_detections:
                patient_symptoms_list.append(d["phenotype"])
                print(f"BiomedCLIP detected: '{d['phenotype']}' (Confidence: {d['confidence']*100:.1f}%)")

        print(f"Presented Symptoms: {patient_symptoms_list}")

        patient_symptoms_list = self.query_transformer.transform_query(patient_symptoms_list)
        print(f"Translated Clinical Terms: {patient_symptoms_list}")

        # Step 1: Map text symptoms to graph IDs via SapBERT Semantic Entity Linker
        symptom_indices = []
        valid_english_symptoms = []
        for symp in patient_symptoms_list:
            match = self.entity_linker.link_symptom(symp)
            if match:
                idx = match["graph_idx"]
                if idx not in symptom_indices:
                    symptom_indices.append(idx)
                    valid_english_symptoms.append(match["matched_term"])
                    print(f"Mapped '{symp}' -> '{match['matched_term']}' (HPO: {match['hpo_id']}, Score: {match['score']})")
            else:
                print(f"Warning: Symptom '{symp}' could not be linked to ontology.")

        if not symptom_indices:
            return {"error": "Could not map the input data to valid clinical ontology terms."}

        patient_tensor = torch.tensor(symptom_indices, dtype=torch.long).to(self.device)

        # Step 2: GNN Forward Pass
        with torch.no_grad():
            node_embeddings = self.gnn(self.edge_index_dict, self.edge_attr_dict)
            scores = self.gnn.predict_diagnosis(patient_tensor, node_embeddings)

        # Step 3: Get Top Diagnosis
        top_k = 3
        top_scores, top_indices = torch.topk(scores, top_k)
        
        print("\nTop Diagnoses from GNN:")
        for idx in top_indices.tolist():
            print(f"- {self.idx_to_disease[idx]}")

        top_disease_id = self.idx_to_disease[top_indices[0].item()]
        top_disease_name = self.disease_to_name[top_disease_id]

        # Step 4: Targeted RAG Retrieval
        search_query = f"{top_disease_name} case report presenting with {' '.join(valid_english_symptoms)}"
        print(f"\nFormulated PubMed Query: '{search_query}'")
        
        print("\nFetching literature evidence...")
        evidence = self.retriever.search(search_query, top_k=2)
        for i, doc in enumerate(evidence):
            print(f"\n[Evidence {i+1}] {doc[:200]}...")

        # Step 5: Generate the Summary
        print("\nSynthesizing Final Diagnosis...")
        response = self.generator.generate_clinical_summary(valid_english_symptoms, top_disease_name, evidence)

        # Step 6: Faithfulness Verification
        print("\nRunning MedNLI Faithfulness Verification...")
        context_string = "\n".join(evidence)
        is_faithful = self.verifier.check_hallucination(context_string, response)

        if is_faithful:
            print(f"\n[ VERIFIED DIAGNOSIS]\n{response}")
        else:
            print(f"\n[ HALLUCINATION BLOCKED]\nInternal Generator Output: {response}\nReason: The generated claim could not be mathematically entailed by the retrieved PubMed context.")

        # (Replace your final print statements in pipeline.py with this)
        return {
            "mapped_symptoms": valid_english_symptoms,
            "top_diseases": [self.idx_to_disease[idx] for idx in top_indices.tolist()],
            "summary": response,
            "is_faithful": is_faithful,
            "evidence": evidence,
            "visual_detections": visual_detections
        }


if __name__ == "__main__":
    from PIL import Image
    engine = DiagnosticPipeline()
    test_image = Image.new("RGB", (224, 224), color=(60, 60, 70))
    engine.run(
        patient_symptoms_list="My fingers are unusually long and skinny",
        image_input=test_image
    )