import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class FaithfulnessVerifier:
    def __init__(self, device='cpu'):
        print("Loading MedNLI Verifier...")
        self.device = device
        
        # CORRECTED MODEL ID: Valid public Hugging Face repository
        model_id = "cnut1648/biolinkbert-mednli"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def check_hallucination(self, retrieved_document, generated_claim):
        """
        Premise: The retrieved PubMed case report.
        Hypothesis: The sentence the AI generated.
        Returns: True (Entailment) or False (Contradiction/Neutral)
        """
        inputs = self.tokenizer(
            retrieved_document, 
            generated_claim, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            predicted_class = torch.argmax(logits, dim=1).item()

        # MedNLI labels: 0 = Entailment, 1 = Neutral, 2 = Contradiction
        # We strictly require Entailment (0) to pass the safety check
        is_faithful = (predicted_class == 0)
        
        return is_faithful