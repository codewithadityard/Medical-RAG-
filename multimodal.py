import os
import torch
from PIL import Image
import io
from open_clip import create_model_from_pretrained, get_tokenizer

# Standard observable phenotypes in rare genetic and clinical syndromes
DEFAULT_OBSERVABLE_PHENOTYPES = {
    # Skeletal & Thoracic
    "arachnodactyly": "arachnodactyly abnormally long slender spider fingers",
    "pectus excavatum": "pectus excavatum sunken depressed chest wall deformity",
    "pectus carinatum": "pectus carinatum protruding pigeon chest wall",
    "scoliosis": "scoliosis abnormal lateral curvature of spine X-ray",
    "joint hypermobility": "joint hypermobility loose flexible double-jointed limbs",
    
    # Dermatological / Cutaneous
    "malar butterfly rash": "erythematous malar butterfly rash on cheeks and bridge of nose",
    "cafe-au-lait spots": "cafe-au-lait macules light brown pigmented skin spots",
    "hypopigmented macules": "hypopigmented ash leaf skin macules pale skin patches",
    "telangiectasia": "facial cutaneous telangiectasia dilated small blood vessels on skin",
    "hyperkeratosis": "cutaneous hyperkeratosis thick scaly skin lesions",
    
    # Ocular / Ophthalmology
    "ectopia lentis": "ectopia lentis displaced subluxated eye lens slit lamp examination",
    "coloboma": "ocular coloboma iris or retinal cleft notch defect",
    "blue sclerae": "blue sclerae thin translucent sclera of the eyes",
    
    # Craniofacial Dysmorphism
    "hypertelorism": "ocular hypertelorism abnormally wide-spaced eyes",
    "micrognathia": "micrognathia abnormally small underdeveloped lower jaw",
    
    # Radiological Manifestations
    "aortic aneurysm": "aortic root aneurysm aortic dilation cardiovascular imaging",
    "cardiomegaly": "cardiomegaly enlarged cardiac silhouette on chest X-ray",
}


class BiomedCLIPPhenotypeClassifier:
    """
    Multimodal clinical phenotype classifier powered by Microsoft BiomedCLIP.
    Pretrained on 15M+ PubMed Central biomedical image-text pairs.
    """
    def __init__(self, device='cpu'):
        self.device = device
        print("Loading Microsoft BiomedCLIP multimodal vision-language model...")
        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        self.model, self.preprocess = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Cache pre-tokenized default phenotypes
        self.phenotype_keys = list(DEFAULT_OBSERVABLE_PHENOTYPES.keys())
        self.phenotype_descriptions = list(DEFAULT_OBSERVABLE_PHENOTYPES.values())
        
        text_tokens = self.tokenizer(self.phenotype_descriptions).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            self.cached_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def _load_image(self, image_input):
        """Converts file path, bytes, or PIL Image into a preprocessed PyTorch tensor."""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif hasattr(image_input, "read"):  # Uploaded file buffer (Streamlit / FastAPI)
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def classify_image(self, image_input, top_k=3, min_similarity=0.20):
        """
        Classifies an input clinical image against observable phenotype definitions.
        Returns top-k detected phenotypes with similarity and confidence scores.
        """
        image_tensor = self._load_image(image_input)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = (image_features @ self.cached_text_features.T).squeeze(0)
            # Scaled temperature for softmax probability estimation
            probs = torch.softmax(similarity * 40.0, dim=-1)

        top_scores, top_indices = torch.topk(similarity, min(top_k, len(self.phenotype_keys)))
        
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            phenotype_name = self.phenotype_keys[idx]
            prob = probs[idx].item()
            if score >= min_similarity:
                results.append({
                    "phenotype": phenotype_name,
                    "description": self.phenotype_descriptions[idx],
                    "similarity": round(score, 4),
                    "confidence": round(prob, 4)
                })

        return results

    def extract_phenotypes(self, image_input, top_k=2):
        """Returns detected phenotype names as a list of plain strings for the diagnostic pipeline."""
        detections = self.classify_image(image_input, top_k=top_k)
        return [d["phenotype"] for d in detections]


if __name__ == "__main__":
    classifier = BiomedCLIPPhenotypeClassifier(device='cpu')

    # Test with a synthetic chest-colored image
    test_img = Image.new("RGB", (224, 224), color=(60, 60, 70))
    print("\nRunning Test Image Classification (Chest X-Ray simulation)...")
    detected = classifier.classify_image(test_img, top_k=3)
    for d in detected:
        print(f"- {d['phenotype']} (Sim: {d['similarity']}, Conf: {d['confidence']*100:.1f}%)")
