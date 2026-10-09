import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class FaithfulnessVerifier:
    def __init__(self, device='cpu', min_factuality_score=0.5):
        print("Loading MedNLI Verifier...")
        self.device = device
        self.min_factuality_score = min_factuality_score

        # Valid public Hugging Face repository
        model_id = "cnut1648/biolinkbert-mednli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.last_results = None

    def _extract_claim_sentences(self, text: str):
        """
        Deconstructs clinical summary into atomic factual claim sentences.
        Filters out markdown headers, table lines, bullets, and short noise.
        """
        sentences = []
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Ignore table rows, markdown headers, horizontal rules
            if not line or line.startswith('|') or line.startswith('#') or line.startswith('---'):
                continue
            # Strip bold/italic markdown formatting
            cleaned = re.sub(r'[*_]{1,3}', '', line)
            # Strip leading list markers (- , * , 1. , etc.)
            cleaned = re.sub(r'^[\s\-*•\d.]+\s*', '', cleaned).strip()
            # Skip header-like labels ending with a colon
            if cleaned.endswith(':'):
                continue
            # Split line into individual sentences
            for s in re.split(r'(?<=[.!?])\s+', cleaned):
                s = s.strip()
                # Keep meaningful factual sentences with at least 4 words
                if len(s.split()) >= 4:
                    sentences.append(s)
        return sentences

    def _chunk_text(self, text: str, max_words: int = 250, overlap: int = 40):
        """
        Splits long evidence documents into overlapping passages to prevent 512-token truncation.
        """
        words = text.split()
        if len(words) <= max_words:
            return [text]
        chunks = []
        step = max_words - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + max_words])
            if chunk:
                chunks.append(chunk)
        return chunks

    def verify_sentences(self, retrieved_document, generated_claim):
        """
        Performs sentence-by-sentence verification of the generated summary
        against chunked evidence passages using BioLinkBERT-MedNLI.
        """
        claims = self._extract_claim_sentences(generated_claim)

        # Extract passages from retrieved document
        if isinstance(retrieved_document, str):
            raw_docs = [d.strip() for d in retrieved_document.split('\n\n') if d.strip()]
        elif isinstance(retrieved_document, list):
            raw_docs = [str(d).strip() for d in retrieved_document if str(d).strip()]
        else:
            raw_docs = [str(retrieved_document).strip()]

        passages = []
        for doc in raw_docs:
            passages.extend(self._chunk_text(doc))

        if not claims:
            return {
                "is_faithful": True,
                "factuality_score": 1.0,
                "entailed_count": 0,
                "contradicted_count": 0,
                "neutral_count": 0,
                "total_claims": 0,
                "claim_details": [],
            }

        if not passages:
            return {
                "is_faithful": False,
                "factuality_score": 0.0,
                "entailed_count": 0,
                "contradicted_count": 0,
                "neutral_count": len(claims),
                "total_claims": len(claims),
                "claim_details": [{"sentence": c, "status": "neutral", "entailed": False} for c in claims],
            }

        # Build cross-pairs between claim sentences and evidence passages
        pairs_premise = []
        pairs_hypo = []
        pair_mapping = []

        for c_idx, claim in enumerate(claims):
            for p_idx, passage in enumerate(passages):
                pairs_premise.append(passage)
                pairs_hypo.append(claim)
                pair_mapping.append((c_idx, p_idx))

        # Batched inference over pairs to optimize CPU runtime
        batch_size = 16
        all_preds = []
        for b_start in range(0, len(pairs_premise), batch_size):
            b_prem = pairs_premise[b_start:b_start + batch_size]
            b_hypo = pairs_hypo[b_start:b_start + batch_size]
            inputs = self.tokenizer(
                b_prem,
                b_hypo,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                b_preds = torch.argmax(logits, dim=1).tolist()
                all_preds.extend(b_preds)

        # Aggregate per-sentence results
        # MedNLI labels: 0 = Entailment, 1 = Neutral, 2 = Contradiction
        claim_stats = {i: {"entail": 0, "neutral": 0, "contradict": 0} for i in range(len(claims))}
        for (c_idx, _), pred in zip(pair_mapping, all_preds):
            if pred == 0:
                claim_stats[c_idx]["entail"] += 1
            elif pred == 1:
                claim_stats[c_idx]["neutral"] += 1
            elif pred == 2:
                claim_stats[c_idx]["contradict"] += 1

        entailed_count = 0
        contradicted_count = 0
        neutral_count = 0
        claim_details = []

        print("\n--- Sentence-by-Sentence MedNLI Verification ---")
        for i, claim in enumerate(claims):
            st = claim_stats[i]
            if st["entail"] > 0:
                status = "entailed"
                entailed_count += 1
                icon = "✅ Entailed"
            elif st["contradict"] > 0:
                status = "contradicted"
                contradicted_count += 1
                icon = "❌ Contradicted"
            else:
                status = "neutral"
                neutral_count += 1
                icon = "⚠️ Neutral/Unverified"

            claim_details.append({
                "sentence": claim,
                "status": status,
                "entail_matches": st["entail"],
                "contradict_matches": st["contradict"],
            })
            print(f"[{i + 1}] \"{claim}\" ➔ {icon}")

        total_claims = len(claims)
        factuality_score = round(entailed_count / total_claims, 3) if total_claims > 0 else 0.0

        # Faithful if no direct contradictions and meets minimum entailment ratio
        is_faithful = (contradicted_count == 0 and factuality_score >= self.min_factuality_score)

        pct = factuality_score * 100
        print(f"\nFactuality Score: {entailed_count}/{total_claims} verified ({pct:.1f}%) | Contradictions: {contradicted_count}")
        print(f"Overall Status: {'[VERIFIED FAITHFUL]' if is_faithful else '[HALLUCINATION DETECTED]'}")

        result = {
            "is_faithful": is_faithful,
            "factuality_score": factuality_score,
            "entailed_count": entailed_count,
            "contradicted_count": contradicted_count,
            "neutral_count": neutral_count,
            "total_claims": total_claims,
            "claim_details": claim_details,
        }
        self.last_results = result
        return result

    def check_hallucination(self, retrieved_document, generated_claim, return_details: bool = False):
        """
        Checks whether the generated text is entailed by the retrieved document.
        If return_details=False (default): returns boolean is_faithful.
        If return_details=True: returns full verification dictionary.
        """
        res = self.verify_sentences(retrieved_document, generated_claim)
        return res if return_details else res["is_faithful"]


if __name__ == "__main__":
    verifier = FaithfulnessVerifier(device='cpu')

    sample_doc = """
    Marfan syndrome is an autosomal dominant genetic disorder caused by mutations in the FBN1 gene.
    Cardiovascular manifestations include aortic root enlargement and aortic dissection.
    Skeletal features include tall stature, arachnodactyly, and pectus excavatum.
    """

    sample_report = """
    - Marfan syndrome is caused by mutations in the FBN1 gene.
    - Common skeletal symptoms include arachnodactyly and pectus excavatum.
    - Marfan syndrome is strictly caused by vitamin D deficiency in children.
    """

    print("\nRunning Standalone Verifier Test:")
    results = verifier.check_hallucination(sample_doc, sample_report, return_details=True)
    print("\nResult Dictionary:", results)