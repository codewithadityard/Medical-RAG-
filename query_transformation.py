import os 
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class MedicalQueryTransformer:
    def __init__(self):
        # 1. Define the specific Groq model
        self.model_name = "openai/gpt-oss-120b"
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def transform_query(self, query: list) -> list:
        try:
            # 2. Inject the query directly into the prompt
            prompt = f"""
            You are a medical terminology translator. 
            Convert these patient descriptions into standard clinical phenotypes.
            
            Patient Input: {query}
            
            Rules:
            1. Output ONLY a comma-separated list of formal medical terms.
            2. Do not include introductory text, bullet points, or explanations.
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # 3. Extract just the text content
            raw_output = response.choices[0].message.content
            
            # 4. Clean and split the string into a Python list for the GNN
            normalized_list = [symp.strip().lower() for symp in raw_output.split(',')]
            return normalized_list
            
        except Exception as e:
            print(f"Error: {e}. The query transformation failed, falling back to the normal query.")
            return query