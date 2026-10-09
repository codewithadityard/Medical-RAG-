import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
class MedicalGenerator:
    def generate_clinical_summary(self,symptoms,disease_name,evidence_docs):

        client=Groq(api_key=os.environ.get("GROQ_API_KEY"))
        context = "\n\n".join(evidence_docs)

        prompt = f"""
    You are a medical expert. 
    You are given the list of symptoms: {symptoms} and name of the disease: {disease_name}
    
    Medical Literature:
    {context}
    
    You need to check if the medical literature is accurate for the given symptoms and disease. 
    I need you to write a well-detailed clinical report. Make it short, focused, and concise.
    Do NOT invent any medical facts. Only answer from the context and the data you have been given.
    """

        response=client.chat.completions.create(
            messages=[{"role":"user","content":prompt}],
            model="openai/gpt-oss-120b"
        )
        return response.choices[0].message.content