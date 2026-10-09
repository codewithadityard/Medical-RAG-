from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import base64
from pipeline import DiagnosticPipeline

app = FastAPI(title="Medical RAG API")

# Initialize heavy models once when server starts
print("Loading heavy models into memory...")
engine = DiagnosticPipeline(device='cpu') 


class PatientCase(BaseModel):
    symptoms: List[str] = []
    image_base64: Optional[str] = None


@app.post("/diagnose")
async def diagnose_patient(case: PatientCase):
    try:
        image_bytes = None
        if case.image_base64:
            image_bytes = base64.b64decode(case.image_base64)
            
        result = engine.run(patient_symptoms_list=case.symptoms, image_input=image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)