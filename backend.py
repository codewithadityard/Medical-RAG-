from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from pipeline import DiagnosticPipeline

app = FastAPI(title="Medical RAG API")

# Initialize the heavy models once when the server starts
print("Loading heavy models into memory...")
engine = DiagnosticPipeline(device='cpu') 

class PatientCase(BaseModel):
    symptoms: List[str]

@app.post("/diagnose")
async def diagnose_patient(case: PatientCase):
    try:
        # Run your pipeline
        result = engine.run(case.symptoms)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)