from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
import tempfile
import os
import requests
import pdfplumber
import fitz  # PyMuPDF
import docx
import numpy as np
import xgboost as xgb
import pickle

from ai_component import compute_fit, resume_quality

app = FastAPI(title="Fare Hire ML Service", description="ML Backend for scoring resumes using Rule-based + XGBoost")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# XGBoost Setup & Preprocessing
# ---------------------------------------------------------

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

scaler = StandardScaler()

@app.on_event("startup")
def startup_event():
    try:
        df = pd.read_csv("ai_resume_screening.csv")
        numerical_cols = ['years_experience', 'skills_match_score', 'resume_length']
        scaler.fit(df[numerical_cols])
        print("StandardScaler fitted successfully from ai_resume_screening.csv.")
    except Exception as e:
        print(f"Warning: Could not fit scaler from dataset: {e}")
        scaler.mean_ = np.array([7.50656667, 73.68265333, 572.5847])
        scaler.scale_ = np.array([4.62402677, 16.76562974, 178.70693913])
        scaler.var_ = scaler.scale_ ** 2

def get_education_level(text: str) -> int:
    text_lower = text.lower()
    if any(k in text_lower for k in ["phd", "ph.d", "doctorate"]):
        return 3
    if any(k in text_lower for k in ["msc", "m.sc", "ms", "master", "m.tech", "mtech"]):
        return 2
    if any(k in text_lower for k in ["bsc", "b.sc", "bs", "bachelor", "b.tech", "btech", "be"]):
        return 1
    return 0

# ---------------------------------------------------------
# Pydantic Models & Storage
# ---------------------------------------------------------

# In-memory storage for scores
# {score_id: score_details}
scores_db: Dict[str, Any] = {}

class JobRequirements(BaseModel):
    skills: List[str]
    experience: Optional[int] = 0
    education: Optional[List[str]] = []

class ScoreRequest(BaseModel):
    job_id: str
    candidate_id: str
    requirements: JobRequirements
    resume_text: Optional[str] = None
    resume_url: Optional[str] = None

class ScoreResponse(BaseModel):
    id: str
    job_id: str
    candidate_id: str
    fit_score: float
    xgboost_rank: float
    fit_breakdown: Dict[str, Any]
    resume_quality: Dict[str, Any]

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

ALLOWED_EXTENSIONS = {"pdf", "docx"}

def extract_text_from_file(file_path: str) -> str:
    ext = file_path.rsplit(".", 1)[1].lower()
    text = ""
    if ext == "pdf":
        # First try pdfplumber
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception:
            pass
            
        # Fallback to PyMuPDF
        if not text.strip():
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text("text") + "\n"
    elif ext == "docx":
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def download_and_extract(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Try to infer extension
        ext = "pdf"
        if "docx" in url.lower():
            ext = "docx"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
            
        text = extract_text_from_file(temp_file_path)
        os.remove(temp_file_path)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch or parse resume from URL: {str(e)}")

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "ok", "message": "ML Service is running"}

@app.post("/api/ml/score", response_model=ScoreResponse)
def score_resume(request: ScoreRequest):
    if not request.resume_text and not request.resume_url:
        raise HTTPException(status_code=400, detail="Either resume_text or resume_url must be provided")
        
    # Get the text
    text_to_analyze = request.resume_text
    if not text_to_analyze and request.resume_url:
        text_to_analyze = download_and_extract(request.resume_url)
        
    if not text_to_analyze:
        raise HTTPException(status_code=400, detail="Could not extract text from the provided resume")

    # 1. Rule-based filtering & scoring from ai_component.py
    req_dict = request.requirements.dict()
    rule_score, breakdown = compute_fit(req_dict, text_to_analyze)
    quality = resume_quality(text_to_analyze)
    
    from ai_component import extract_years_of_experience
    years_exp = extract_years_of_experience(text_to_analyze) or 0.0
    skills_score = breakdown.get("skills", 0)
    edu_level = get_education_level(text_to_analyze)
    resume_length = len(text_to_analyze.split())
    
    # Scale numerical features
    num_features = pd.DataFrame(
        [[years_exp, skills_score, resume_length]], 
        columns=['years_experience', 'skills_match_score', 'resume_length']
    )
    scaled_nums = scaler.transform(num_features)[0]
    
    # XGBoost Ranking
    # features order: ['years_experience', 'skills_match_score', 'education_level', 'resume_length']
    # Remember years_experience, skills_match_score, and resume_length are scaled. edu_level is ordinal encoded (not scaled).
    features = pd.DataFrame([[
        scaled_nums[0], 
        scaled_nums[1], 
        edu_level, 
        scaled_nums[2]
    ]], columns=['years_experience', 'skills_match_score', 'education_level', 'resume_length'])
    
    xgb_pred = model.predict_proba(features)[0][1] # Probability of Class 1 ('Yes')
    final_rank = round(float(xgb_pred * 100), 2)
    
    score_id = str(uuid.uuid4())
    score_data = {
        "id": score_id,
        "job_id": request.job_id,
        "candidate_id": request.candidate_id,
        "fit_score": rule_score,
        "xgboost_rank": final_rank,
        "fit_breakdown": breakdown,
        "resume_quality": quality
    }
    
    # Store in memory
    scores_db[score_id] = score_data
    
    return score_data

@app.get("/api/ml/score", response_model=List[ScoreResponse])
def get_all_scores():
    return list(scores_db.values())

@app.get("/api/ml/score/{score_id}", response_model=ScoreResponse)
def get_score_by_id(score_id: str):
    if score_id not in scores_db:
        raise HTTPException(status_code=404, detail="Score not found")
    return scores_db[score_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
