import pandas as pd
from sklearn.preprocessing import StandardScaler
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

from pymongo import MongoClient
from lime.lime_tabular import LimeTabularExplainer
from bson import ObjectId

app = FastAPI(title="Fare Hire ML Service", description="ML Backend for scoring resumes using Rule-based + XGBoost + LIME")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# DB Connection Setup
# ---------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/farehire")
client_db = MongoClient(MONGO_URI)
try:
    db = client_db.get_default_database()
except Exception:
    db = client_db["farehire"]

# ---------------------------------------------------------
# XGBoost & LIME Setup
# ---------------------------------------------------------

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

scaler = StandardScaler()
explainer = None

@app.on_event("startup")
def startup_event():
    global explainer
    try:
        df = pd.read_csv("ai_resume_screening.csv")
        numerical_cols = ['years_experience', 'skills_match_score', 'resume_length']
        scaler.fit(df[numerical_cols])
        print("StandardScaler fitted successfully from ai_resume_screening.csv.")
        
        # Build explainer
        df_encoded = df.copy()
        edu_map = {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}
        df_encoded['education_level'] = df_encoded['education_level'].map(edu_map).fillna(0)
        df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
        
        X_train = df_encoded[['years_experience', 'skills_match_score', 'education_level', 'resume_length']]
        
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=['Not Shortlisted', 'Shortlisted'],
            mode='classification'
        )
        print("LIME Explainer initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not fit scaler or explainer from dataset: {e}")
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

class ScoreRequest(BaseModel):
    job_id: str
    candidate_id: str

class ScoreResponse(BaseModel):
    job_id: str
    candidate_id: str
    fit_score: float
    xgboost_rank: float
    lime_data: List[Any]
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
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception:
            pass
        if not text.strip():
            try:
                doc = fitz.open(file_path)
                for page in doc:
                    text += page.get_text("text") + "\n"
            except Exception:
                pass
    elif ext == "docx":
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def download_and_extract(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        
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
    job_id = request.job_id
    candidate_id = request.candidate_id
    
    # 1. Fetch Job from DB
    job = None
    try:
        job = db.jobs.find_one({"_id": ObjectId(job_id)})
    except Exception:
        job = db.jobs.find_one({"_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found in DB")
        
    # 2. Fetch Candidate from DB
    candidate = None
    try:
        candidate = db.candidates.find_one({"_id": ObjectId(candidate_id)})
    except Exception:
        candidate = db.candidates.find_one({"_id": candidate_id})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found in DB")
        
    req_dict = job.get("requirements", {})
    resume_url = candidate.get("resume_url")
    if not resume_url:
        raise HTTPException(status_code=400, detail="Candidate does not have a resume_url")
        
    text_to_analyze = download_and_extract(resume_url)
    if not text_to_analyze:
        raise HTTPException(status_code=400, detail="Could not extract text from the provided resume")

    # 3. Rule-based filtering
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
    
    # Features order: ['years_experience', 'skills_match_score', 'education_level', 'resume_length']
    features = pd.DataFrame([[
        scaled_nums[0], 
        scaled_nums[1], 
        edu_level, 
        scaled_nums[2]
    ]], columns=['years_experience', 'skills_match_score', 'education_level', 'resume_length'])
    
    # 4. XGBoost Predict
    xgb_pred = model.predict_proba(features)[0][1]
    final_rank = round(float(xgb_pred * 100), 2)
    
    # 5. Generte LIME explanation
    global explainer
    lime_data = []
    if explainer is not None:
        try:
            exp = explainer.explain_instance(
                data_row=features.iloc[0].values,
                predict_fn=model.predict_proba
            )
            lime_data = exp.as_list()
        except Exception as e:
            print("LIME explanation failed:", e)

    # 6. Update DB with [cand-id: [score, lime_data]] under the job table
    try:
        update_filter = {"_id": job["_id"]}
        update_action = {"$set": {f"candidates.{candidate_id}": [final_rank, lime_data]}}
        db.jobs.update_one(update_filter, update_action)
    except Exception as e:
        print("Failed to update DB:", e)
        # We don't want to error out the API if saving fails, but practically it should work.

    return {
        "job_id": job_id,
        "candidate_id": candidate_id,
        "fit_score": rule_score,
        "xgboost_rank": final_rank,
        "lime_data": lime_data,
        "fit_breakdown": breakdown,
        "resume_quality": quality
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
