
from fastapi.testclient import TestClient
from app import app
import app as app_module
import mongomock
from bson import ObjectId
import json
import httpx

# Mock HTTP request for the resume
class MockResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPError("Bad Request")

def mock_get(url, *args, **kwargs):
    # Dummy PDF or Docx text content
    return MockResponse(b"Dummy resume data")

def run_tests():
    print("Testing ML Backend with MongoDB integration...")
    
    # 1. Setup Mock DB
    mock_client = mongomock.MongoClient()
    mock_db = mock_client.db
    
    # Patch the app.py DB with our mock DB
    app_module.client_db = mock_client
    app_module.db = mock_db
    
    # Insert dummy Job and Candidate into mock DB
    job_id = "job-new-123"
    candidate_id = "cand-new-456"
    
    mock_db.jobs.insert_one({
        "_id": job_id,
        "requirements": {
            "skills": ["python", "machine learning", "fastapi"],
            "experience": 2,
            "education": ["computer science"]
        },
        "candidates": {} # Ensure Array or Map exists
    })
    
    mock_db.candidates.insert_one({
        "_id": candidate_id,
        "resume_url": "http://example.com/dummy.pdf"
    })
    
    # 2. Patch requests.get so it doesn't really download
    # We will instead inject the extracted text directly via a monkeypatch
    # In real scenarios we'd mock the resume fetch or the extractor. 
    # To test end to end easily, let's mock `download_and_extract`
    
    original_downloader = app_module.download_and_extract
    app_module.download_and_extract = lambda url: (
        "Senior lead machine learning engineer with 15 years of experience in AI, NLP, and deep learning. "
        "Led multiple successful projects using python, tensorflow, pytorch, and fastapi. "
        "Education: PhD in Computer Science from a top university. "
        "Extensive background spanning data science, etl pipelines, and cloud computing. "
        "Responsible for building scalable microservices and robust machine learning models "
        "deploying them to production via docker and kubernetes. "
    )
    
    with TestClient(app) as client:
        payload = {
            "job_id": job_id,
            "candidate_id": candidate_id
        }
        
        response = client.post("/api/ml/score", json=payload)
        app_module.download_and_extract = original_downloader # Restore
        
        print("POST /api/ml/score response status:", response.status_code)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        print("Response Data:", json.dumps(data, indent=2))
        
        assert "fit_score" in data
        assert "xgboost_rank" in data
        assert "lime_data" in data
        assert data["job_id"] == job_id
        assert data["candidate_id"] == candidate_id
        
        # Verify LIME output exists
        assert isinstance(data["lime_data"], list)
        if len(data["lime_data"]) > 0:
            print(f"LIME extracted successfully: {data['lime_data'][0]}")
            
        # 3. Verify that DB was updated
        updated_job = mock_db.jobs.find_one({"_id": job_id})
        assert "candidates" in updated_job
        assert candidate_id in updated_job["candidates"]
        saved_score, saved_lime = updated_job["candidates"][candidate_id]
        
        assert saved_score == data["xgboost_rank"]
        print(f"Verified DB Update! Saved Score: {saved_score}")
        
        print("All tests passed successfully!")

if __name__ == "__main__":
    run_tests()
