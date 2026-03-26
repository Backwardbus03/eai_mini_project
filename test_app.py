from fastapi.testclient import TestClient
from app import app
import json

def run_tests():
    print("Testing ML Backend...")
    
    with TestClient(app) as client:
        # 1. Test POST /api/ml/score
        payload = {
            "job_id": "job-123",
            "candidate_id": "cand-456",
            "requirements": {
                "skills": ["python", "machine learning", "fastapi"],
                "experience": 2,
                "education": ["computer science"]
            },
            "resume_text": "Experienced machine learning engineer. 3 years of using python and fastapi in production. Education: B.Tech in Computer Science."
        }
        
        response = client.post("/api/ml/score", json=payload)
        print("POST /api/ml/score response status:", response.status_code)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        print("Response Data:", json.dumps(data, indent=2))
        
        assert "id" in data
        assert "fit_score" in data
        assert "xgboost_rank" in data
        assert data["job_id"] == "job-123"
        assert data["candidate_id"] == "cand-456"
        
        score_id = data["id"]
        
        # 2. Test GET /api/ml/score
        response_all = client.get("/api/ml/score")
        print("GET /api/ml/score response status:", response_all.status_code)
        assert response_all.status_code == 200
        all_data = response_all.json()
        assert len(all_data) >= 1
        
        # 3. Test GET /api/ml/score/{id}
        response_single = client.get(f"/api/ml/score/{score_id}")
        print(f"GET /api/ml/score/{score_id} status:", response_single.status_code)
        assert response_single.status_code == 200
        single_data = response_single.json()
        assert single_data["id"] == score_id
        
        print("All tests passed successfully!")

if __name__ == "__main__":
    run_tests()
