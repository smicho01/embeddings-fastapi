from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import secrets
import uvicorn
from dotenv import load_dotenv
import os

from app import embeddings

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Embedding Service")
security = HTTPBasic()

# Get configuration from environment variables
USERNAME = os.getenv('API_USERNAME')
PASSWORD = os.getenv('API_PASSWORD')

# Load model globally - it will stay in memory
model = SentenceTransformer('all-MiniLM-L6-v2')

# Request/Response models
class EmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int


# TODO: Dodac .env !!!!!!!

# ============================ Auth  ============================ #
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify username and password"""
    is_username_correct = secrets.compare_digest(credentials.username, USERNAME)
    is_password_correct = secrets.compare_digest(credentials.password, PASSWORD)
    
    if not (is_username_correct and is_password_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return credentials.username

# ============================ API endpoints  ============================ #

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    username: str = Depends(verify_credentials)
    ):
    try:
        # Generate embeddings of length 384
        embeddings = model.encode(request.texts)
        
        return {
            "embeddings": embeddings.tolist(),
            "dimensions": len(embeddings[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

   
@app.post("/embed2", response_model=EmbeddingResponse)
async def create_embeddings(
        request: EmbeddingRequest,
        username: str = Depends(verify_credentials)
        ):
    try:
        # Generate embeddings of length 768
        generated_embeddings = embeddings.get_embeddings_fixed_dim(request.texts, 768)
        
        return {
            "embeddings": generated_embeddings,
            "dimensions": len(generated_embeddings[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "all-MiniLM-L6-v2"}

if __name__ == "__main__":
    # Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
    uvicorn.run(app, host="0.0.0.0", port=8000)