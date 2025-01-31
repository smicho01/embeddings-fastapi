from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Embedding Service")

# Load model globally - it will stay in memory
model = SentenceTransformer('all-MiniLM-L6-v2')

# Request/Response models
class EmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    try:
        # Generate embeddings
        embeddings = model.encode(request.texts)
        
        return {
            "embeddings": embeddings.tolist(),
            "dimensions": len(embeddings[0])
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