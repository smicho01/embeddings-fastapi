from sentence_transformers import SentenceTransformer
import torch.nn as nn
from typing import List
import numpy as np

class DimensionReducer:
    def __init__(self, original_model: str, target_dim: int):
        """
        Initialize with desired output dimension
        
        Args:
            original_model: Name or path of the base model
            target_dim: Desired output dimension
        """
        self.model = SentenceTransformer(original_model)
        self.original_dim = self.model.get_sentence_embedding_dimension()
        self.target_dim = target_dim
        
        # Add linear dimension reduction layer
        self.model._first_module().auto_model.pooler = nn.Linear(
            self.original_dim, 
            self.target_dim
        )

def get_embeddings_fixed_dim(texts: List[str], dimension: int) -> List[List[float]]:
    """
    Generate embeddings with specified dimension
    
    Method 1: Choose a model with desired dimensions
    """
    if dimension == 384:
        model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
    elif dimension == 768:
        model = SentenceTransformer('all-mpnet-base-v2')  # 768 dimensions
    elif dimension <= 512:
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # 384 dimensions
    else:
        model = SentenceTransformer('all-mpnet-base-v2')  # 768 dimensions
    
    embeddings = model.encode(texts)
    return embeddings.tolist()

def get_embeddings_reduced_dim(texts: List[str], target_dim: int) -> List[List[float]]:
    """
    Method 2: Use dimension reduction after embedding
    """
    # Get original embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    
    # Perform PCA for dimension reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=target_dim)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    return reduced_embeddings.tolist()

def get_embeddings_custom_dim(texts: List[str], target_dim: int) -> List[List[float]]:
    """
    Method 3: Use custom dimension reducer
    """
    reducer = DimensionReducer('all-MiniLM-L6-v2', target_dim)
    embeddings = reducer.model.encode(texts)
    return embeddings.tolist()