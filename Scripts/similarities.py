import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import time

class SimilarityRetriever:
    def __init__(self, 
                 sbert_embeddings: np.ndarray, 
                 tfidf_embeddings: np.ndarray,
                 alpha: float = 0.7):
        """
        Initialize with both types of embeddings
        
        Args:
            sbert_embeddings: SBERT embeddings array
            tfidf_embeddings: TF-IDF embeddings array
            alpha: Weight for SBERT similarity (1-alpha for TF-IDF)
        """
        # Verify and convert embeddings to float32
        try:
            self.sbert_embeddings = np.array(sbert_embeddings, dtype=np.float32)
            self.tfidf_embeddings = np.array(tfidf_embeddings, dtype=np.float32)
        except ValueError as e:
            print("Error converting embeddings to float32. Please check your embedding files.")
            raise
            
        self.alpha = alpha
        
        # Normalize embeddings for faster cosine similarity
        try:
            self.sbert_normalized = self._normalize_embeddings(self.sbert_embeddings)
            self.tfidf_normalized = self._normalize_embeddings(self.tfidf_embeddings)
        except Exception as e:
            print(f"Error during normalization: {str(e)}")
            raise
        
        print(f"Initialized with shapes: SBERT {self.sbert_embeddings.shape}, TF-IDF {self.tfidf_embeddings.shape}")
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-10
        return embeddings / norms

# Loading and verification function
def load_embeddings(sbert_path: str, tfidf_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and verify embeddings from files
    
    Args:
        sbert_path: Path to SBERT embeddings
        tfidf_path: Path to TF-IDF embeddings
    
    Returns:
        Tuple of (SBERT embeddings, TF-IDF embeddings)
    """
    try:
        print(f"Loading SBERT embeddings from {sbert_path}")
        sbert = np.load(sbert_path)
        print(f"SBERT embeddings shape: {sbert.shape}")
        
        print(f"Loading TF-IDF embeddings from {tfidf_path}")
        tfidf = np.load(tfidf_path)
        print(f"TF-IDF embeddings shape: {tfidf.shape}")
        
        # Verify data types and contents
        if not np.issubdtype(sbert.dtype, np.number):
            raise ValueError(f"SBERT embeddings contain non-numeric data: {sbert.dtype}")
        if not np.issubdtype(tfidf.dtype, np.number):
            raise ValueError(f"TF-IDF embeddings contain non-numeric data: {tfidf.dtype}")
             # Check for NaN or infinite values
        if np.isnan(sbert).any() or np.isinf(sbert).any():
            raise ValueError("SBERT embeddings contain NaN or infinite values")
        if np.isnan(tfidf).any() or np.isinf(tfidf).any():
            raise ValueError("TF-IDF embeddings contain NaN or infinite values")
            
        return sbert, tfidf
        
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        raise
