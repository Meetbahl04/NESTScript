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

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import torch
from typing import Dict, List, Tuple
import time

class SimilarityRetriever:
    def __init__(self, 
                 sbert_embeddings: np.ndarray, 
                 tfidf_embeddings: csr_matrix,
                 alpha: float = 0.7):
        """
        Initialize with both types of embeddings
        
        Args:
            sbert_embeddings: SBERT embeddings array
            tfidf_embeddings: TF-IDF embeddings sparse matrix
            alpha: Weight for SBERT similarity (1-alpha for TF-IDF)
        """
        # Verify and convert embeddings to float32
        self.sbert_embeddings = np.array(sbert_embeddings, dtype=np.float32)
        self.tfidf_embeddings = tfidf_embeddings.astype(np.float32)
        
        self.alpha = alpha
        
        # Normalize SBERT embeddings for cosine similarity
        sbert_norms = np.linalg.norm(self.sbert_embeddings, axis=1, keepdims=True)
        sbert_norms[sbert_norms == 0] = 1e-10
        self.sbert_normalized = torch.from_numpy(self.sbert_embeddings / sbert_norms).cuda()
        
        # TF-IDF is kept sparse and normalized already
        # Assuming TF-IDF is already L2 normalized; if not, normalize it
        # For sparse matrices, assume they are normalized
        
        print(f"Initialized with shapes: SBERT {self.sbert_embeddings.shape}, TF-IDF {self.tfidf_embeddings.shape}")
    
    def find_similar_trials(self, query_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find top-k most similar trials to the query trial
        
        Args:
            query_idx: Index of the query trial
            top_k: Number of top similar trials to retrieve
        
        Returns:
            List of tuples (index, combined_similarity_score)
        """
        # Compute SBERT similarities
        query_sbert = self.sbert_normalized[query_idx].unsqueeze(0)
        sbert_similarities = torch.mm(query_sbert, self.sbert_normalized.t()).squeeze()
        sbert_similarities = sbert_similarities.cpu().numpy()
        
        # Compute TF-IDF similarities
        query_tfidf = self.tfidf_embeddings[query_idx]
        tfidf_similarities = self.tfidf_embeddings.dot(query_tfidf.T).A.flatten()
        
        # Combine similarities
        combined_similarities = (self.alpha * sbert_similarities +
                                 (1 - self.alpha) * tfidf_similarities)
        
        # Get top-k indices
        top_indices = np.argsort(combined_similarities)[::-1][:top_k]
        top_scores = combined_similarities[top_indices]
        
        return list(zip(top_indices, top_scores))
    
    def analyze_similarity_components(self, query_idx: int, trial_idx: int) -> Dict[str, float]:
        """
        Analyze similarity components for a pair of trials
        
        Args:
            query_idx: Index of the query trial
            trial_idx: Index of the target trial
        
        Returns:
            Dictionary with 'sbert_similarity' and 'tfidf_similarity'
        """
        sbert_sim = torch.dot(self.sbert_normalized[query_idx], self.sbert_normalized[trial_idx]).item()
        tfidf_sim = self.tfidf_embeddings[trial_idx].dot(self.tfidf_embeddings[query_idx].T).toarray()[0, 0]
        
        return {
            'sbert_similarity': sbert_sim,
            'tfidf_similarity': tfidf_sim
        }

# Loading and verification function
def load_embeddings(sbert_path: str, tfidf_path: str) -> Tuple[np.ndarray, csr_matrix]:
    """
    Load and verify embeddings from files
    
    Args:
        sbert_path: Path to SBERT embeddings
        tfidf_path: Path to TF-IDF embeddings
    
    Returns:
        Tuple of (SBERT embeddings, TF-IDF embeddings as sparse matrix)
    """
    try:
        print(f"Loading SBERT embeddings from {sbert_path}")
        sbert = np.load(sbert_path)
        print(f"SBERT embeddings shape: {sbert.shape}")
        
        print(f"Loading TF-IDF embeddings from {tfidf_path}")
        tfidf_sparse = np.load(tfidf_path, allow_pickle=True)
        tfidf = csr_matrix((tfidf_sparse['data'], 
                            tfidf_sparse['indices'], 
                            tfidf_sparse['indptr']),
                           shape=tuple(tfidf_sparse['shape']))
        print(f"TF-IDF embeddings shape: {tfidf.shape}")
        
        # Verify data types and contents
        if not np.issubdtype(sbert.dtype, np.number):
            raise ValueError(f"SBERT embeddings contain non-numeric data: {sbert.dtype}")
        if not np.issubdtype(tfidf.dtype, np.number):
            raise ValueError(f"TF-IDF embeddings contain non-numeric data: {tfidf.dtype}")

        # Check for NaN or infinite values
        if np.isnan(sbert).any() or np.isinf(sbert).any():
            raise ValueError("SBERT embeddings contain NaN or infinite values")
        if np.isnan(tfidf.data).any() or np.isinf(tfidf.data).any():
            raise ValueError("TF-IDF embeddings contain NaN or infinite values")
            
        return sbert, tfidf
        
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        raise

# Load and check embeddings
sbert_path = '/kaggle/input/nest-ps-1-sbert-embeddings/sbert_embeddings_final_20241228-170631.npy'
tfidf_path = '/kaggle/input/nest-ps-1-tfidf-embeddings/tfidf_vectors_final_20241228-170631.npz'

sbert, tfidf_sparse = load_embeddings(sbert_path, tfidf_path)

print("SBERT shape:", sbert.shape)
print("SBERT dtype:", sbert.dtype)
print("TF-IDF shape:", tfidf_sparse.shape)
print("TF-IDF dtype:", tfidf_sparse.dtype)

# Now initialize the retriever with the dense arrays
retriever = SimilarityRetriever(sbert, tfidf_sparse, alpha=0.7)

# Test the retriever
query_idx = 0
similar_trials = retriever.find_similar_trials(query_idx, top_k=10)

# Print results
print(f"\nMost similar trials to trial {query_idx}:")
for idx, score in similar_trials:
    analysis = retriever.analyze_similarity_components(query_idx, idx)
    print(f"\nTrial {idx}")
    print(f"Combined similarity: {score:.4f}")
    print(f"SBERT similarity: {analysis['sbert_similarity']:.4f}")
    print(f"TF-IDF similarity: {analysis['tfidf_similarity']:.4f}")
