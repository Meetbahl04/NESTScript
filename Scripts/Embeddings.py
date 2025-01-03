from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np
from typing import List, Dict
from tqdm.auto import tqdm
import time
import psutil
import gc
import os

def get_memory_usage():
    process = psutil.Process()
    memory_use = process.memory_info().rss / (1024 ** 2)  # in MB
    return memory_use

class TrialEncoder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        print(f"Memory before loading SBERT model: {get_memory_usage():.2f} MB")
        
        # Load model with optimizations
        self.sbert_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        
        # Convert to half precision if using CUDA
        if self.device == 'cuda':
            self.sbert_model.half()  # Use FP16 for faster processing
        
        self.sbert_model.to(self.device)
        print(f"Memory after loading SBERT model: {get_memory_usage():.2f} MB")
        # Initialize TF-IDF with optimized parameters
        self.tfidf = TfidfVectorizer(
            max_features=10000, 
            stop_words='english',
            dtype=np.float32  # Use float32 instead of float64 for memory efficiency
        )
    
    def combine_text_fields(self, row: pd.Series) -> str:
        return (f"TITLE: {row['Study Title_Cleaned']} "
                f"PRIMARY: {row['Primary Outcome Measures_Cleaned']} "
                f"SECONDARY: {row['Secondary Outcome Measures_Cleaned']} "
                f"CRITERIA: {row['criteria_Cleaned']}")

    @torch.cuda.amp.autocast()  # Enable automatic mixed precision
    def encode_sbert(self, texts: List[str]) -> np.ndarray:
        return self.sbert_model.encode(
            texts,
            show_progress_bar=True,
            device=self.device,
            batch_size=256,  # Increased batch size for better GPU utilization
            normalize_embeddings=True,
            convert_to_numpy=True
        )
    
    def encode_tfidf(self, texts: List[str]) -> np.ndarray:
        return self.tfidf.fit_transform(texts).astype(np.float32).toarray()

def process_in_chunks(df: pd.DataFrame, chunk_size: int = 5000, save_path: str = './embeddings/') -> None:
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize encoder
    encoder = TrialEncoder()
    total_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
    
    # Initialize arrays and progress bar
    all_sbert = []
    all_tfidf = []
    # Main progress bar
    with tqdm(total=total_chunks, desc="Processing chunks", position=0) as chunk_pbar:
        try:
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(df))
                
                # Update progress description
                chunk_pbar.set_description(
                    f"Chunk {chunk_idx + 1}/{total_chunks} [Rows {start_idx}-{end_idx}]"
                )
                
                # Process chunk
                chunk_df = df.iloc[start_idx:end_idx]
                
                # Combine texts efficiently
                combined_texts = [
                    encoder.combine_text_fields(row) 
                    for _, row in tqdm(chunk_df.iterrows(), 
                                     desc="Combining texts", 
                                     position=1, 
                                     leave=False)
                ]
                
                # Generate embeddings
                sbert_embeddings = encoder.encode_sbert(combined_texts)
                all_sbert.append(sbert_embeddings)
                
                tfidf_vectors = encoder.encode_tfidf(combined_texts)
                all_tfidf.append(tfidf_vectors)
                
                # Clear memory
                del combined_texts, sbert_embeddings, tfidf_vectors
                gc.collect()
                torch.cuda.empty_cache()
                
                chunk_pbar.update(1)
                
               # Save backup every 5 chunks
                if (chunk_idx + 1) % 5 == 0:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    backup_path = os.path.join(save_path, 'backups')
                    os.makedirs(backup_path, exist_ok=True)
                    
                    np.save(os.path.join(backup_path, f'sbert_backup_{timestamp}.npy'), 
                           np.vstack(all_sbert))
                    np.save(os.path.join(backup_path, f'tfidf_backup_{timestamp}.npy'), 
                           np.vstack(all_tfidf))
                    
                    chunk_pbar.write(f"Backup saved at chunk {chunk_idx + 1}")
            
            # Save final combined embeddings
            chunk_pbar.write("\nSaving final embeddings...")
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            final_sbert = np.vstack(all_sbert)
            np.save(os.path.join(save_path, f'sbert_embeddings_final_{timestamp}.npy'), 
                   final_sbert)
            chunk_pbar.write(f"Final SBERT shape: {final_sbert.shape}")
            del final_sbert
            
            final_tfidf = np.vstack(all_tfidf)
            np.save(os.path.join(save_path, f'tfidf_vectors_final_{timestamp}.npy'), 
                   final_tfidf)
            chunk_pbar.write(f"Final TF-IDF shape: {final_tfidf.shape}")
            del final_tfidf
            
        except Exception as e:
            chunk_pbar.write(f"Error occurred: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        total_start_time = time.time()
        print(f"Starting processing with initial memory: {get_memory_usage():.2f} MB")
        
        process_in_chunks(df, chunk_size=4000)
        
        total_time = time.time() - total_start_time
        print(f"\nTotal execution time: {total_time/60:.2f} minutes")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
