import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os
import numpy as np
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class TrialDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Combine text fields
        text1 = f"{row['Study Title_Cleaned_1']} {row['Primary Outcome Measures_Cleaned_1']} {row['criteria_Cleaned_1']}"
        text2 = f"{row['Study Title_Cleaned_2']} {row['Primary Outcome Measures_Cleaned_2']} {row['criteria_Cleaned_2']}"
        
        # Tokenize texts
        encoded1 = self.tokenizer(
            text1,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        encoded2 = self.tokenizer(
            text2,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Prepare numeric features
        numeric1 = torch.tensor([
            row['completeness_score_1'],
            row['Secondary_Outcome_Missing_1'],
            row['Primary_Outcome_Missing_1']
        ], dtype=torch.float)
        
        numeric2 = torch.tensor([
            row['completeness_score_2'],
            row['Secondary_Outcome_Missing_2'],
            row['Primary_Outcome_Missing_2']
        ], dtype=torch.float)
        
        return {
            'input_ids1': encoded1['input_ids'].squeeze(),
            'attention_mask1': encoded1['attention_mask'].squeeze(),
            'input_ids2': encoded2['input_ids'].squeeze(),
            'attention_mask2': encoded2['attention_mask'].squeeze(),
            'numeric1': numeric1,
            'numeric2': numeric2,
            'label': torch.tensor(row['label'], dtype=torch.float)
        }
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, emb1, emb2, label):
        distance = F.pairwise_distance(emb1, emb2)
        loss = torch.mean((1 - label) * torch.pow(distance, 2) + 
                         label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss

class ValidationMetrics:
    def __init__(self, k_values=[1, 5, 10]):
        self.k_values = k_values
    
    def precision_at_k(self, similarities, labels, k):
        _, top_k_indices = similarities.topk(k)
        relevant = labels[top_k_indices].float()
        return (relevant.sum() / k).mean().item()
    
    def mean_reciprocal_rank(self, similarities, labels):
        ranks = torch.where(labels[similarities.argsort(descending=True)])[0] + 1
        return (1.0 / ranks.float()).mean().item()
    
    def compute_metrics(self, embeddings1, embeddings2, labels):
        similarities = F.cosine_similarity(embeddings1.unsqueeze(1), 
                                        embeddings2.unsqueeze(0), dim=2)
        
        metrics = {
            'mrr': self.mean_reciprocal_rank(similarities, labels)
        }
        
        for k in self.k_values:
            metrics[f'p@{k}'] = self.precision_at_k(similarities, labels, k)
            
        return metrics
class MetricLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_mrr': [],
            'learning_rate': [],
            'epoch': []
        }
        for k in [1, 5, 10]:
            self.metrics[f'val_p@{k}'] = []
            
    def log_metrics(self, epoch, train_loss, val_metrics, current_lr):
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_metrics['loss'])
        self.metrics['val_mrr'].append(val_metrics['mrr'])
        self.metrics['learning_rate'].append(current_lr)
        
        for k in [1, 5, 10]:
            self.metrics[f'val_p@{k}'].append(val_metrics[f'p@{k}'])
    
    def save_metrics(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        df = pd.DataFrame(self.metrics)
        csv_path = self.log_dir / f'metrics_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        
        json_path = self.log_dir / f'metrics_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        return csv_path, json_path
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        try:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            emb1, emb2 = model(batch)
            loss = criterion(emb1, emb2, batch['label'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        except RuntimeError as e:
            print(f"Error in batch: {str(e)}")
            continue
            
    return total_loss / len(train_loader)
def validate(model, val_loader, criterion, metrics, device):
    model.eval()
    total_loss = 0
    all_emb1, all_emb2, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            emb1, emb2 = model(batch)
            loss = criterion(emb1, emb2, batch['label'])
            
            total_loss += loss.item()
            all_emb1.append(emb1.cpu())
            all_emb2.append(emb2.cpu())
            all_labels.append(batch['label'].cpu())
            
            torch.cuda.empty_cache()
    
    all_emb1 = torch.cat(all_emb1)
    all_emb2 = torch.cat(all_emb2)
    all_labels = torch.cat(all_labels)
    
    validation_metrics = metrics.compute_metrics(all_emb1, all_emb2, all_labels)
    validation_metrics['loss'] = total_loss / len(val_loader)
    
    return validation_metrics
def setup_training(config):
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    train_dataset = TrialDataset(balanced_train_df, tokenizer, max_length=config['max_length'])
    val_dataset = TrialDataset(val_df, tokenizer, max_length=config['max_length'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    model = HybridEncoder(
        model_name=config['model_name'],
        text_dim=config['text_dim'],
        tfidf_dim=config['tfidf_dim'],
        numeric_dim=config['numeric_dim']
    ).to(config['device'])
    
    return model, train_loader, val_loader
def save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f'checkpoint_epoch_{epoch}_{timestamp}.pt'
    )
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
    }, checkpoint_path)
    
    return checkpoint_path
def train_model(model, train_loader, val_loader, config):
    try:
        logger = MetricLogger(config['log_dir'])
        
        criterion = ContrastiveLoss(margin=config['margin'])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        metrics = ValidationMetrics(k_values=[1, 5, 10])
        best_val_loss = float('inf')
        best_model_path = None
        
        for epoch in range(config['epochs']):
            print(f"\nEpoch {epoch+1}/{config['epochs']}")
            
            train_loss = train_epoch(model, train_loader, criterion, optimizer, config['device'])
            val_metrics = validate(model, val_loader, criterion, metrics, config['device'])
            
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_metrics(epoch, train_loss, val_metrics, current_lr)
            
            scheduler.step(val_metrics['loss'])
            
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch, val_metrics, config['checkpoint_dir']
            )
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
                torch.save(model.state_dict(), best_model_path)
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_metrics['loss']:.4f}")
            print(f"Validation MRR: {val_metrics['mrr']:.4f}")
            for k in metrics.k_values:
                print(f"Validation P@{k}: {val_metrics[f'p@{k}']:.4f}")
            
            if val_metrics['loss'] > best_val_loss * (1 + config['early_stopping_threshold']):
                config['patience_counter'] += 1
                if config['patience_counter'] >= config['patience']:
                    print("Early stopping triggered")
                    break
            else:
                config['patience_counter'] = 0
        
        csv_path, json_path = logger.save_metrics()
        print(f"\nMetrics saved to:")
        print(f"CSV: {csv_path}")
        print(f"JSON: {json_path}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        if 'model' in locals() and 'optimizer' in locals():
            save_checkpoint(model, optimizer, epoch, val_metrics, config['checkpoint_dir'])
        raise e
config = {
    'model_name': 'allenai/scibert_scivocab_uncased',
    'text_dim': 768,
    'tfidf_dim': 5000,
    'numeric_dim': 3,
    'max_length': 512,
    'batch_size': 16,
    'num_workers': 4,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'margin': 1.0,
    'epochs': 10,
    'patience': 3,
    'early_stopping_threshold': 0.01,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'training_logs',
    'seed': 42,
    'patience_counter': 0,
    'grad_clip': 1.0
}

if __name__ == "__main__":
    print("Initializing training...")
    model, train_loader, val_loader = setup_training(config)
    
    print("Starting training...")
    train_model(model, train_loader, val_loader, config)
