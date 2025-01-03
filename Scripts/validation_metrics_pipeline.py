class ValidationMetrics:
    def __init__(self, k_values=[1, 5, 10]):
        self.k_values = k_values
    
    def precision_at_k(self, similarities, labels, k):
        """Calculate Precision@K"""
        _, top_k_indices = similarities.topk(k)
        relevant = labels[top_k_indices].float()
        return (relevant.sum() / k).mean().item()
    
    def mean_reciprocal_rank(self, similarities, labels):
        """Calculate MRR"""
        ranks = torch.where(labels[similarities.argsort(descending=True)])[0] + 1
        return (1.0 / ranks.float()).mean().item()
    
    def compute_metrics(self, embeddings1, embeddings2, labels):
        """Compute all validation metrics"""
        # Compute similarity matrix
        similarities = F.cosine_similarity(embeddings1.unsqueeze(1), 
                                        embeddings2.unsqueeze(0), dim=2)
        
        metrics = {
            'mrr': self.mean_reciprocal_rank(similarities, labels)
        }
        
        for k in self.k_values:
            metrics[f'p@{k}'] = self.precision_at_k(similarities, labels, k)
            
        return metrics

def validate(model, val_loader, criterion, metrics, device):
    """Validation step"""
    model.eval()
    total_loss = 0
    all_emb1, all_emb2, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Forward pass
            emb1, emb2 = model(batch)
            
            # Compute loss
            loss = criterion(emb1, emb2, batch['label'])
            total_loss += loss.item()
            
            # Store embeddings and labels for metrics
            all_emb1.append(emb1)
            all_emb2.append(emb2)
            all_labels.append(batch['label'])
    
    # Concatenate all batches
    all_emb1 = torch.cat(all_emb1)
    all_emb2 = torch.cat(all_emb2)
    all_labels = torch.cat(all_labels)
    
    # Compute metrics
    validation_metrics = metrics.compute_metrics(all_emb1, all_emb2, all_labels)
    validation_metrics['loss'] = total_loss / len(val_loader)
    
    return validation_metrics
