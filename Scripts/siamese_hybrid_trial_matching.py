import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SiameseNetwork(nn.Module):
    def __init__(self, model_name='allenai/scibert_scivocab_uncased', hidden_size=768):
        super(SiameseNetwork, self).__init__()
        
        # Initialize BERT encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward_once(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use CLS token representation
        pooled = outputs.last_hidden_state[:, 0]
        
        # Project to final embedding
        projected = self.projection(pooled)
        return projected
        
    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        # Get embeddings for both trials
        emb1 = self.forward_once(input_ids1, attention_mask1)
        emb2 = self.forward_once(input_ids2, attention_mask2)
        return emb1, emb2

class HybridEncoder(nn.Module):
    def __init__(self, 
                 model_name='allenai/scibert_scivocab_uncased', 
                 text_dim=768,
                 tfidf_dim=5000,
                 numeric_dim=3):  # completeness_score + 2 missing flags
        super(HybridEncoder, self).__init__()
        
        # Text encoder (BERT)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        
        # TF-IDF encoder
        self.tfidf_encoder = nn.Sequential(
            nn.Linear(tfidf_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # Numeric features encoder
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Fusion layer
        combined_dim = text_dim + 128 + 16  # BERT + TF-IDF + numeric
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
    def forward_once(self, input_ids, attention_mask, tfidf_vector, numeric_features):
        # Get BERT embeddings
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_embeds = text_outputs.last_hidden_state[:, 0]
        
        # Encode TF-IDF
        tfidf_embeds = self.tfidf_encoder(tfidf_vector)
        
        # Encode numeric features
        numeric_embeds = self.numeric_encoder(numeric_features)
        
        # Combine all features
        combined = torch.cat([text_embeds, tfidf_embeds, numeric_embeds], dim=1)
        
        # Final fusion
        fused = self.fusion(combined)
        return fused
        
    def forward(self, batch):
        # Process first trial
        emb1 = self.forward_once(
            batch['input_ids1'],
            batch['attention_mask1'],
            batch['tfidf1'],
            batch['numeric1']
        )
        
        # Process second trial
        emb2 = self.forward_once(
            batch['input_ids2'],
            batch['attention_mask2'],
            batch['tfidf2'],
            batch['numeric2']
        )
        
        return emb1, emb2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, emb1, emb2, label):
        # Calculate euclidean distance
        distance = F.pairwise_distance(emb1, emb2)
        
        # Contrastive loss
        loss = torch.mean((1 - label) * torch.pow(distance, 2) + 
                         label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss

class TrialDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Tokenize text fields
        text1 = f"{row['Study Title_Cleaned_1']} {row['Primary Outcome Measures_Cleaned_1']} {row['criteria_Cleaned_1']}"
        text2 = f"{row['Study Title_Cleaned_2']} {row['Primary Outcome Measures_Cleaned_2']} {row['criteria_Cleaned_2']}"
        
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

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        emb1, emb2 = model(batch)
        
        # Compute loss
        loss = criterion(emb1, emb2, batch['label'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
