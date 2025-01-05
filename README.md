**Project Overview**
--------------------

This project aims to solve the problem of clinical trial similarity and matching using a **hybrid contrastive learning approach**. The objective is to create a deep learning pipeline capable of identifying similar clinical trials based on text and numerical features. The system utilizes transformer-based models like **SciBERT**, **TF-IDF embeddings**, and **numeric features** for a multi-modal representation of clinical trial data. It incorporates a **Siamese Network architecture** and **contrastive loss** to distinguish between similar and dissimilar pairs of trials.

**Table of Contents**
---------------------

1.  Dataset Preparation
    
2.  Data Preprocessing
    
3.  Embeddings Generation
    
4.  Training Pair Generation
    
5.  Model Architecture
    
6.  Training Pipeline
    
7.  Validation and Metrics
    
8.  Project Phases

**1\. Dataset Preparation**
---------------------------

Clinical trial data is sourced from multiple files in CSV and Excel formats, containing details such as:

*   Study Title
    
*   Primary Outcome Measures
    
*   Secondary Outcome Measures
    
*   Eligibility Criteria
    
*   Funder TypeThese datasets are combined, cleaned, and preprocessed to ensure uniformity and usability for downstream tasks.
    

Key Steps:

*   **Merging datasets:** Data is merged on common identifiers like nct\_id.
    
*   **Handling missing values:** Flags for missing features are created, and placeholder text is added to maintain consistency.
    
*   **Data completeness score:** A scoring mechanism is implemented to quantify the completeness of each trial's information.
    

**2\. Data Preprocessing**
--------------------------

Preprocessing is critical for ensuring high-quality input for the model. This includes:

*   **Text Cleaning:**
    
    *   Expansion of medical abbreviations (e.g., htn -> hypertension, t2dm -> type 2 diabetes mellitus).
        
    *   Removal of special characters, HTML tags, and bullet points.
        
    *   Standardization of units (e.g., 2 years -> 2 year).
        
*   **Feature Engineering:**
    
    *   Text length calculations for different fields.
        
    *   Flags for missing primary/secondary outcome measures and eligibility criteria.
        
*   **Preprocessing Pipeline:** A function preprocess\_text is implemented to perform comprehensive text cleaning for all relevant fields.
    

Output:

*   A cleaned dataset with additional features such as \_Cleaned fields, missing value flags, and completeness scores.
    

**3\. Embeddings Generation**
-----------------------------

The project uses two types of embeddings to represent clinical trial data:

1.  **SciBERT Embeddings (Dense):**
    
    *   A **Sentence-BERT** model (pritamdeka/S-PubMedBert-MS-MARCO) captures semantic relationships between trials.
        
    *   Mixed precision FP16 is used for memory efficiency.
        
2.  **TF-IDF Embeddings (Sparse):**
    
    *   A **TF-IDF vectorizer** is used for high-dimensional sparse representations of textual fields.
        

### Processing Pipeline:

*   Text fields are combined into a single string for each trial.
    
*   Embeddings are generated in chunks to handle large datasets efficiently.
    
*   Generated embeddings are saved as .npy files for SBERT and sparse matrix files for TF-IDF.
    

**4\. Training Pair Generation**
--------------------------------

To train the Siamese Network, **positive** and **negative pairs** of clinical trials are generated.

### **Positive Pairs:**

*   Trials with high similarity in both SBERT and TF-IDF embeddings (e.g., SBERT similarity > 0.8, TF-IDF similarity > 0.5).
    
*   Positive pairs represent trials that are semantically similar.
    

### **Negative Pairs:**

*   Randomly sampled trial pairs with low similarity (SBERT similarity < 0.4, TF-IDF similarity < 0.2).
    
*   Negative pairs ensure the model learns to distinguish dissimilar trials effectively.
    

### **Pair Dataset:**

*   A balanced dataset with a 1:1 ratio of positive and negative pairs is created.
    
*   The dataset is split into **training** and **validation** sets with a typical 80:20 split.
    

**5\. Model Architecture**
--------------------------

The project uses a **Hybrid Siamese Network** with the following components:

1.  **SciBERT Encoder:**
    
    *   Extracts dense embeddings from textual fields.
        
    *   Uses the CLS token representation from the transformer model.
        
2.  **TF-IDF Encoder:**
    
    *   A feed-forward layer reduces the TF-IDF vector dimensionality.
        
3.  **Numeric Feature Encoder:**
    
    *   Encodes numeric features like completeness score and missing value flags.
        
4.  **Fusion Layer:**
    
    *   Combines the outputs of SciBERT, TF-IDF, and numeric encoders.
        
    *   Projects the combined representation into a lower-dimensional space.
        

The model is trained using **contrastive loss**, which minimizes the distance between similar pairs while maximizing the distance for dissimilar ones.

**6\. Training Pipeline**
-------------------------

The training pipeline includes the following components:

1.  **Data Loading:**
    
    *   A custom PyTorch Dataset processes tokenized fields, TF-IDF vectors, and numeric features for each trial pair.
        
2.  **Training Loop:**
    
    *   The Siamese Network is trained using **contrastive loss**.
        
    *   Gradients are clipped to avoid exploding gradients.
        
3.  **Learning Rate Scheduler:**
    
    *   A ReduceLROnPlateau scheduler adjusts the learning rate based on validation loss.
        
4.  **Checkpointing:**
    
    *   Best-performing models are saved for future evaluation.
        

**7\. Validation and Metrics**
------------------------------

The project uses several metrics to evaluate the model's performance:

1.  **Mean Reciprocal Rank (MRR):**
    
    *   Measures the ranking quality of similar trials.
        
2.  **Precision@K:**
    
    *   Evaluates the percentage of relevant trials in the top K results.
        
3.  **Validation Loss:**
    
    *   Average contrastive loss on the validation set.
        

**8\. Project Phases**
----------------------

The project is structured into four phases:

### **Phase 1: Data Preparation and Label Generation**

*   Combine datasets and preprocess text fields.
    
*   Generate positive and negative trial pairs.
    

### **Phase 2: Model Architecture Design**

*   Implement a **Siamese Network** with hybrid encoders for text, TF-IDF, and numeric features.
    

### **Phase 3: Training Pipeline Setup**

*   Design training and validation loops.
    
*   Implement logging, checkpointing, and learning rate scheduling.
    

### **Phase 4: Model Evaluation**

*   Evaluate the trained model using metrics like MRR and Precision@K.
    
*   Analyze the results and fine-tune hyperparameters.



This project provides an effective solution for clinical trial similarity matching, leveraging state-of-the-art transformer models and multi-modal data integration. Future improvements can enhance the system's robustness and expand its applicability in the medical domain.
