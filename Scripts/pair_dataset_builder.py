SBERT_THRESHOLD = 0.75
TFIDF_THRESHOLD = 0.5
POS_NEG_RATIO = 1.0
TOTAL_PAIRS = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 256  # More conservative batch size

torch.manual_seed(42)
np.random.seed(42)



pair_generator = DataPairGenerator(
    df=df_subset,
    sbert_embeddings=sbert_subset,
    tfidf_embeddings=tfidf_subset,
    sbert_threshold=SBERT_THRESHOLD,
    tfidf_threshold=TFIDF_THRESHOLD,
    pos_neg_ratio=POS_NEG_RATIO,
    batch_size=BATCH_SIZE
)

print(f"\nGenerating training pairs (total pairs: {TOTAL_PAIRS})...")
try:
    with timer("Total dataset creation"):  # Changed from test_generator.timer
        pair_datasets = pair_generator.create_training_pairs(
            total_pairs=TOTAL_PAIRS,
            validation_split=VALIDATION_SPLIT
        )
        
    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    print("\nSaving datasets...")
    train_path = f'training_pairs_{timestamp}.csv'
    pair_datasets['train'].to_csv(train_path, index=False)
    print(f"Training dataset saved to: {train_path}")
    
    val_path = f'validation_pairs_{timestamp}.csv'
    pair_datasets['validation'].to_csv(val_path, index=False)
    print(f"Validation dataset saved to: {val_path}")
    print("\nDataset Statistics:")
    print(f"Training pairs: {len(pair_datasets['train']):,}")
    print(f"Validation pairs: {len(pair_datasets['validation']):,}")
    
    train_pos = (pair_datasets['train']['label'] == 1).sum()
    train_neg = (pair_datasets['train']['label'] == 0).sum()
    val_pos = (pair_datasets['validation']['label'] == 1).sum()
    val_neg = (pair_datasets['validation']['label'] == 0).sum()
    
    print("\nClass Distribution:")
    print(f"Training - Positive: {train_pos:,} ({train_pos/len(pair_datasets['train'])*100:.1f}%)")
    print(f"Training - Negative: {train_neg:,} ({train_neg/len(pair_datasets['train'])*100:.1f}%)")
    print(f"Validation - Positive: {val_pos:,} ({val_pos/len(pair_datasets['validation'])*100:.1f}%)")
    print(f"Validation - Negative: {val_neg:,} ({val_neg/len(pair_datasets['validation'])*100:.1f}%)")

except Exception as e:
    print(f"\nError during pair generation: {str(e)}")
    raise
