import pandas as pd
import numpy as np
from typing import Dict, List

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive missing value handling based on the proposed strategy
    """
    # Create a copy to avoid modifying original
    processed_df = df.copy()
    
    # 1. Secondary Outcome Measures (High missing rate)
    # We already have Secondary_Outcome_Missing flag
    # Replace missing values with standardized text
    processed_df['Secondary Outcome Measures_Cleaned'] = processed_df['Secondary Outcome Measures_Cleaned'].fillna('no secondary outcomes provided')
    
    # 2. Primary Outcome Measures (Moderate missing rate)
    # Create flag for missing primary outcomes
    processed_df['Primary_Outcome_Missing'] = processed_df['Primary Outcome Measures_Cleaned'].isna().astype(int)
    
    # Replace missing values with standardized text
    # We're using a placeholder here - in the next phase, we can implement similar-trial imputation
    processed_df['Primary Outcome Measures_Cleaned'] = processed_df['Primary Outcome Measures_Cleaned'].fillna('primary outcome not specified')
    
    # 3. Criteria (Low missing rate)
    # Create flag for missing criteria
    processed_df['Criteria_Missing'] = processed_df['criteria_Cleaned'].isna().astype(int)
    
    # Replace missing values with standardized text
    processed_df['criteria_Cleaned'] = processed_df['criteria_Cleaned'].fillna('no criteria provided')
    
    # 4. Create a data completeness score
    processed_df['completeness_score'] = 1.0
    # Reduce score for missing values
    processed_df.loc[processed_df['Secondary_Outcome_Missing'] == 1, 'completeness_score'] -= 0.2
    processed_df.loc[processed_df['Primary_Outcome_Missing'] == 1, 'completeness_score'] -= 0.4
    processed_df.loc[processed_df['Criteria_Missing'] == 1, 'completeness_score'] -= 0.4
    
    return processed_df

def generate_missing_data_report(original_df: pd.DataFrame, processed_df: pd.DataFrame) -> None:
    """
    Generate a detailed report of missing data handling
    """
    print("Missing Data Report")
    print("-" * 50)
    
    # Original missing values
    print("\nOriginal Missing Values:")
    for column in original_df.columns:
        missing_count = original_df[column].isna().sum()
        total_count = len(original_df)
        missing_percentage = (missing_count / total_count) * 100
        print(f"{column}: {missing_count} missing ({missing_percentage:.2f}%)")
    
    # Processed missing values
    print("\nProcessed Missing Values:")
    for column in processed_df.columns:
        missing_count = processed_df[column].isna().sum()
        total_count = len(processed_df)
        missing_percentage = (missing_count / total_count) * 100
        print(f"{column}: {missing_count} missing ({missing_percentage:.2f}%)")
    
    # Completeness score statistics
    print("\nCompleteness Score Statistics:")
    print(f"Mean: {processed_df['completeness_score'].mean():.2f}")
    print(f"Median: {processed_df['completeness_score'].median():.2f}")
    print(f"Min: {processed_df['completeness_score'].min():.2f}")
    print(f"Max: {processed_df['completeness_score'].max():.2f}")

# Main execution
if __name__ == "__main__":
    # Process the dataframe
    processed_df = handle_missing_values(df)
    
    # Generate report
    generate_missing_data_report(df, processed_df)
    
    # Save processed dataframe
    processed_df.to_csv('clinical_trials_processed_with_missing_handled.csv', index=False)
