#!/usr/bin/env python3
"""Load saved models (classifier_RF) and run predictions on new data."""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import argparse

# Add src directory to path to import classifier_RF functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier_RF import (
    load_model, 
    predict_with_saved_model, 
    FAMILY_TO_COLS,
    split_families
)

def load_and_predict(
    data_path: str,
    model_dir: str, 
    family: str,
    task_type: str,
    use_final_model: bool = True
) -> pd.DataFrame:
    """Run predictions for one family using a saved binary/multiclass model."""
    # Load data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("Data file must be CSV or Parquet")
    
    # Extract features for the specified family
    families_clean = split_families(df, {family: FAMILY_TO_COLS[family]})
    X = families_clean[family].to_numpy(dtype=float)
    
    if X.shape[1] == 0:
        raise ValueError(f"No valid features found for family {family}")
    
    # Determine model path
    if use_final_model:
        model_filename = f"{family}_{task_type}_final.joblib"
    else:
        model_filename = f"{family}_{task_type}_fold1.joblib"
    
    model_path = os.path.join(model_dir, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Make predictions
    predictions = predict_with_saved_model(model_path, X)
    
    # Create results dataframe
    results = pd.DataFrame({
        'molecule_id': df['Name'] if 'Name' in df.columns else range(len(df)),
        'family': family,
        'task_type': task_type
    })
    
    if task_type == 'binary':
        results['predicted_class'] = predictions['predictions']  # 0=N, 1=FN
        results['probability_FN'] = predictions['probabilities']
        results['probability_N'] = 1 - predictions['probabilities']
        results['predicted_phase'] = ['FN' if p == 1 else 'N' for p in predictions['predictions']]
    
    elif task_type == 'multiclass':
        results['predicted_transition_type'] = predictions['predictions']
        # Add probability columns for each class
        prob_df = pd.DataFrame(
            predictions['probabilities'], 
            columns=[f'probability_type_{i}' for i in range(predictions['probabilities'].shape[1])]
        )
        results = pd.concat([results, prob_df], axis=1)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Make predictions using saved classifier_RF models')
    parser.add_argument('data_path', help='Path to CSV/Parquet file with data to predict')
    parser.add_argument('--model_dir', default='../data/ml/classifier_RF/models', 
                       help='Directory containing saved models')
    parser.add_argument('--family', choices=list(FAMILY_TO_COLS.keys()), required=True,
                       help='Descriptor family to use')
    parser.add_argument('--task_type', choices=['binary', 'multiclass'], required=True,
                       help='Type of prediction task')
    parser.add_argument('--output', help='Output CSV file (optional)')
    parser.add_argument('--use_fold_model', action='store_true',
                       help='Use fold 1 model instead of final model')
    
    args = parser.parse_args()
    
    # Make predictions
    try:
        results = load_and_predict(
            data_path=args.data_path,
            model_dir=args.model_dir,
            family=args.family,
            task_type=args.task_type,
            use_final_model=not args.use_fold_model
        )
        
        # Save or display results
        if args.output:
            results.to_csv(args.output, index=False)
            print(f"Predictions saved to: {args.output}")
        else:
            print("\nPredictions:")
            print(results.to_string(index=False))
            
    except Exception as e:
        print(f"Error making predictions: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
