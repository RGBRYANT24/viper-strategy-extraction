import numpy as np
import joblib
import os
import argparse
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabulate import tabulate

def evaluate_models(dataset_path, model_paths):
    print(f"Loading dataset from {dataset_path}...")
    dataset = joblib.load(dataset_path)
    X = dataset['states']
    y_true_value = dataset['values']
    
    # Flatten states for sklearn trees
    X_flat = X.reshape(X.shape[0], -1)
    
    results = []
    
    print(f"\nEvaluating {len(model_paths)} models on {len(X)} samples...\n")
    
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Warning: Model path {path} does not exist. Skipping.")
            continue
            
        try:
            model = joblib.load(path)
            
            # Predict
            y_pred = model.predict(X_flat)
            
            # Metrics
            mse = mean_squared_error(y_true_value, y_pred)
            mae = mean_absolute_error(y_true_value, y_pred)
            r2 = r2_score(y_true_value, y_pred)
            
            # Tree Stats
            depth = model.tree_.max_depth
            leaves = model.tree_.n_leaves
            
            # Model Name (filename)
            name = os.path.basename(path)
            
            results.append({
                "Model": name,
                "MSE": mse,
                "MAE": mae,
                "R2": r2,
                "Depth": depth,
                "Leaves": leaves
            })
            
        except Exception as e:
            print(f"Error loading/evaluating {path}: {e}")

    # Sort by MSE (best first)
    results.sort(key=lambda x: x["MSE"])
    
    # Print Table
    print(tabulate(results, headers="keys", tablefmt="grid", floatfmt=".5f"))
    
    # Optional: Save to file
    # with open("benchmark_results.txt", "w") as f:
    #     f.write(tabulate(results, headers="keys", tablefmt="grid", floatfmt=".5f"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Value Trees against Oracle Dataset")
    parser.add_argument('--dataset', type=str, default='data/tictactoe_oracle_dataset.joblib',
                        help='Path to the generated test set')
    parser.add_argument('models', nargs='+', help='Paths to joblib decision tree models')
    
    args = parser.parse_args()
    
    evaluate_models(args.dataset, args.models)
