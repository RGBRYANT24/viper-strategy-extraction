import joblib
import json
import argparse
import numpy as np
from sklearn.tree import _tree

def export_tree_to_json(model, feature_names=None):
    tree_ = model.tree_
    
    # If feature names are not provided, try to get them from the model or generate generic ones
    if feature_names is None:
        if hasattr(model, "feature_names_in_"):
            feature_names = model.feature_names_in_
        else:
            feature_names = [f"Feature {i}" for i in range(tree_.n_features)]

    def recurse(node_id, depth):
        is_leaf = tree_.children_left[node_id] == _tree.TREE_LEAF
        
        node_data = {
            "id": int(node_id),
            "samples": int(tree_.n_node_samples[node_id]),
            "impurity": float(tree_.impurity[node_id]),
            "type": "leaf" if is_leaf else "node"
        }

        if is_leaf:
            # For classification, value is class counts. We take the argmax as the action/class.
            # For regression, value is the predicted value.
            value = tree_.value[node_id]
            if model.n_outputs_ == 1:
                value = value[0, :]
            
            # Assuming classification for "action" and "score" terminology in view.html
            # If it's a classifier
            if hasattr(model, "classes_"):
                class_idx = np.argmax(value)
                node_data["action"] = str(model.classes_[class_idx])
                # Score could be the probability or the raw count
                node_data["score"] = float(value[class_idx] / value.sum())
            else:
                # Regression
                node_data["action"] = f"Value: {value[0]:.2f}"
                node_data["score"] = float(value[0])
                
        else:
            feature_idx = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]
            feature_name = feature_names[feature_idx]
            
            node_data["label"] = f"{feature_name}"
            node_data["condition"] = f"<= {threshold:.2f}"
            
            left_child = recurse(tree_.children_left[node_id], depth + 1)
            right_child = recurse(tree_.children_right[node_id], depth + 1)
            node_data["children"] = [left_child, right_child]

        return node_data

    tree_json = recurse(0, 0)
    
    return {
        "metadata": {
            "n_nodes": int(tree_.node_count),
            "max_depth": int(tree_.max_depth)
        },
        "tree": tree_json
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export sklearn Decision Tree to JSON for visualization.")
    parser.add_argument("input_file", help="Path to the .joblib file containing the model.")
    parser.add_argument("output_file", help="Path to save the output JSON file.")
    
    args = parser.parse_args()
    
    try:
        print(f"Loading model from {args.input_file}...")
        model = joblib.load(args.input_file)
        
        if not hasattr(model, "tree_"):
            raise ValueError("The loaded model is not a Decision Tree (missing `tree_` attribute).")
            
        print("Exporting to JSON...")
        json_data = export_tree_to_json(model)
        
        with open(args.output_file, "w") as f:
            json.dump(json_data, f, indent=2)
            
        print(f"Successfully saved JSON to {args.output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
