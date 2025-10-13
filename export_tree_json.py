"""
Export Decision Tree to JSON Format
Converts the trained decision tree into a structured JSON representation
"""
import argparse
import json
import numpy as np
from model.paths import get_viper_path
from model.tree_wrapper import TreeWrapper


def position_name(pos):
    """Get position name"""
    names = [
        "TopLeft", "TopCenter", "TopRight",
        "MidLeft", "Center", "MidRight",
        "BotLeft", "BotCenter", "BotRight"
    ]
    return names[pos]


def interpret_threshold(threshold):
    """Interpret threshold value"""
    if threshold < -0.75:
        return {"operator": "==", "value": "opponent"}
    elif -0.75 <= threshold < -0.25:
        return {"operator": "==", "value": "opponent"}
    elif -0.25 <= threshold < 0.25:
        return {"operator": "==", "value": "empty"}
    elif 0.25 <= threshold < 0.75:
        return {"operator": "!=", "value": "player"}
    else:
        return {"operator": "<=", "value": float(threshold)}


def tree_to_dict(tree, node_id=0):
    """Recursively convert tree to dictionary structure"""
    feature = tree.tree_.feature[node_id]
    threshold = tree.tree_.threshold[node_id]
    n_samples = int(tree.tree_.n_node_samples[node_id])
    value = tree.tree_.value[node_id][0].tolist()

    # Check if leaf
    is_leaf = tree.tree_.children_left[node_id] == -1

    if is_leaf:
        # Leaf node
        action = int(np.argmax(value))
        confidence = float(value[action] / sum(value))

        return {
            "type": "leaf",
            "node_id": int(node_id),
            "action": action,
            "action_name": position_name(action),
            "samples": n_samples,
            "confidence": confidence,
            "value_distribution": value
        }
    else:
        # Internal node
        left_child = tree.tree_.children_left[node_id]
        right_child = tree.tree_.children_right[node_id]

        condition = interpret_threshold(threshold)

        return {
            "type": "split",
            "node_id": int(node_id),
            "feature": int(feature),
            "feature_name": position_name(feature),
            "threshold": float(threshold),
            "condition": condition,
            "samples": n_samples,
            "left": tree_to_dict(tree, left_child),
            "right": tree_to_dict(tree, right_child)
        }


def extract_rules_as_json(tree, min_samples=10):
    """Extract all decision rules as JSON array"""
    rules = []

    def get_path_to_leaf(node_id, path_conditions, path_features):
        feature = tree.tree_.feature[node_id]
        threshold = tree.tree_.threshold[node_id]

        # Check if leaf
        is_leaf = tree.tree_.children_left[node_id] == -1

        if is_leaf:
            value = tree.tree_.value[node_id][0]
            action = int(np.argmax(value))
            n_samples = int(tree.tree_.n_node_samples[node_id])
            confidence = float(value[action] / value.sum())

            if n_samples >= min_samples:
                rule = {
                    "rule_id": len(rules),
                    "conditions": path_conditions.copy(),
                    "features_checked": path_features.copy(),
                    "action": action,
                    "action_name": position_name(action),
                    "samples": n_samples,
                    "confidence": confidence
                }
                rules.append(rule)
        else:
            left_child = tree.tree_.children_left[node_id]
            right_child = tree.tree_.children_right[node_id]

            condition_info = interpret_threshold(threshold)

            # Left branch
            left_condition = {
                "feature": int(feature),
                "feature_name": position_name(feature),
                "operator": condition_info["operator"],
                "value": condition_info["value"],
                "branch": "left"
            }
            path_conditions.append(left_condition)
            path_features.append(int(feature))
            get_path_to_leaf(left_child, path_conditions, path_features)
            path_conditions.pop()
            path_features.pop()

            # Right branch (negated condition)
            right_condition = {
                "feature": int(feature),
                "feature_name": position_name(feature),
                "operator": "!=" if condition_info["operator"] == "==" else ">",
                "value": condition_info["value"],
                "branch": "right"
            }
            path_conditions.append(right_condition)
            path_features.append(int(feature))
            get_path_to_leaf(right_child, path_conditions, path_features)
            path_conditions.pop()
            path_features.pop()

    get_path_to_leaf(0, [], [])

    # Sort by samples
    rules.sort(key=lambda x: x['samples'], reverse=True)

    # Update rule IDs after sorting
    for i, rule in enumerate(rules):
        rule['rule_id'] = i

    return rules


def get_tree_metadata(tree):
    """Extract tree metadata and statistics"""
    feature_importance = tree.feature_importances_

    # Feature usage count
    feature_count = {}
    for node_id in range(tree.tree_.node_count):
        if tree.tree_.children_left[node_id] != -1:
            feature = int(tree.tree_.feature[node_id])
            feature_count[feature] = feature_count.get(feature, 0) + 1

    # Action distribution
    action_counts = {}
    for node_id in range(tree.tree_.node_count):
        if tree.tree_.children_left[node_id] == -1:
            value = tree.tree_.value[node_id][0]
            action = int(np.argmax(value))
            samples = int(tree.tree_.n_node_samples[node_id])
            action_counts[action] = action_counts.get(action, 0) + samples

    return {
        "total_nodes": int(tree.tree_.node_count),
        "leaf_nodes": int(tree.tree.get_n_leaves()),
        "internal_nodes": int(tree.tree_.node_count - tree.tree.get_n_leaves()),
        "max_depth": int(tree.tree.get_depth()),
        "total_samples": int(tree.tree_.n_node_samples[0]),
        "feature_importance": {
            int(i): {"name": position_name(i), "importance": float(imp)}
            for i, imp in enumerate(feature_importance)
        },
        "feature_usage": {
            int(feat): {"name": position_name(feat), "count": count}
            for feat, count in feature_count.items()
        },
        "action_distribution": {
            int(action): {"name": position_name(action), "samples": count}
            for action, count in action_counts.items()
        }
    }


def main(args):
    """Main export function"""
    print("=" * 80)
    print("Exporting Decision Tree to JSON")
    print("=" * 80)

    # Load model
    print("\nLoading decision tree...")
    model = TreeWrapper.load(get_viper_path(args))
    tree = model.tree

    # Export formats
    exports = {}

    # 1. Full tree structure
    print("\nConverting tree structure to JSON...")
    exports["tree_structure"] = tree_to_dict(tree)

    # 2. Rules
    print("Extracting decision rules...")
    exports["rules"] = extract_rules_as_json(tree, min_samples=args.min_samples)
    print(f"  Found {len(exports['rules'])} rules (min {args.min_samples} samples)")

    # 3. Metadata
    print("Gathering tree metadata...")
    exports["metadata"] = get_tree_metadata(tree)

    # 4. Add header info
    exports["info"] = {
        "description": "TicTacToe Decision Tree exported from VIPER",
        "environment": args.env_name,
        "max_depth": args.max_depth,
        "max_leaves": args.max_leaves,
        "min_samples_for_rules": args.min_samples
    }

    # Save to file
    output_file = "decision_tree.json"
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(exports, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved: {output_file}")

    # Also save rules separately for easier access
    rules_file = "tree_rules.json"
    print(f"\nSaving rules to {rules_file}...")
    with open(rules_file, 'w', encoding='utf-8') as f:
        json.dump({
            "info": exports["info"],
            "rules": exports["rules"],
            "metadata": exports["metadata"]
        }, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved: {rules_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("EXPORT SUMMARY")
    print("=" * 80)
    print(f"  Tree Structure: {exports['metadata']['total_nodes']} nodes, "
          f"{exports['metadata']['max_depth']} depth")
    print(f"  Rules Extracted: {len(exports['rules'])}")
    print(f"  Output Files:")
    print(f"    - {output_file} (complete tree + rules + metadata)")
    print(f"    - {rules_file} (rules + metadata only)")
    print("\n" + "=" * 80)
    print("Export Complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export decision tree to JSON")
    parser.add_argument("--env-name", type=str, default="TicTacToe-v0")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--max-leaves", type=int, default=None)
    parser.add_argument("--log-prefix", type=str, default="")
    parser.add_argument("--min-samples", type=int, default=10,
                       help="Minimum samples for a rule to be included")

    args = parser.parse_args()
    main(args)
