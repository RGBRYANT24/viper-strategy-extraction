"""
Export Decision Tree as Clean Text Format
No graphics, no overlap, just clear rules
"""
import argparse
import numpy as np
from model.paths import get_viper_path
from model.tree_wrapper import TreeWrapper


def position_name(pos):
    """Get position name"""
    names = [
        "TopLeft(0)", "TopCenter(1)", "TopRight(2)",
        "MidLeft(3)", "Center(4)", "MidRight(5)",
        "BotLeft(6)", "BotCenter(7)", "BotRight(8)"
    ]
    return names[pos]


def interpret_condition(feature, threshold):
    """Interpret a tree condition in human language"""
    pos = feature
    pos_name = position_name(pos)

    if threshold < -0.75:
        return f"{pos_name} == Opponent(O)", f"{pos_name} != Opponent(O)"
    elif -0.75 <= threshold < -0.25:
        return f"{pos_name} == Opponent(O)", f"{pos_name} != Opponent(O)"
    elif -0.25 <= threshold < 0.25:
        return f"{pos_name} == Empty", f"{pos_name} != Empty"
    elif 0.25 <= threshold < 0.75:
        return f"{pos_name} != Yours(X)", f"{pos_name} == Yours(X)"
    else:
        return f"{pos_name} <= {threshold:.2f}", f"{pos_name} > {threshold:.2f}"


def export_tree_as_text(tree, max_depth=None):
    """Export tree as readable text with indentation"""
    lines = []

    def recurse(node_id, depth, prefix, is_left, parent_condition):
        if max_depth is not None and depth > max_depth:
            lines.append(f"{prefix}... (tree continues deeper)")
            return

        feature = tree.tree_.feature[node_id]
        threshold = tree.tree_.threshold[node_id]
        n_samples = tree.tree_.n_node_samples[node_id]
        value = tree.tree_.value[node_id][0]

        # Check if leaf
        is_leaf = tree.tree_.children_left[node_id] == -1

        if is_leaf:
            # Leaf node - show final action
            action = np.argmax(value)
            confidence = value[action] / value.sum()
            lines.append(f"{prefix}→ ACTION: Position {action} ({position_name(action)}) "
                        f"[samples={n_samples}, confidence={confidence:.1%}]")
        else:
            # Internal node - show condition
            left_child = tree.tree_.children_left[node_id]
            right_child = tree.tree_.children_right[node_id]

            left_samples = tree.tree_.n_node_samples[left_child]
            right_samples = tree.tree_.n_node_samples[right_child]

            left_cond, right_cond = interpret_condition(feature, threshold)

            # Show node info
            lines.append(f"{prefix}[Node {node_id}] Check {position_name(feature)} (samples={n_samples})")

            # Left branch
            lines.append(f"{prefix}├─ IF {left_cond}:")
            new_prefix = prefix + "│  "
            if depth < (max_depth or float('inf')) - 1:
                recurse(left_child, depth + 1, new_prefix, True, left_cond)

            # Right branch
            lines.append(f"{prefix}└─ ELSE {right_cond}:")
            new_prefix = prefix + "   "
            if depth < (max_depth or float('inf')) - 1:
                recurse(right_child, depth + 1, new_prefix, False, right_cond)

    lines.append("=" * 80)
    lines.append("DECISION TREE STRUCTURE")
    lines.append("=" * 80)
    lines.append("")
    recurse(0, 0, "", True, "ROOT")

    return "\n".join(lines)


def export_tree_as_rules(tree, min_samples=10):
    """Export tree as IF-THEN rules"""
    rules = []

    def get_path_to_leaf(node_id, path_conditions, path_directions):
        feature = tree.tree_.feature[node_id]
        threshold = tree.tree_.threshold[node_id]

        # Check if leaf
        is_leaf = tree.tree_.children_left[node_id] == -1

        if is_leaf:
            # Generate rule
            value = tree.tree_.value[node_id][0]
            action = np.argmax(value)
            n_samples = tree.tree_.n_node_samples[node_id]
            confidence = value[action] / value.sum()

            if n_samples >= min_samples:  # Only include significant rules
                rule = {
                    'conditions': path_conditions.copy(),
                    'action': action,
                    'samples': n_samples,
                    'confidence': confidence
                }
                rules.append(rule)
        else:
            left_child = tree.tree_.children_left[node_id]
            right_child = tree.tree_.children_right[node_id]

            left_cond, right_cond = interpret_condition(feature, threshold)

            # Left path
            path_conditions.append(left_cond)
            get_path_to_leaf(left_child, path_conditions, path_directions + ['L'])
            path_conditions.pop()

            # Right path
            path_conditions.append(right_cond)
            get_path_to_leaf(right_child, path_conditions, path_directions + ['R'])
            path_conditions.pop()

    get_path_to_leaf(0, [], [])

    # Sort by samples (most common paths first)
    rules.sort(key=lambda x: x['samples'], reverse=True)

    return rules


def format_rules_as_text(rules, max_rules=50):
    """Format rules as readable text"""
    lines = []
    lines.append("=" * 80)
    lines.append("DECISION RULES (IF-THEN FORMAT)")
    lines.append("=" * 80)
    lines.append(f"\nShowing top {min(max_rules, len(rules))} rules (sorted by frequency)\n")

    for i, rule in enumerate(rules[:max_rules], 1):
        lines.append(f"\n{'─' * 80}")
        lines.append(f"RULE #{i} (used in {rule['samples']} games, {rule['confidence']:.1%} confidence)")
        lines.append(f"{'─' * 80}")
        lines.append("IF:")
        for cond in rule['conditions']:
            lines.append(f"  • {cond}")
        lines.append(f"THEN:")
        lines.append(f"  → Place at Position {rule['action']} ({position_name(rule['action'])})")

    return "\n".join(lines)


def export_tree_statistics(tree):
    """Export detailed tree statistics"""
    lines = []
    lines.append("=" * 80)
    lines.append("TREE STATISTICS")
    lines.append("=" * 80)

    # Basic stats
    lines.append(f"\nBasic Information:")
    lines.append(f"  Total Nodes: {tree.tree_.node_count}")
    lines.append(f"  Leaf Nodes: {tree.tree.get_n_leaves()}")
    lines.append(f"  Internal Nodes: {tree.tree_.node_count - tree.tree.get_n_leaves()}")
    lines.append(f"  Tree Depth: {tree.tree.get_depth()}")
    lines.append(f"  Total Training Samples: {tree.tree_.n_node_samples[0]}")

    # Feature usage
    feature_count = {}
    for node_id in range(tree.tree_.node_count):
        if tree.tree_.children_left[node_id] != -1:  # Not a leaf
            feature = tree.tree_.feature[node_id]
            feature_count[feature] = feature_count.get(feature, 0) + 1

    lines.append(f"\nFeature Usage (how many times each position is checked):")
    for pos in sorted(feature_count.keys(), key=lambda x: feature_count[x], reverse=True):
        lines.append(f"  {position_name(pos):20} used {feature_count[pos]:3} times in tree")

    # Leaf distribution
    leaf_actions = []
    for node_id in range(tree.tree_.node_count):
        if tree.tree_.children_left[node_id] == -1:  # Is a leaf
            value = tree.tree_.value[node_id][0]
            action = np.argmax(value)
            samples = tree.tree_.n_node_samples[node_id]
            leaf_actions.append((action, samples))

    action_samples = {}
    for action, samples in leaf_actions:
        action_samples[action] = action_samples.get(action, 0) + samples

    total_samples = sum(action_samples.values())
    lines.append(f"\nAction Distribution (across all paths):")
    for action in sorted(action_samples.keys(), key=lambda x: action_samples[x], reverse=True):
        pct = action_samples[action] / total_samples * 100
        bar = "█" * int(pct / 2)
        lines.append(f"  Position {action} ({position_name(action):20}): {action_samples[action]:6} samples ({pct:5.1f}%) {bar}")

    return "\n".join(lines)


def main(args):
    """Main export function"""
    print("Loading decision tree...")
    model = TreeWrapper.load(get_viper_path(args))
    tree = model.tree

    # Choose export format
    print("\nSelect export format:")
    print("1. Tree structure (indented text)")
    print("2. IF-THEN rules (top 50)")
    print("3. Statistics only")
    print("4. Everything (all formats)")

    # For now, export everything to files

    # 1. Export full tree structure (limited depth)
    print("\n" + "=" * 80)
    print("Exporting tree structure...")
    with open("tree_structure_depth5.txt", "w") as f:
        f.write(export_tree_as_text(tree, max_depth=5))
    print("✓ Saved: tree_structure_depth5.txt (first 5 levels)")

    with open("tree_structure_full.txt", "w") as f:
        f.write(export_tree_as_text(tree, max_depth=None))
    print("✓ Saved: tree_structure_full.txt (complete tree, may be very long)")

    # 2. Export as rules
    print("\nExtracting IF-THEN rules...")
    rules = export_tree_as_rules(tree, min_samples=10)
    with open("tree_rules_top50.txt", "w") as f:
        f.write(format_rules_as_text(rules, max_rules=50))
    print(f"✓ Saved: tree_rules_top50.txt ({len(rules)} total rules, showing top 50)")

    with open("tree_rules_all.txt", "w") as f:
        f.write(format_rules_as_text(rules, max_rules=len(rules)))
    print(f"✓ Saved: tree_rules_all.txt (all {len(rules)} rules)")

    # 3. Export statistics
    print("\nGenerating statistics...")
    with open("tree_statistics.txt", "w") as f:
        f.write(export_tree_statistics(tree))
    print("✓ Saved: tree_statistics.txt")

    # Also print statistics to console
    print("\n" + export_tree_statistics(tree))

    # Print sample of rules
    print("\n" + format_rules_as_text(rules, max_rules=5))

    print("\n" + "=" * 80)
    print("Export Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. tree_structure_depth5.txt  - Tree structure (first 5 levels, readable)")
    print("  2. tree_structure_full.txt    - Complete tree structure (all levels)")
    print("  3. tree_rules_top50.txt       - Top 50 most common decision rules")
    print("  4. tree_rules_all.txt         - All decision rules")
    print("  5. tree_statistics.txt        - Detailed statistics")
    print("\nRecommendation: Start by reading tree_rules_top50.txt!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export decision tree as text")
    parser.add_argument("--env-name", type=str, default="TicTacToe-v0")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--max-leaves", type=int, default=None)
    parser.add_argument("--log-prefix", type=str, default="")

    args = parser.parse_args()
    main(args)
