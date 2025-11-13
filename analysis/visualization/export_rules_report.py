#!/usr/bin/env python3
"""
Export Decision Tree Rules as English Report
Generate a comprehensive document showing all rules with board states
Sorted by priority for presentation
"""

import json
import argparse
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime


def position_name(pos):
    """Position name in English"""
    names = [
        "Top-Left", "Top-Center", "Top-Right",
        "Mid-Left", "Center", "Mid-Right",
        "Bottom-Left", "Bottom-Center", "Bottom-Right"
    ]
    return names[pos]


def parse_board_from_antecedents(antecedents: List[List]) -> Tuple[np.ndarray, Dict[int, str]]:
    """Parse board state from rule antecedents"""
    board = np.full(9, 2, dtype=np.float32)
    position_constraints = {i: [] for i in range(9)}

    # Collect constraints for each position
    for condition in antecedents:
        pos, op, threshold = condition
        position_constraints[pos].append((op, threshold))

    # Analyze constraints
    for pos in range(9):
        constraints = position_constraints[pos]

        if not constraints:
            position_constraints[pos] = "?"
            continue

        # Check if opponent's piece: X[pos] <= -0.5
        if any(op == "<=" and th == -0.5 for op, th in constraints):
            board[pos] = -1
            position_constraints[pos] = "O"
            continue

        # Check if player's piece: X[pos] > 0.5
        if any(op == ">" and th == 0.5 for op, th in constraints):
            board[pos] = 1
            position_constraints[pos] = "X"
            continue

        # Analyze constraint combinations
        has_gt_minus_half = any(op == ">" and th == -0.5 for op, th in constraints)
        has_le_half = any(op == "<=" and th == 0.5 for op, th in constraints)

        # X[pos] > -0.5 AND X[pos] <= 0.5 means empty
        if has_gt_minus_half and has_le_half:
            board[pos] = 0
            position_constraints[pos] = "."
            continue

        # X[pos] > -0.5 means not opponent (empty or player)
        if has_gt_minus_half and not has_le_half:
            board[pos] = 2
            position_constraints[pos] = "?/X"
            continue

        # X[pos] <= 0.5 means not player (empty or opponent)
        if has_le_half and not has_gt_minus_half:
            board[pos] = 2
            position_constraints[pos] = "?/O"
            continue

        position_constraints[pos] = "?"

    return board, position_constraints


def render_board_simple(position_constraints: Dict[int, str]) -> str:
    """Render board in simple ASCII format"""
    lines = []
    lines.append("     Board State:")
    lines.append("     ┌─────┬─────┬─────┐")
    for row in range(3):
        cells = []
        for col in range(3):
            idx = row * 3 + col
            constraint = position_constraints.get(idx, "?")
            cells.append(f" {constraint:^3} ")
        lines.append("     │" + "│".join(cells) + "│")
        if row < 2:
            lines.append("     ├─────┼─────┼─────┤")
    lines.append("     └─────┴─────┴─────┘")
    return "\n".join(lines)


def softmax(logits: List[float]) -> np.ndarray:
    """Calculate softmax probability distribution"""
    logits = np.array(logits)
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


def format_action_distribution(output_vector: List[float], board: np.ndarray, best_action: int) -> str:
    """Format action distribution table"""
    actions_with_scores = [(action, score) for action, score in enumerate(output_vector)]
    actions_with_scores.sort(key=lambda x: x[1], reverse=True)
    probs = softmax(output_vector)

    lines = []
    lines.append("     Action Priority (sorted by score):")
    lines.append("     " + "-" * 70)
    lines.append(f"     {'Rank':<6} {'Pos':<5} {'Position':<14} {'Score':<10} {'Prob':<8} {'Status':<10} {'Mark'}")
    lines.append("     " + "-" * 70)

    for rank, (action, score) in enumerate(actions_with_scores, 1):
        prob = probs[action]

        # Determine position status
        if board[action] == 0:
            status = "Empty"
        elif board[action] == 1:
            status = "Player(X)"
        elif board[action] == -1:
            status = "Opp(O)"
        else:
            status = "Unknown"

        marker = "✓ BEST" if action == best_action else ""
        lines.append(f"     {rank:<6} {action:<5} {position_name(action):<14} {score:<10.3f} {prob:<8.2%} {status:<10} {marker}")

    return "\n".join(lines)


def format_rule(rule: Dict, rule_idx: int) -> str:
    """Format a single rule as text"""
    lines = []

    # Rule header
    lines.append("\n" + "=" * 80)
    lines.append(f"RULE #{rule_idx + 1}")
    lines.append("=" * 80)

    # Basic info
    lines.append(f"  Priority Score:  {rule['priority']:.4f}")
    lines.append(f"  Support Count:   {rule['support_count']} samples")
    lines.append(f"  Best Action:     {rule['best_action']} ({position_name(rule['best_action'])})")
    lines.append("")

    # Parse board state
    board, position_constraints = parse_board_from_antecedents(rule['antecedents'])

    # Board visualization
    lines.append(render_board_simple(position_constraints))
    lines.append("")
    lines.append("     Legend: X=Player, O=Opponent, .=Empty, ?=Unconstrained")
    lines.append("             ?/X=Not opponent, ?/O=Not player")
    lines.append("")

    # Conditions in readable format
    lines.append("  Rule Conditions:")
    for i, condition in enumerate(rule['antecedents'], 1):
        pos, op, threshold = condition
        if op == "<=" and threshold == -0.5:
            desc = f"Pos {pos} ({position_name(pos)}) is Opponent (O)"
        elif op == ">" and threshold == 0.5:
            desc = f"Pos {pos} ({position_name(pos)}) is Player (X)"
        elif op == "<=" and threshold == 0.5:
            desc = f"Pos {pos} ({position_name(pos)}) is not Player"
        elif op == ">" and threshold == -0.5:
            desc = f"Pos {pos} ({position_name(pos)}) is not Opponent"
        else:
            desc = f"Pos {pos} ({position_name(pos)}) {op} {threshold:.2f}"
        lines.append(f"    {i}. {desc}")
    lines.append("")

    # Action distribution
    lines.append(format_action_distribution(rule['output_vector'], board, rule['best_action']))
    lines.append("")

    # Sanity check
    issues = []
    if board[rule['best_action']] == -1:
        issues.append(f"⚠ Best action position occupied by opponent!")
    elif board[rule['best_action']] == 1:
        issues.append(f"⚠ Best action position occupied by player!")

    if issues:
        lines.append("  ⚠ Potential Issues:")
        for issue in issues:
            lines.append(f"    - {issue}")
    else:
        lines.append("  ✓ Rule appears valid")

    return "\n".join(lines)


def generate_summary_statistics(rules: List[Dict]) -> str:
    """Generate summary statistics"""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY STATISTICS")
    lines.append("=" * 80)
    lines.append("")

    # Basic stats
    lines.append(f"Total Rules:           {len(rules)}")
    lines.append(f"Total Support Samples: {sum(r['support_count'] for r in rules)}")
    lines.append("")

    # Priority distribution
    priorities = [r['priority'] for r in rules]
    lines.append(f"Priority Range:        [{min(priorities):.2f}, {max(priorities):.2f}]")
    lines.append(f"Priority Mean:         {np.mean(priorities):.2f} ± {np.std(priorities):.2f}")
    lines.append("")

    # Most common actions
    action_counts = {}
    for rule in rules:
        action = rule['best_action']
        support = rule['support_count']
        action_counts[action] = action_counts.get(action, 0) + support

    lines.append("Action Distribution (by support count):")
    for action in sorted(action_counts.keys(), key=lambda a: action_counts[a], reverse=True):
        count = action_counts[action]
        pct = count / sum(action_counts.values()) * 100
        lines.append(f"  Position {action} ({position_name(action):<14}): {count:>6} samples ({pct:>5.1f}%)")
    lines.append("")

    # Average conditions per rule
    avg_conditions = np.mean([len(r['antecedents']) for r in rules])
    lines.append(f"Average Conditions per Rule: {avg_conditions:.1f}")

    return "\n".join(lines)


def generate_header() -> str:
    """Generate report header"""
    lines = []
    lines.append("=" * 80)
    lines.append(" " * 20 + "DECISION TREE RULES REPORT")
    lines.append(" " * 25 + "TicTacToe VIPER Policy")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("This report shows all decision tree rules learned by the VIPER algorithm.")
    lines.append("Rules are sorted by priority score (higher = more important).")
    lines.append("")
    lines.append("For each rule, we show:")
    lines.append("  - The board state (derived from rule conditions)")
    lines.append("  - The action priorities and probabilities")
    lines.append("  - The best action recommended by this rule")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Export rules as English report")
    parser.add_argument("json_file", type=str, nargs='?',
                       default="log/viper_mask_ppo_tictactoe/viper_mask_ppo_tree_20251107_001324_iter10_samples50000_depth10_leaves50.rules.json",
                       help="Rules JSON file path")
    parser.add_argument("--output", "-o", type=str, default="decision_tree_rules_report.txt",
                       help="Output file name")
    parser.add_argument("--max-rules", type=int, default=None,
                       help="Maximum number of rules to include (default: all)")

    args = parser.parse_args()

    # Load rules
    print(f"Loading rules from: {args.json_file}")
    with open(args.json_file, 'r') as f:
        data = json.load(f)

    rules = data['rules']
    print(f"Loaded {len(rules)} rules")

    # Limit number of rules if specified
    if args.max_rules:
        rules = rules[:args.max_rules]
        print(f"Limiting to top {args.max_rules} rules")

    # Generate report
    print(f"Generating report...")

    report_lines = []

    # Header
    report_lines.append(generate_header())

    # Summary statistics
    report_lines.append(generate_summary_statistics(rules))
    report_lines.append("\n\n")

    # All rules (already sorted by priority in JSON)
    report_lines.append("=" * 80)
    report_lines.append("DETAILED RULES")
    report_lines.append("=" * 80)

    for i, rule in enumerate(rules):
        report_lines.append(format_rule(rule, i))

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(rules)} rules...")

    # Footer
    report_lines.append("\n\n")
    report_lines.append("=" * 80)
    report_lines.append(" " * 30 + "END OF REPORT")
    report_lines.append("=" * 80)

    # Write to file
    report_text = "\n".join(report_lines)

    print(f"\nSaving report to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"✓ Report saved successfully!")
    print(f"\nReport Statistics:")
    print(f"  Total rules:     {len(rules)}")
    print(f"  Output file:     {args.output}")
    print(f"  File size:       {len(report_text):,} characters")
    print(f"  Lines:           {len(report_lines):,}")
    print()
    print("You can now share this report with your advisor!")


if __name__ == "__main__":
    main()
