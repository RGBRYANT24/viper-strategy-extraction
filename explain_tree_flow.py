"""
Explain how decision tree actually works - it's a TREE, not a rule list!
"""
import numpy as np
from model.paths import get_viper_path
from model.tree_wrapper import TreeWrapper
import argparse


def trace_decision_path(tree, state, verbose=True):
    """
    Trace the path through the decision tree for a given state
    Shows EXACTLY which nodes are checked and in what order
    """
    node_id = 0  # Start at root
    path = []

    position_names = [
        "TopLeft(0)", "TopCenter(1)", "TopRight(2)",
        "MidLeft(3)", "Center(4)", "MidRight(5)",
        "BotLeft(6)", "BotCenter(7)", "BotRight(8)"
    ]

    def get_state_value(pos):
        val = state[pos]
        if val == 1.0:
            return "X (yours)"
        elif val == -1.0:
            return "O (opponent)"
        else:
            return "Empty"

    if verbose:
        print("\n" + "=" * 80)
        print("TRACING DECISION PATH")
        print("=" * 80)
        print("\nCurrent Board State:")
        symbols = {1.0: 'X', -1.0: 'O', 0.0: '.'}
        board_2d = state.reshape(3, 3)
        print("  +---+---+---+")
        for i in range(3):
            row = "  | " + " | ".join([symbols[board_2d[i][j]] for j in range(3)]) + " |"
            print(row)
            if i < 2:
                print("  +---+---+---+")
        print("  +---+---+---+")
        print()

    step = 1
    while True:
        feature = tree.tree_.feature[node_id]
        threshold = tree.tree_.threshold[node_id]

        # Check if this is a leaf node
        if tree.tree_.children_left[node_id] == -1:  # Leaf node
            value = tree.tree_.value[node_id][0]
            action = np.argmax(value)
            n_samples = tree.tree_.n_node_samples[node_id]

            path.append({
                'type': 'leaf',
                'node_id': node_id,
                'action': action,
                'samples': n_samples
            })

            if verbose:
                print(f"{'─' * 80}")
                print(f"STEP {step}: REACHED LEAF NODE {node_id}")
                print(f"{'─' * 80}")
                print(f"  This is a FINAL DECISION node")
                print(f"  Training samples that reached here: {n_samples}")
                print(f"  DECISION: Place at Position {action} ({position_names[action]})")

            return path, action

        # Internal node - make a decision
        state_value = state[feature]
        go_left = state_value <= threshold

        path.append({
            'type': 'decision',
            'node_id': node_id,
            'feature': feature,
            'threshold': threshold,
            'state_value': state_value,
            'go_left': go_left
        })

        if verbose:
            print(f"{'─' * 80}")
            print(f"STEP {step}: DECISION NODE {node_id}")
            print(f"{'─' * 80}")
            print(f"  Question: Is {position_names[feature]} <= {threshold:.2f}?")
            print(f"  Current state: {position_names[feature]} = {state_value:.2f} ({get_state_value(feature)})")

            # Interpret the threshold
            if abs(threshold - 0.5) < 0.01:
                interpret = "Is it NOT yours? (empty or opponent)"
            elif abs(threshold - (-0.5)) < 0.01:
                interpret = "Is it opponent's?"
            else:
                interpret = f"Value <= {threshold:.2f}"

            print(f"  Interpretation: {interpret}")
            print(f"  Answer: {'YES (go left)' if go_left else 'NO (go right)'}")

        # Move to next node
        if go_left:
            node_id = tree.tree_.children_left[node_id]
        else:
            node_id = tree.tree_.children_right[node_id]

        step += 1


def demonstrate_tree_structure():
    """
    Demonstrate that rules are NOT checked in sequence
    """
    print("\n" + "=" * 80)
    print("UNDERSTANDING: TREE vs RULE LIST")
    print("=" * 80)

    print("""
❌ WRONG CONCEPTION (Linear Rule List):
────────────────────────────────────────
    Start
      ↓
   Check Rule #1 conditions
      ↓
   All satisfied? → YES → Use Rule #1
      ↓ NO
   Check Rule #2 conditions
      ↓
   All satisfied? → YES → Use Rule #2
      ↓ NO
   Check Rule #3 conditions
   ...


✓ CORRECT CONCEPTION (Decision Tree):
────────────────────────────────────────
                    Start (Root Node)
                         ↓
            Check Position 2 (TopRight)
                    /        \\
                Empty?      Occupied?
                  /            \\
            Check Pos 0     Check Pos 3
              /    \\          /    \\
           ...     ...      ...    ...
            ↓       ↓        ↓      ↓
         Rule #1  Rule #7  Rule #4  Rule #15


KEY DIFFERENCES:

1. TREE STRUCTURE:
   • You follow ONE path from root to leaf
   • Each path is a series of YES/NO questions
   • You only check 5-8 positions (not all rules!)

2. RULE NUMBER IS JUST A LABEL:
   • Rule #1, #2, #3 are just leaf node IDs
   • They are NOT checked in order
   • You might reach Rule #50 before Rule #2!

3. PATH DETERMINES RULE:
   • Your current state determines the path
   • The path determines which leaf (rule) you reach
   • Different states reach different leaves
    """)


def compare_different_states(model):
    """
    Show how different states reach different rules via different paths
    """
    print("\n" + "=" * 80)
    print("EXAMPLE: DIFFERENT STATES → DIFFERENT PATHS → DIFFERENT RULES")
    print("=" * 80)

    test_cases = [
        {
            "name": "EMPTY BOARD (Opening)",
            "state": np.zeros(9, dtype=np.float32)
        },
        {
            "name": "OPPONENT TOOK CENTER",
            "state": np.array([0, 0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32)
        },
        {
            "name": "MUST BLOCK (Opponent has 0,1)",
            "state": np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'═' * 80}")
        print(f"SCENARIO {i}: {case['name']}")
        print(f"{'═' * 80}")

        path, action = trace_decision_path(model.tree, case['state'], verbose=True)

        print(f"\nSUMMARY:")
        print(f"  • Total steps in path: {len(path)}")
        print(f"  • Positions checked: {len([p for p in path if p['type'] == 'decision'])}")
        print(f"  • Final decision: Position {action}")


def explain_rule_ordering():
    """
    Explain why rules are ordered by frequency
    """
    print("\n" + "=" * 80)
    print("WHY ARE RULES ORDERED BY FREQUENCY?")
    print("=" * 80)

    print("""
The rule ordering (Rule #1, #2, #3...) is for HUMAN UNDERSTANDING, not for
the tree's operation!

FREQUENCY = How many training samples reached this leaf

Why Rule #1 has 1039 samples (highest):
  1. It handles opening/early game states
  2. Every game passes through opening
  3. Opening states are very similar across games
  4. So many samples end up in the same leaf

Why Rule #59 has few samples:
  1. It handles a rare, specific situation
  2. Not every game reaches that state
  3. That state requires specific sequence of moves

ORDERING PURPOSE:
  • Helps humans understand the tree
  • Rule #1 is "most important" because it's most common
  • But the tree doesn't care about ordering!
  • Tree only follows the path dictated by YES/NO questions

ANALOGY:
  Think of a flowchart:
    - You don't check all boxes in order
    - You follow arrows based on answers
    - Different inputs → different paths → different endpoints

  Rule numbers are like "endpoint names"
  The tree structure determines which endpoint you reach!
    """)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="TicTacToe-v0")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--max-leaves", type=int, default=None)
    parser.add_argument("--log-prefix", type=str, default="")
    args = parser.parse_args()

    print("=" * 80)
    print("DECISION TREE OPERATION EXPLAINED")
    print("=" * 80)

    # Load model
    model = TreeWrapper.load(get_viper_path(args))

    # Step 1: Explain tree vs list
    demonstrate_tree_structure()

    # Step 2: Show actual paths
    compare_different_states(model)

    # Step 3: Explain ordering
    explain_rule_ordering()

    print("\n" + "=" * 80)
    print("EXPLANATION COMPLETE")
    print("=" * 80)

    print("""
KEY TAKEAWAYS:

1. Decision tree is a TREE, not a sequential rule list
2. You follow ONE path from root to leaf (5-8 decisions)
3. Rule numbers are just labels for leaves
4. Frequency shows how common that leaf is in training
5. High frequency = handles common situations
6. Low frequency = handles rare edge cases

The tree doesn't "try Rule #1, then Rule #2, then Rule #3..."
Instead: "Ask question 1, go left/right, ask question 2, go left/right... reach leaf"
    """)


if __name__ == "__main__":
    main()
