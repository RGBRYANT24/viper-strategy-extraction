"""
TicTacToe Strategy Analysis - Extract Meaningful Insights
Instead of visualizing the full tree, analyze what strategy it learned
"""
import argparse
import numpy as np
from collections import defaultdict
from model.paths import get_viper_path
from model.tree_wrapper import TreeWrapper
from gym_env.tictactoe import TicTacToeEnv


def visualize_board(state):
    """Visualize a board state"""
    symbols = {1.0: 'X', -1.0: 'O', 0.0: '.'}
    board_2d = state.reshape(3, 3)

    lines = []
    lines.append("  +---+---+---+")
    for i in range(3):
        row = "  | " + " | ".join([symbols[board_2d[i][j]] for j in range(3)]) + " |"
        lines.append(row)
        if i < 2:
            lines.append("  +---+---+---+")
    lines.append("  +---+---+---+")
    return "\n".join(lines)


def position_name(pos):
    """Get human-readable position name"""
    names = [
        "Top-Left", "Top-Center", "Top-Right",
        "Mid-Left", "Center", "Mid-Right",
        "Bottom-Left", "Bottom-Center", "Bottom-Right"
    ]
    return names[pos]


def test_scenarios(model, env):
    """Test the decision tree on common game scenarios"""
    print("\n" + "=" * 80)
    print("TESTING DECISION TREE ON COMMON SCENARIOS")
    print("=" * 80)

    scenarios = [
        {
            "name": "Opening Move (Empty Board)",
            "state": np.zeros(9, dtype=np.float32),
            "description": "First move of the game"
        },
        {
            "name": "Center Taken by Opponent",
            "state": np.array([0, 0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32),
            "description": "Opponent took center, your turn"
        },
        {
            "name": "Corner Opening",
            "state": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            "description": "You took top-left corner"
        },
        {
            "name": "Block Opponent Win (Horizontal)",
            "state": np.array([0, 0, 0, -1, -1, 0, 0, 0, 0], dtype=np.float32),
            "description": "Opponent has 2 in a row (positions 3-4), must block at 5"
        },
        {
            "name": "Block Opponent Win (Vertical)",
            "state": np.array([0, -1, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32),
            "description": "Opponent has 2 in column (positions 1-4), must block at 7"
        },
        {
            "name": "Block Diagonal Win",
            "state": np.array([-1, 0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32),
            "description": "Opponent has 2 on diagonal (positions 0-4), must block at 8"
        },
        {
            "name": "Win Opportunity (Horizontal)",
            "state": np.array([1, 1, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32),
            "description": "You have 2 in a row (positions 0-1), should take 2 to win"
        },
        {
            "name": "Fork Opportunity",
            "state": np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
            "description": "You control opposite corners (0 and 4)"
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'─' * 80}")
        print(f"Scenario {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(visualize_board(scenario['state']))

        # Get model's decision
        action = model.predict(scenario['state'].reshape(1, -1))[0][0]
        print(f"\n  → Decision: Place at Position {action} ({position_name(action)})")

        # Visualize the resulting board
        result_state = scenario['state'].copy()
        result_state[action] = 1
        print(f"\n  Resulting board:")
        print(visualize_board(result_state))


def analyze_first_moves(model):
    """Analyze what the model chooses as opening moves"""
    print("\n" + "=" * 80)
    print("OPENING MOVE PREFERENCES")
    print("=" * 80)

    # Test empty board multiple times (should be deterministic)
    empty_board = np.zeros(9, dtype=np.float32)
    action = model.predict(empty_board.reshape(1, -1))[0][0]

    print(f"\nOn an empty board, the model ALWAYS chooses:")
    print(f"  Position {action}: {position_name(action)}")
    print(visualize_board(empty_board))

    # Test after opponent takes different positions
    print("\n" + "─" * 80)
    print("Response to opponent's opening moves:")
    print("─" * 80)

    for opp_pos in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        state = np.zeros(9, dtype=np.float32)
        state[opp_pos] = -1
        response = model.predict(state.reshape(1, -1))[0][0]
        print(f"  Opponent takes {position_name(opp_pos):16} → Respond at {position_name(response)}")


def analyze_tree_structure(tree):
    """Analyze the decision tree structure"""
    print("\n" + "=" * 80)
    print("DECISION TREE STRUCTURE ANALYSIS")
    print("=" * 80)

    # Feature importance
    feature_importance = tree.tree.feature_importances_
    print("\nFeature Importance (which positions are checked most):")
    print("─" * 60)

    # Sort by importance
    importance_sorted = sorted(enumerate(feature_importance), key=lambda x: x[1], reverse=True)

    # Categorize positions
    corners = [0, 2, 6, 8]
    edges = [1, 3, 5, 7]
    center = [4]

    corner_importance = sum(feature_importance[i] for i in corners)
    edge_importance = sum(feature_importance[i] for i in edges)
    center_importance = feature_importance[4]

    print(f"\nPosition Category Importance:")
    print(f"  Corners (0,2,6,8):  {corner_importance:.4f} ({corner_importance*100:.1f}%)")
    print(f"  Center (4):         {center_importance:.4f} ({center_importance*100:.1f}%)")
    print(f"  Edges (1,3,5,7):    {edge_importance:.4f} ({edge_importance*100:.1f}%)")

    print(f"\nTop 5 Most Important Positions:")
    for rank, (pos, importance) in enumerate(importance_sorted[:5], 1):
        bar = "█" * int(importance * 50)
        category = "Corner" if pos in corners else ("Center" if pos == 4 else "Edge")
        print(f"  {rank}. Pos {pos} ({position_name(pos):16}) [{category:6}] {importance:.4f} {bar}")

    # Tree statistics
    print(f"\nTree Statistics:")
    print(f"  Depth: {tree.tree.get_depth()}")
    print(f"  Leaves: {tree.tree.get_n_leaves()}")
    print(f"  Total Nodes: {tree.tree.tree_.node_count}")
    print(f"  Average samples per leaf: {tree.tree.tree_.n_node_samples[0] / tree.tree.get_n_leaves():.1f}")


def evaluate_strategic_understanding(model):
    """Evaluate if the model learned key strategic concepts"""
    print("\n" + "=" * 80)
    print("STRATEGIC UNDERSTANDING TEST")
    print("=" * 80)

    tests = {
        "Win When Possible": [],
        "Block Opponent Win": [],
        "Take Center/Corners": [],
        "Avoid Edges": []
    }

    # Test 1: Can it take a winning move?
    print("\n1. Testing if model takes winning moves...")
    win_scenarios = [
        # Horizontal wins
        (np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), 2, "Complete top row"),
        (np.array([0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.float32), 5, "Complete middle row"),
        # Vertical wins
        (np.array([1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32), 6, "Complete left column"),
        # Diagonal wins
        (np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32), 8, "Complete main diagonal"),
    ]

    wins_correct = 0
    for state, correct_action, description in win_scenarios:
        action = model.predict(state.reshape(1, -1))[0][0]
        correct = action == correct_action
        wins_correct += correct
        status = "✓" if correct else "✗"
        print(f"  {status} {description}: Expected pos {correct_action}, Got pos {action}")

    tests["Win When Possible"] = f"{wins_correct}/{len(win_scenarios)}"

    # Test 2: Can it block opponent wins?
    print("\n2. Testing if model blocks opponent wins...")
    block_scenarios = [
        (np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), 2, "Block top row"),
        (np.array([0, 0, 0, -1, -1, 0, 0, 0, 0], dtype=np.float32), 5, "Block middle row"),
        (np.array([-1, 0, 0, -1, 0, 0, 0, 0, 0], dtype=np.float32), 6, "Block left column"),
        (np.array([-1, 0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32), 8, "Block diagonal"),
    ]

    blocks_correct = 0
    for state, correct_action, description in block_scenarios:
        action = model.predict(state.reshape(1, -1))[0][0]
        correct = action == correct_action
        blocks_correct += correct
        status = "✓" if correct else "✗"
        print(f"  {status} {description}: Expected pos {correct_action}, Got pos {action}")

    tests["Block Opponent Win"] = f"{blocks_correct}/{len(block_scenarios)}"

    # Test 3: Preference for strategic positions
    print("\n3. Testing preference for center/corners over edges...")
    empty = np.zeros(9, dtype=np.float32)
    first_move = model.predict(empty.reshape(1, -1))[0][0]

    corners = [0, 2, 6, 8]
    is_strategic = first_move == 4 or first_move in corners
    status = "✓" if is_strategic else "✗"
    print(f"  {status} Opening move is position {first_move} ({position_name(first_move)})")

    tests["Take Center/Corners"] = "Yes" if is_strategic else "No"

    # Summary
    print("\n" + "=" * 80)
    print("STRATEGY SUMMARY")
    print("=" * 80)
    for test_name, result in tests.items():
        print(f"  {test_name:25} {result}")


def main(args):
    """Main analysis function"""
    print("=" * 80)
    print("TicTacToe Decision Tree - Strategy Analysis")
    print("=" * 80)

    # Load model
    model = TreeWrapper.load(get_viper_path(args))
    env = TicTacToeEnv()

    # Run analyses
    analyze_tree_structure(model)
    analyze_first_moves(model)
    test_scenarios(model, env)
    evaluate_strategic_understanding(model)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TicTacToe strategy learned by decision tree")
    parser.add_argument("--env-name", type=str, default="TicTacToe-v0")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--max-leaves", type=int, default=None)
    parser.add_argument("--log-prefix", type=str, default="")

    args = parser.parse_args()
    main(args)
