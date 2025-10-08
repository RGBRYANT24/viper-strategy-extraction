"""
Diagnose why the decision tree learned corner preference instead of center
"""
import numpy as np
from model.paths import get_oracle_path, get_viper_path
from model.tree_wrapper import TreeWrapper
from train.oracle import get_model_cls
from gym_env import make_env
from gym_env.tictactoe import TicTacToeEnv


def test_oracle_opening_preference(args):
    """Test what the ORACLE (neural network) prefers"""
    print("=" * 80)
    print("TESTING ORACLE (NEURAL NETWORK) OPENING PREFERENCE")
    print("=" * 80)

    # Load oracle
    env = make_env(args)
    model_cls, _ = get_model_cls(args)

    try:
        oracle = model_cls.load(get_oracle_path(args), env=env)
        print("✓ Oracle loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load oracle: {e}")
        print("\nAssuming oracle was not saved. Testing with decision tree only.")
        return None

    # Test on empty board
    empty_board = np.zeros((1, 9), dtype=np.float32)

    print("\nOracle's choice on empty board:")
    action, _ = oracle.predict(empty_board, deterministic=True)
    print(f"  → Position {action[0]} (deterministic)")

    # Test with exploration
    print("\nOracle's choices with exploration (10 samples):")
    action_counts = {}
    for _ in range(100):
        action, _ = oracle.predict(empty_board, deterministic=False)
        action_counts[action[0]] = action_counts.get(action[0], 0) + 1

    print("  Distribution:")
    for pos in sorted(action_counts.keys(), key=lambda x: action_counts[x], reverse=True):
        pct = action_counts[pos] / 100 * 100
        print(f"    Position {pos}: {action_counts[pos]:3} times ({pct:5.1f}%)")

    return oracle


def test_tree_opening_preference(args):
    """Test what the DECISION TREE prefers"""
    print("\n" + "=" * 80)
    print("TESTING DECISION TREE OPENING PREFERENCE")
    print("=" * 80)

    tree_model = TreeWrapper.load(get_viper_path(args))

    empty_board = np.zeros((1, 9), dtype=np.float32)
    action = tree_model.predict(empty_board)[0][0]

    print(f"\nDecision tree's choice on empty board:")
    print(f"  → Position {action}")

    return tree_model


def analyze_feature_importance_mismatch(args):
    """Analyze why feature importance doesn't match actual strategy quality"""
    print("\n" + "=" * 80)
    print("ANALYZING FEATURE IMPORTANCE vs ACTUAL IMPORTANCE")
    print("=" * 80)

    tree_model = TreeWrapper.load(get_viper_path(args))
    tree = tree_model.tree

    print("\nFeature Importance (from sklearn):")
    feature_importance = tree.feature_importances_

    positions = ["TL(0)", "TC(1)", "TR(2)", "ML(3)", "C(4)", "MR(5)", "BL(6)", "BC(7)", "BR(8)"]

    for i in range(9):
        bar = "█" * int(feature_importance[i] * 50)
        print(f"  {positions[i]}: {feature_importance[i]:.4f} {bar}")

    print("\n" + "─" * 80)
    print("EXPLANATION: Feature Importance Meaning")
    print("─" * 80)
    print("""
Feature importance measures:
  • How often a feature is used in the tree
  • How much it reduces impurity (entropy/gini)
  • NOT the same as strategic importance!

Why Position 0 might have high importance:
  1. It's checked frequently in the tree (many branches)
  2. It helps split the decision space well
  3. But this doesn't mean it's the BEST opening move!

Think of it like this:
  • "Is opponent at top-left?" is a useful QUESTION
  • But "Should I start at top-left?" is a different DECISION
    """)


def check_training_data_bias(args):
    """Check if training data has bias"""
    print("\n" + "=" * 80)
    print("CHECKING FOR TRAINING DATA BIAS")
    print("=" * 80)

    print("""
Potential issues in VIPER training:

1. SAMPLING BIAS:
   • VIPER samples trajectories using current policy
   • If policy initially prefers corners, it generates more corner data
   • Decision tree trained on biased data → reinforces corner preference

2. BETA PARAMETER:
   • beta=1 in first iteration: uses only oracle policy
   • beta=0 in later iterations: uses learned tree policy
   • If tree learns wrong strategy early, it snowballs

3. STATE DISTRIBUTION:
   • Tree trained on states that occur during play
   • Empty board might be rare in training data!
   • Tree optimizes for mid-game positions, not opening

Let's check the VIPER training parameters...
    """)

    # Check VIPER code
    print("\nVIPER Training Parameters:")
    print("  n_iter: Number of VIPER iterations")
    print("  beta: Probability of using oracle vs tree (0=tree, 1=oracle)")
    print("  total_timesteps: Steps per iteration")

    print("\n" + "─" * 80)
    print("HYPOTHESIS: The decision tree learned from biased data")
    print("─" * 80)
    print("""
Most likely cause:
  → The oracle (neural network) might also prefer corners!
  → Or the tree was trained on mostly mid-game states
  → Opening move preference is underrepresented in training data

To verify: Check what the oracle (DQN) does on empty board.
    """)


def suggest_fixes(args):
    """Suggest how to fix the problem"""
    print("\n" + "=" * 80)
    print("SUGGESTED FIXES")
    print("=" * 80)

    print("""
1. CHECK ORACLE BEHAVIOR:
   • Test if DQN also prefers corners
   • If yes, the problem is in DQN training
   • If no, the problem is in VIPER extraction

2. RETRAIN WITH BETTER REWARD SHAPING:
   • Add bonus for taking center on empty board
   • Penalize losing when you had center advantage
   • This guides learning toward better strategy

3. INCREASE OPENING STATE SAMPLING:
   • Ensure VIPER sees more empty/near-empty boards
   • Currently might be biased toward mid-game states

4. USE EXPERT KNOWLEDGE:
   • Manually encode "if board empty, prefer center"
   • Then let tree learn the rest

5. CHECK EXPLORATION:
   • DQN exploration_final_eps = 0.05
   • Might be too low, not exploring center enough

RECOMMENDED ACTION:
  1. First check: python diagnose_problem.py
  2. If oracle prefers corners → retrain DQN with better hyperparameters
  3. If oracle prefers center → retrain VIPER with more iterations
    """)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="TicTacToe-v0")
    parser.add_argument("--n-env", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--max-leaves", type=int, default=None)
    parser.add_argument("--log-prefix", type=str, default="")
    parser.add_argument("--ep-horizon", type=int, default=150)
    parser.add_argument("--rand-ball-start", action='store_true')
    parser.add_argument("--render", action='store_true')
    args = parser.parse_args()

    print("=" * 80)
    print("DIAGNOSING DECISION TREE STRATEGY PROBLEM")
    print("=" * 80)

    # Test oracle
    oracle = test_oracle_opening_preference(args)

    # Test tree
    tree = test_tree_opening_preference(args)

    # Analyze feature importance
    analyze_feature_importance_mismatch(args)

    # Check training bias
    check_training_data_bias(args)

    # Suggest fixes
    suggest_fixes(args)

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
