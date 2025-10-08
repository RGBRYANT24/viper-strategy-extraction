"""
TicTacToe Decision Tree Visualization - English Version
Usage: python visualize_tictactoe_english.py
"""
import argparse
import numpy as np
from sklearn.tree import export_text
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.paths import get_viper_path
from model.tree_wrapper import TreeWrapper


def position_to_grid(pos):
    """Convert position number to grid coordinates"""
    # Position mapping:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    row = pos // 3
    col = pos % 3
    return row, col


def visualize_board_positions():
    """Visualize board position numbers"""
    print("\nBoard Position Layout:")
    print("+---+---+---+")
    for row in range(3):
        print("|", end="")
        for col in range(3):
            pos = row * 3 + col
            print(f" {pos} |", end="")
        print()
        if row < 2:
            print("+---+---+---+")
    print("+---+---+---+")
    print("\nPosition Names:")
    position_names = [
        "Top-Left", "Top-Center", "Top-Right",
        "Mid-Left", "Center", "Mid-Right",
        "Bottom-Left", "Bottom-Center", "Bottom-Right"
    ]
    for i, name in enumerate(position_names):
        print(f"  Position {i}: {name}")
    print()


def analyze_tree(args):
    """Detailed analysis of decision tree"""
    print("=" * 80)
    print("TicTacToe Decision Tree Analysis")
    print("=" * 80)

    # Load decision tree
    model = TreeWrapper.load(get_viper_path(args))
    tree = model.tree

    # 1. Basic Information
    print("\n[BASIC INFORMATION]")
    print(f"Tree Depth: {tree.get_depth()}")
    print(f"Number of Leaves: {tree.get_n_leaves()}")
    print(f"Total Nodes: {tree.tree_.node_count}")
    print(f"Input Features: {tree.n_features_in_} (9 board positions)")
    print(f"Output Classes: {tree.n_classes_} (9 possible actions)")

    # 2. Board Layout
    visualize_board_positions()

    # 3. Feature Importance
    print("\n[FEATURE IMPORTANCE] (Which positions matter most)")
    feature_importance = tree.feature_importances_
    important_features = sorted(enumerate(feature_importance), key=lambda x: x[1], reverse=True)

    print("\nPos | Importance | Location          | Bar Chart")
    print("-" * 70)
    position_names = [
        "Top-Left", "Top-Center", "Top-Right",
        "Mid-Left", "Center", "Mid-Right",
        "Bottom-Left", "Bottom-Center", "Bottom-Right"
    ]
    for pos, importance in important_features:
        row, col = position_to_grid(pos)
        bar = "█" * int(importance * 50)
        print(f" {pos}  | {importance:7.4f}  | {position_names[pos]:16} | {bar}")

    # 4. Key Insights
    print("\n[KEY INSIGHTS]")
    top_3_positions = [pos for pos, _ in important_features[:3]]
    print(f"Top 3 Most Important Positions: {top_3_positions}")
    print(f"  - These positions are checked most often in the decision tree")

    # Strategic interpretation
    strategic_positions = []
    if 4 in top_3_positions:
        strategic_positions.append("Center (pos 4) - Critical control point")
    if any(p in top_3_positions for p in [0, 2, 6, 8]):
        strategic_positions.append("Corners - Strategic positions")
    if any(p in top_3_positions for p in [1, 3, 5, 7]):
        strategic_positions.append("Edges - Blocking/attacking positions")

    if strategic_positions:
        print(f"\n  Strategy Focus:")
        for strat in strategic_positions:
            print(f"    • {strat}")

    # 5. Text Rules (first 5 levels)
    print("\n" + "=" * 80)
    print("[DECISION RULES] (First 5 levels)")
    print("=" * 80)
    print("\nInterpretation Guide:")
    print("  State Values: -1.0 = Opponent(O), 0.0 = Empty, 1.0 = Your(X)")
    print()
    print("  Decision Conditions:")
    print("  • pos_X <= -0.5  means: Position X IS opponent's piece (O)")
    print("  • pos_X > -0.5   means: Position X is NOT opponent's (empty or yours)")
    print("  • pos_X <= 0.5   means: Position X is NOT yours (empty or opponent's)")
    print("  • pos_X > 0.5    means: Position X IS your piece (X)")
    print()

    feature_names = [f"pos_{i}" for i in range(9)]
    tree_rules = export_text(
        tree,
        feature_names=feature_names,
        max_depth=5
    )
    print(tree_rules)

    # 6. Generate Visualizations
    print("\n" + "=" * 80)
    print("[GENERATING VISUALIZATIONS]")
    print("=" * 80)

    try:
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(36, 12))

        # Subplot 1: Decision Tree (first 3 levels for clarity)
        ax1 = plt.subplot(1, 3, 1)
        from sklearn.tree import plot_tree
        plot_tree(
            tree,
            ax=ax1,
            feature_names=feature_names,
            class_names=[f"action_{i}" for i in range(9)],
            filled=True,
            rounded=True,
            fontsize=10,
            max_depth=3  # Only 3 levels to avoid overlap
        )
        ax1.set_title("Decision Tree Structure (First 3 Levels)", fontsize=18, pad=20, weight='bold')

        # Subplot 2: Feature Importance
        ax2 = plt.subplot(1, 3, 2)
        positions = list(range(9))
        importances = [feature_importance[i] for i in positions]
        colors = plt.cm.RdYlGn(np.array(importances))

        bars = ax2.barh(positions, importances, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_yticks(positions)
        ax2.set_yticklabels([f"Pos {i} ({position_names[i]})" for i in positions], fontsize=11)
        ax2.set_xlabel("Importance Score", fontsize=14, weight='bold')
        ax2.set_title("Feature Importance by Position", fontsize=18, pad=20, weight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{importances[i]:.3f}',
                    ha='left', va='center', fontsize=10, weight='bold')

        # Subplot 3: Board Layout Heatmap
        ax3 = plt.subplot(1, 3, 3)
        board_importance = np.array(importances).reshape(3, 3)

        im = ax3.imshow(board_importance, cmap='RdYlGn', interpolation='nearest')
        ax3.set_xticks([0, 1, 2])
        ax3.set_yticks([0, 1, 2])
        ax3.set_xticklabels(['Col 0', 'Col 1', 'Col 2'], fontsize=12)
        ax3.set_yticklabels(['Row 0', 'Row 1', 'Row 2'], fontsize=12)
        ax3.set_title("Board Position Importance Heatmap", fontsize=18, pad=20, weight='bold')

        # Add text annotations
        for i in range(3):
            for j in range(3):
                pos = i * 3 + j
                text = ax3.text(j, i, f'Pos {pos}\n{importances[pos]:.3f}',
                               ha="center", va="center", color="black",
                               fontsize=12, weight='bold',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Importance', fontsize=12, weight='bold')

        # Save figure
        output_path = f"tictactoe_tree_analysis_english.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✓ Visualization saved: {output_path}")

        # Also save a high-level tree view
        fig2 = plt.figure(figsize=(40, 20))
        plot_tree(
            tree,
            feature_names=feature_names,
            class_names=[f"Act_{i}" for i in range(9)],
            filled=True,
            rounded=True,
            fontsize=11,
            max_depth=4,
            proportion=True
        )
        plt.title("Full Decision Tree (First 4 Levels)", fontsize=24, pad=30, weight='bold')
        output_path2 = f"tictactoe_tree_full.png"
        plt.savefig(output_path2, dpi=200, bbox_inches='tight')
        print(f"✓ Full tree visualization saved: {output_path2}")

    except Exception as e:
        print(f"⚠ Error generating visualization: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  1. tictactoe_tree_analysis_english.png - Comprehensive analysis (3 panels)")
    print("  2. tictactoe_tree_full.png - Full decision tree structure")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TicTacToe Decision Tree Analysis (English)")
    parser.add_argument("--env-name", type=str, default="TicTacToe-v0", help="Environment name")
    parser.add_argument("--max-depth", type=int, default=10, help="Max tree depth")
    parser.add_argument("--max-leaves", type=int, default=None, help="Max leaf nodes")
    parser.add_argument("--log-prefix", type=str, default="", help="Log prefix")

    args = parser.parse_args()
    analyze_tree(args)
