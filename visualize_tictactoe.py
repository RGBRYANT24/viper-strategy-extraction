"""
TicTacToe 决策树可视化 - 带棋盘解释
使用方法: python visualize_tictactoe.py
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
    """将位置编号转换为棋盘坐标"""
    # 位置编号:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    row = pos // 3
    col = pos % 3
    return row, col


def visualize_board_positions():
    """可视化棋盘位置编号"""
    print("\n棋盘位置编号说明:")
    print("┌───┬───┬───┐")
    for row in range(3):
        print("│", end="")
        for col in range(3):
            pos = row * 3 + col
            print(f" {pos} │", end="")
        print()
        if row < 2:
            print("├───┼───┼───┤")
    print("└───┴───┴───┘")
    print()


def explain_feature(feature_idx, threshold, samples):
    """解释特征条件"""
    pos = feature_idx
    row, col = position_to_grid(pos)

    # threshold 接近的值
    if abs(threshold - (-0.5)) < 0.1:
        state = "空或X (≤ -0.5)"
    elif abs(threshold - 0.5) < 0.1:
        state = "不是O (≤ 0.5)"
    elif abs(threshold - 0) < 0.1:
        state = "空 (= 0)"
    elif threshold < -0.5:
        state = "O (< -0.5)"
    elif threshold > 0.5:
        state = "X (> 0.5)"
    else:
        state = f"阈值 {threshold:.2f}"

    explanation = f"位置 {pos} (第{row}行第{col}列) {state}"
    return explanation


def get_sample_decision_paths(tree, n_samples=5):
    """获取一些样本决策路径"""
    paths = []

    # 获取一些叶子节点
    leaves = [i for i in range(tree.tree_.node_count) if tree.tree_.children_left[i] == -1]

    # 随机选择几个叶子
    sample_leaves = np.random.choice(leaves, min(n_samples, len(leaves)), replace=False)

    for leaf_id in sample_leaves:
        path = []
        node_id = 0  # 从根节点开始

        # 追踪到叶子的路径
        while node_id != leaf_id:
            feature = tree.tree_.feature[node_id]
            threshold = tree.tree_.threshold[node_id]

            # 判断应该走左还是右（这里简化处理）
            left_child = tree.tree_.children_left[node_id]
            right_child = tree.tree_.children_right[node_id]

            path.append({
                'node': node_id,
                'feature': feature,
                'threshold': threshold,
            })

            # 简单启发式：选择样本更多的分支
            if tree.tree_.n_node_samples[left_child] > tree.tree_.n_node_samples[right_child]:
                node_id = left_child
                path[-1]['direction'] = 'left'
            else:
                node_id = right_child
                path[-1]['direction'] = 'right'

        # 获取最终动作
        value = tree.tree_.value[leaf_id]
        action = np.argmax(value)
        paths.append((path, action))

    return paths


def analyze_tree(args):
    """详细分析决策树"""
    print("=" * 80)
    print("TicTacToe 决策树分析")
    print("=" * 80)

    # 加载决策树
    model = TreeWrapper.load(get_viper_path(args))
    tree = model.tree

    # 1. 基本信息
    print("\n【基本信息】")
    print(f"树深度: {tree.get_depth()}")
    print(f"叶子节点数: {tree.get_n_leaves()}")
    print(f"总节点数: {tree.tree_.node_count}")
    print(f"输入特征数: {tree.n_features_in_} (对应棋盘9个位置)")
    print(f"输出类别数: {tree.n_classes_} (对应9个可能的落子位置)")

    # 2. 棋盘说明
    visualize_board_positions()

    # 3. 特征重要性
    print("\n【特征重要性】(哪些棋盘位置最重要)")
    feature_importance = tree.feature_importances_
    important_features = sorted(enumerate(feature_importance), key=lambda x: x[1], reverse=True)

    print("\n位置编号 | 重要性  | 棋盘位置")
    print("-" * 40)
    for pos, importance in important_features[:9]:
        row, col = position_to_grid(pos)
        position_name = ["左上", "上中", "右上", "左中", "中心", "右中", "左下", "下中", "右下"][pos]
        bar = "█" * int(importance * 50)
        print(f"   {pos}     | {importance:.4f} | 第{row}行第{col}列 ({position_name}) {bar}")

    # 4. 文本规则（前5层）
    print("\n" + "=" * 80)
    print("【决策规则】(前5层)")
    print("=" * 80)

    feature_names = [f"pos_{i}" for i in range(9)]
    tree_rules = export_text(
        tree,
        feature_names=feature_names,
        max_depth=5
    )
    print(tree_rules)

    # 5. 示例决策路径
    print("\n" + "=" * 80)
    print("【示例决策路径】")
    print("=" * 80)

    print("\n解读说明:")
    print("- 状态值: -1 表示对手(O), 0 表示空, 1 表示己方(X)")
    print("- 决策树通过检查各个位置的状态来选择最佳落子位置\n")

    # 生成图片
    print("\n" + "=" * 80)
    print("【生成可视化图片】")
    print("=" * 80)

    try:
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

        # 左图：决策树（前4层）
        from sklearn.tree import plot_tree
        plot_tree(
            tree,
            ax=ax1,
            feature_names=feature_names,
            class_names=[f"action_{i}" for i in range(9)],
            filled=True,
            rounded=True,
            fontsize=8,
            max_depth=4
        )
        ax1.set_title("决策树结构 (前4层)", fontsize=16, pad=20)

        # 右图：特征重要性
        positions = list(range(9))
        importances = [feature_importance[i] for i in positions]
        colors = plt.cm.RdYlGn(np.array(importances))

        ax2.barh(positions, importances, color=colors)
        ax2.set_yticks(positions)
        ax2.set_yticklabels([f"位置{i}" for i in positions])
        ax2.set_xlabel("重要性", fontsize=12)
        ax2.set_title("棋盘位置重要性", fontsize=16, pad=20)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)

        # 保存
        output_path = f"tictactoe_tree_analysis.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 可视化图片已保存: {output_path}")

    except Exception as e:
        print(f"⚠️  生成图片时出错: {e}")

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TicTacToe 决策树详细分析")
    parser.add_argument("--env-name", type=str, default="TicTacToe-v0", help="环境名称")
    parser.add_argument("--max-depth", type=int, default=10, help="决策树最大深度")
    parser.add_argument("--max-leaves", type=int, default=None, help="决策树最大叶子数")
    parser.add_argument("--log-prefix", type=str, default="", help="日志前缀")

    args = parser.parse_args()
    analyze_tree(args)
