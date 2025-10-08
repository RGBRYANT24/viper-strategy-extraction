"""
可视化 VIPER 提取的决策树
使用方法: python visualize_tree.py --env-name TicTacToe-v0
"""
import argparse
from sklearn.tree import export_text, plot_tree
import matplotlib
matplotlib.use('Agg')  # 无需显示器的后端
import matplotlib.pyplot as plt
from model.paths import get_viper_path
from model.tree_wrapper import TreeWrapper


def visualize_tree(args):
    """可视化决策树"""
    # 加载决策树
    model = TreeWrapper.load(get_viper_path(args))
    tree = model.tree

    print("=" * 60)
    print("决策树信息")
    print("=" * 60)
    model.print_info()
    print()

    # 定义特征名称（TicTacToe 9个位置）
    feature_names = [f"pos_{i}" for i in range(9)]

    # 定义类别名称（9个可能的动作）
    class_names = [f"action_{i}" for i in range(9)]

    # 1. 文本形式展示决策树规则
    print("=" * 60)
    print("决策树规则（文本形式）")
    print("=" * 60)
    tree_rules = export_text(
        tree,
        feature_names=feature_names,
        max_depth=5  # 只显示前5层，避免过长
    )
    print(tree_rules)
    print()

    # 2. 图形化展示（保存为图片）
    print("=" * 60)
    print("生成决策树可视化图...")
    print("=" * 60)

    plt.figure(figsize=(25, 15))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=4  # 只显示前4层，避免图片过大
    )

    output_path = f"tree_visualization_{args.env_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 决策树图片已保存到: {output_path}")
    print()

    # 3. 显示一些关键统计
    print("=" * 60)
    print("决策树统计信息")
    print("=" * 60)
    print(f"节点总数: {tree.tree_.node_count}")
    print(f"叶子节点数: {tree.get_n_leaves()}")
    print(f"树深度: {tree.get_depth()}")
    print(f"特征数量: {tree.n_features_in_}")
    print(f"输出类别数: {tree.n_classes_}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 VIPER 决策树")
    parser.add_argument("--env-name", type=str, default="TicTacToe-v0", help="环境名称")
    parser.add_argument("--max-depth", type=int, default=None, help="决策树最大深度")
    parser.add_argument("--max-leaves", type=int, default=None, help="决策树最大叶子数")
    parser.add_argument("--log-prefix", type=str, default="", help="日志前缀")

    args = parser.parse_args()
    visualize_tree(args)
