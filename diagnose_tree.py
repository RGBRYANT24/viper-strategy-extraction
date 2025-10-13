"""
诊断决策树为什么一直预测动作1
"""
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier

# 加载决策树
tree_path = "log/viper_TicTacToe-v0_all-leaves_all-depth.joblib"
tree = joblib.load(tree_path)

print("=" * 70)
print("决策树诊断")
print("=" * 70)
print(f"类型: {type(tree)}")
print(f"深度: {tree.get_depth()}")
print(f"叶子数: {tree.get_n_leaves()}")
print(f"特征数: {tree.n_features_in_}")
print(f"类别: {tree.classes_}")
print(f"类别数: {tree.n_classes_}")

# 检查训练数据的类别分布
print("\n" + "=" * 70)
print("检查树的结构")
print("=" * 70)

# 获取树的各个节点的预测
tree_ = tree.tree_
print(f"节点数: {tree_.node_count}")
print(f"叶子节点的值分布:")

for i in range(tree_.node_count):
    if tree_.children_left[i] == tree_.children_right[i]:  # 叶子节点
        values = tree_.value[i][0]
        predicted_class = np.argmax(values)
        samples = np.sum(values)
        print(f"  叶子节点 {i}: 预测={predicted_class}, 样本数={samples}, 分布={values}")

# 测试不同状态的预测
print("\n" + "=" * 70)
print("测试预测")
print("=" * 70)

test_cases = [
    ("空棋盘", np.zeros(9, dtype=np.float32)),
    ("X在位置0", np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
    ("X在位置1", np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
    ("X在位置4", np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)),
    ("X在0,O在1", np.array([1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
]

for name, state in test_cases:
    pred = tree.predict(state.reshape(1, -1))[0]
    # 获取决策路径
    decision_path = tree.decision_path(state.reshape(1, -1))
    leaf_id = decision_path.indices[-1]

    print(f"\n{name}:")
    print(f"  状态: {state}")
    print(f"  预测: {pred}")
    print(f"  叶子节点: {leaf_id}")
    print(f"  合法动作: {np.where(state == 0)[0]}")
    print(f"  是否合法: {pred in np.where(state == 0)[0]}")

# 打印决策树规则
print("\n" + "=" * 70)
print("决策树规则（简化）")
print("=" * 70)

def print_tree(tree, feature_names=None, node=0, depth=0):
    """递归打印决策树"""
    tree_ = tree.tree_
    feature = tree_.feature[node]
    threshold = tree_.threshold[node]

    indent = "  " * depth

    if tree_.children_left[node] == tree_.children_right[node]:  # 叶子
        values = tree_.value[node][0]
        pred_class = np.argmax(values)
        print(f"{indent}→ 预测: {pred_class} (样本数: {int(np.sum(values))})")
    else:
        if feature_names:
            feature_name = feature_names[feature]
        else:
            feature_name = f"位置{feature}"

        print(f"{indent}如果 {feature_name} <= {threshold:.1f}:")
        print_tree(tree, feature_names, tree_.children_left[node], depth + 1)
        print(f"{indent}否则:")
        print_tree(tree, feature_names, tree_.children_right[node], depth + 1)

feature_names = [f"位置{i}" for i in range(9)]
print_tree(tree, feature_names)

# 检查是否所有预测都是1
print("\n" + "=" * 70)
print("随机采样100个状态测试")
print("=" * 70)

np.random.seed(42)
predictions = []
for _ in range(100):
    state = np.random.choice([-1, 0, 1], size=9).astype(np.float32)
    pred = tree.predict(state.reshape(1, -1))[0]
    predictions.append(pred)

unique, counts = np.unique(predictions, return_counts=True)
print("预测分布:")
for val, count in zip(unique, counts):
    print(f"  动作{val}: {count}次 ({count/100*100:.1f}%)")

if len(unique) == 1:
    print("\n⚠️  警告：决策树只预测一个动作！训练数据有问题。")
