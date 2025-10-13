"""
调试决策树模型，查看它的输出
"""
import numpy as np
import joblib

# 加载决策树
tree_path = "log/viper_TicTacToe-v0_all-leaves_all-depth.joblib"
model = joblib.load(tree_path)

print("=" * 70)
print("决策树模型信息")
print("=" * 70)
print(f"模型类型: {type(model)}")
print(f"模型属性: {dir(model)}")

# 检查是否是TreeWrapper
if hasattr(model, 'tree'):
    print("\n这是一个TreeWrapper对象")
    print(f"内部决策树: {type(model.tree)}")
    actual_tree = model.tree
else:
    print("\n这是直接的决策树对象")
    actual_tree = model

print(f"\n决策树深度: {actual_tree.get_depth() if hasattr(actual_tree, 'get_depth') else 'N/A'}")
print(f"叶子节点数: {actual_tree.get_n_leaves() if hasattr(actual_tree, 'get_n_leaves') else 'N/A'}")
print(f"特征数: {actual_tree.n_features_in_ if hasattr(actual_tree, 'n_features_in_') else 'N/A'}")
print(f"类别数: {actual_tree.n_classes_ if hasattr(actual_tree, 'n_classes_') else 'N/A'}")
print(f"类别: {actual_tree.classes_ if hasattr(actual_tree, 'classes_') else 'N/A'}")

# 测试几个状态
print("\n" + "=" * 70)
print("测试预测")
print("=" * 70)

test_states = [
    np.zeros(9, dtype=np.float32),  # 空棋盘
    np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),  # X在左上角
    np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),  # X在中心
]

for i, state in enumerate(test_states):
    print(f"\n测试状态 {i+1}:")
    print(f"  棋盘: {state}")

    # 测试TreeWrapper的predict
    if hasattr(model, 'predict'):
        try:
            action = model.predict(state.reshape(1, -1))
            print(f"  TreeWrapper输出: {action}")
            print(f"  输出类型: {type(action)}")
            print(f"  输出shape: {action.shape if hasattr(action, 'shape') else 'N/A'}")
            if hasattr(action, '__iter__'):
                print(f"  第一个元素: {action[0]}, 类型: {type(action[0])}")
        except Exception as e:
            print(f"  TreeWrapper预测错误: {e}")

    # 测试内部决策树的predict
    if hasattr(model, 'tree'):
        try:
            action = model.tree.predict(state.reshape(1, -1))
            print(f"  内部树输出: {action}")
            print(f"  输出类型: {type(action)}")
        except Exception as e:
            print(f"  内部树预测错误: {e}")

print("\n" + "=" * 70)
print("检查TreeWrapper的predict方法")
print("=" * 70)
if hasattr(model, 'predict'):
    import inspect
    print(inspect.getsource(model.predict))
