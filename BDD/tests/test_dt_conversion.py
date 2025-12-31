
import sys
import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from dd.autoref import BDD

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from BDD.utils.decision_tree_to_bdd import DecisionTreeToBDD

def test_simple_tree_conversion():
    # 1. Setup BDD Manager
    bdd = BDD()
    bdd.configure(reordering=True)
    
    # 2. Train a very simple Decision Tree (Tic-Tac-Toe like input)
    # 9 features (cells), random values for demo
    # We force a small tree with max_depth=2 for readability
    X = np.random.choice([-1, 0, 1], size=(200, 9))
    print("X:", X)
    print("X shape:", X.shape)
    # Dummy target: policy probabilities for 9 actions
    y = np.random.rand(200, 9) 
    print("y:", y)
    print("y shape:", y.shape)
    
    clf = DecisionTreeRegressor(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    print("Decision Tree structure:")
    print(f"Nodes: {clf.tree_.node_count}")
    print("-" * 20)
    
    # 打印底层数组结构帮助理解
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    value = clf.tree_.value

    print("index | left | right | feature | threshold | value")
    for i in range(n_nodes):
        v = value[i][0] if len(value[i]) == 1 else value[i]
        # 简化打印 value，只打印前几个元素如果太长
        val_str = str(v) if len(v) < 5 else str(v[:4]) + "..."
        print(f"{i:5d} | {children_left[i]:4d} | {children_right[i]:5d} | {feature[i]:7d} | {threshold[i]:9.3f} | {val_str}")
    print("-" * 20)

    # 3. Initialize Converter
    converter = DecisionTreeToBDD(bdd, game='tic_tac_toe')
    
    # 4. Run Conversion
    print("Converting Tree to BDD...")
    action_space = 9
    bdd_policies = converter.recursive_build(clf.tree_, 0, action_space)
    
    # 5. Verify Results
    print("\nConversion Result:")
    for action_idx, node in bdd_policies.items():
        count = bdd.count(node)
        print(f"Action {action_idx}: BDD Node {node}, Satisfying assignments: {count}")
        if count > 0:
             print(f"  (Action is possible in specific states)")
        else:
             print(f"  (Action is never taken - might be pruned or low probability)")

    print("\nTest Finished Successfully.")

    # bdd_policies[0].bdd.dump_bdd(bdd_policies[0])

if __name__ == "__main__":
    test_simple_tree_conversion()
