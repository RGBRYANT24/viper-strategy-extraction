
import numpy as np
from sklearn.tree import DecisionTreeRegressor

X = np.random.choice([-1, 0, 1], size=(20, 9))
y = np.random.rand(20, 9) 

clf = DecisionTreeRegressor(max_depth=3, random_state=42)
clf.fit(X, y)

print("clf.tree_.value.shape:", clf.tree_.value.shape)
node_id = 0
print(f"clf.tree_.value[{node_id}] shape:", clf.tree_.value[node_id].shape)
print(f"clf.tree_.value[{node_id}] content:\n", clf.tree_.value[node_id])

leaf_vector_wrong = clf.tree_.value[node_id][0]
print(f"clf.tree_.value[{node_id}][0] (current code):", leaf_vector_wrong)
print(f"len(leaf_vector_wrong):", len(leaf_vector_wrong))

leaf_vector_correct = clf.tree_.value[node_id][:, 0]
print(f"clf.tree_.value[{node_id}][:, 0] (proposed fix):", leaf_vector_correct)
print(f"len(leaf_vector_correct):", len(leaf_vector_correct))
