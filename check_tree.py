import joblib
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('./log/viper_CartPole-v1_all-leaves_all-depth.joblib')

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(model, 
          feature_names=['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity'],
          class_names=['Left', 'Right'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()