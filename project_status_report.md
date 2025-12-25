# 📊 Tic-Tac-Toe VIPER Project Status Report

## 1. 项目目标 (Project Goal)
在 Tic-Tac-Toe 环境中实现并改进 **VIPER (Verifiable Reinforcement Learning via Policy Extraction)** 算法。

*   **核心任务**: 将高性能但不可解释的“神经网络 (Neural Network)”策略，蒸馏为可解释、可验证的“决策树 (Decision Tree)”策略。
*   **预期成果**: 获得一个既拥有高胜率（与 NN 相当），又具备完全可解释性的轻量级 Tic-Tac-Toe AI。

---

## 2. 核心方法论: VIPER 详解 (Methodology)

**VIPER** (Verifiable Reinforcement Learning via Policy Extraction) 的核心理念是将深度强化学习（DRL）的“直觉”转化为决策树的“规则”。我们不直接训练决策树（因为很难训练），而是先训练一个强大的神经网络（Oracle），然后让决策树（Student）去模仿它。

### 2.1 为什么选择 VIPER?
*   **Verifiable (可验证)**: 决策树的逻辑是完全透明的，我们可以从数学上证明它在某些情况下的行为。
*   **Interpretable (可解释)**: 非技术人员也能看懂 `if-else` 规则。
*   **Policy Extraction (策略蒸馏)**: 从复杂的神经网络中“提取”出核心策略。

### 2.2 核心机制: Q-Value 引导的模仿学习
普通的模仿学习（Behavior Cloning）只是简单地照抄老师的动作。**VIPER 更聪明，它利用 Q-Value 来判断错误的严重性。**

*   **Loss Weighting (关键公式)**:
    在训练决策树时，每个样本 $(s, a)$ 都有一个权重 $W(s)$：
    $$W(s) = \max_a Q(s, a) - \min_a Q(s, a)$$
    *   **关键局面 (Critical State)**: 如果这一步走错会导致输掉比赛（Q值差异巨大），权重 $W$ 会非常大 $\rightarrow$ **决策树必须学会！**
    *   **无关紧要 (Trivial State)**: 如果这一步无论怎么走都能赢（Q值差异很小），权重 $W$ 很小 $\rightarrow$ **决策树可以忽略，节省脑容量。**

### 2.3 训练流程 (DAGGER 循环)
我们使用 **DAGGER (Dataset Aggregation)** 算法进行迭代训练：
1.  **初始化**: 收集一批神经网络玩游戏的数据。
2.  **训练**: 用这批数据训练第一棵决策树。
3.  **混合采样 (Rollout)**: 让决策树去玩游戏，但偶尔（概率 $\beta$）让神经网络接管纠正。这能让决策树遇到它“不熟悉”的局面，并由神经网络告诉它正确答案。
4.  **聚合**: 将新遇到的局面加入数据集并重复训练。

### 2.4 我们的改进 (Project Improvements)
为了在 Tic-Tac-Toe 上达到完美效果，我们在 `train_viper_improved.py` 中实现了以下增强：

*   **🔄 对称性数据增强 (Symmetry Augmentation)**
    *   利用井字棋的旋转（90°, 180°）和翻转特性，将每个样本自动扩展为 8 个变体。
    *   **效果**: 数据效率提升 **8倍**，极大加速收敛。
*   **🧭 动态探索策略 (Dynamic Exploration)**
    *   引入 `epsilon_random` 衰减机制。在训练初期强制随机乱走，确保决策树见过各种“烂摊子”局面，学会如何翻盘，避免陷入局部最优。
*   **📊 状态覆盖监控 (State Coverage)**
    *   实时监控模型见过的唯一状态数量 (Unique States)，确保模型真正覆盖了游戏的状态空间。

---

## 3. 当前架构与工作流 (Current Architecture)
我们已经构建了完整的 **训练 -> 蒸馏 -> 评估 -> 分析** 闭环：

*   **🧠 Training (Oracle 构建)**
    *   使用 **PPO (Proximal Policy Optimization)** 结合 **Self-Play** 机制训练神经网络。
    *   代码: `train/train_delta_selfplay_ppo.py`

*   **🌳 Extraction (策略蒸馏)**
    *   利用改进版 VIPER 方法，寻找“最小但最强”的决策树。
    *   代码: `train/train_viper_improved.py`

*   **⚔️ Evaluation (对战评估)**
    *   **NN vs Tree**: 验证决策树能否逼平或战胜导师网络 (`evaluation/battle_nn_vs_tree.py`)。
    *   **Tree vs Tree (Tournament)**: 循环赛机制，让不同参数生成的决策树互相对战，选出最优解 (`evaluation/battle_tree_vs_tree.py`)。

---

## 4. 深度分析工具 (Advanced Analysis Tools)
为了不仅仅看“胜率”，而是深入理解模型逻辑，我们开发了以下工具：

*   **🔍 特征重要性分析 (Feature Importance)**
    *   **功能**: 分析决策树在做决策时最看重棋盘的哪些位置（例如：是否优先占领中心？）。
    *   **代码**: `analysis/comparison/compare_feature_importance.py`

*   **⚡ 关键局面测试 (Critical State Analysis)**
    *   **功能**: 在手动定义的“关键时刻”（如：只差一步就赢、必须防守的局面）强制测试模型，快速定位低级失误。
    *   **代码**: `analysis/comparison/compare_critical_states.py`


