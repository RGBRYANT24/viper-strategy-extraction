#!/bin/bash
# 规则提取示例脚本

# 使用决策树进行重要性排序，不简化规则
PYTHONPATH=.:$PYTHONPATH python analysis/export/extract_tree_rules.py \
  --tree-path log/viper_mask_ppo_tictactoe/viper_mask_ppo_tree_20251107_001324_iter10_samples50000_depth10_leaves50.joblib \
  --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
  --env-name TicTacToe-v0 \
  --no-simplify \
  --compute-priority \
  --sort-by-priority \
  --priority-by-tree \
  --verbose

# 输出文件将自动命名为：
# log/viper_mask_ppo_tictactoe/viper_mask_ppo_tree_20251107_001324_iter10_samples50000_depth10_leaves50_rules_YYYYMMDD_HHMMSS_nosimplify_priority-tree_sorted.txt

# 其他示例：

# 1. 使用Oracle进行重要性排序，不简化规则
# PYTHONPATH=.:$PYTHONPATH python analysis/export/extract_tree_rules.py \
#   --tree-path log/viper_mask_ppo_tictactoe/viper_mask_ppo_tree_20251107_001324_iter10_samples50000_depth10_leaves50.joblib \
#   --oracle-path log/oracle_TicTacToe_selfplay.zip \
#   --env-name TicTacToe-v0 \
#   --no-simplify \
#   --compute-priority \
#   --sort-by-priority \
#   --verbose

# 2. 简化规则 + 使用决策树计算优先级 + 排序
# PYTHONPATH=.:$PYTHONPATH python analysis/export/extract_tree_rules.py \
#   --tree-path log/viper_mask_ppo_tictactoe/viper_mask_ppo_tree_20251107_001324_iter10_samples50000_depth10_leaves50.joblib \
#   --oracle-path log/oracle_TicTacToe_selfplay.zip \
#   --env-name TicTacToe-v0 \
#   --n-samples 5000 \
#   --compute-priority \
#   --sort-by-priority \
#   --priority-by-tree \
#   --verbose

# 3. 仅提取规则，不简化，不计算优先级
# PYTHONPATH=.:$PYTHONPATH python analysis/export/extract_tree_rules.py \
#   --tree-path log/viper_mask_ppo_tictactoe/viper_mask_ppo_tree_20251107_001324_iter10_samples50000_depth10_leaves50.joblib \
#   --no-simplify \
#   --verbose
