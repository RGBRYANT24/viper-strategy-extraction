# 文件整理计划

## 当前问题
根目录有59个文件/文件夹，过于杂乱，需要整理。

## 整理方案

### 1. 创建新目录结构

```bash
# 在服务器执行以下命令

# 创建分析工具目录
mkdir -p analysis/visualization
mkdir -p analysis/export
mkdir -p analysis/comparison

# 创建文档目录
mkdir -p docs/guides/visualization
mkdir -p docs/examples
mkdir -p docs/technical

# 创建评估目录（如果不存在）
mkdir -p evaluation

# 创建训练目录（如果不存在）
mkdir -p train
```

### 2. 移动可视化和分析工具

```bash
# 移动可视化工具
mv visualize_tree_rules.py analysis/visualization/
mv export_rules_report.py analysis/visualization/
mv analyze_rules_tactical.py analysis/visualization/
mv test_visualization.py analysis/visualization/

# 移动导出工具
mv export_tree_json.py analysis/export/
mv export_tree_text.py analysis/export/
mv extract_tree_rules.py analysis/export/
mv summarize_rules.py analysis/export/

# 移动对比分析工具
mv compare_rule_orderings.py analysis/comparison/
mv analyze_viper_results.py analysis/comparison/
```

### 3. 移动评估脚本

```bash
# 移动评估脚本
mv battle_nn_vs_tree.py evaluation/
mv battle_single_tree.py evaluation/
mv evaluate_rules.py evaluation/
```

### 4. 移动训练脚本

```bash
# 移动训练脚本（如果在根目录）
mv train_parallel_viper.py train/
mv train_viper_improved.py train/
mv run_viper_from_ppo.sh train/
```

### 5. 移动测试脚本

```bash
# 移动测试脚本
mv test_minmax.py test/
mv test_delta_selfplay.py test/
mv test_rule_priority.py test/
mv test_trees.py test/
mv example_use_regression_rules.py test/
```

### 6. 整理文档

```bash
# 移动可视化相关文档
mv VISUALIZATION_README.md docs/guides/visualization/
mv 使用说明.md docs/guides/visualization/
mv 战术分析说明.md docs/guides/visualization/
mv 工具总结.md docs/guides/visualization/

# 移动示例文件
mv example_output.txt docs/examples/
mv example_report.txt docs/examples/
mv tactical_analysis_example.txt docs/examples/
mv tictactoe_rules_*.txt docs/examples/

# 移动技术文档到 docs/guides/
mv VIPER_LEARNING_GUIDE.md docs/guides/
mv VIPER_MASKABLE_PPO_GUIDE.md docs/guides/
mv VIPER_QUICK_REFERENCE.md docs/guides/
mv VIPER_SAMPLE_TRAJECTORY_GUIDE.md docs/guides/
mv VIPER_TECHNICAL_ANALYSIS.md docs/guides/
mv TEST_TREES_README.md docs/guides/
mv README_PARALLEL_TRAINING.md docs/guides/
mv RULE_PRIORITY_USAGE.md docs/guides/
mv SUMMARY_RULE_CHANGES.md docs/guides/
```

### 7. 根目录保留文件

```bash
# 只保留这些重要文件在根目录：
# - README.md (项目主README)
# - .gitignore
# - .clinerules (项目规则)
# - requirements.txt
# - main.py (主入口)
# - 1805.08328.pdf (VIPER论文)
# - FILE_REORGANIZATION_PLAN.md (本文件，完成后可删除)
```

### 8. 更新导入路径

整理后需要更新部分文件的导入路径。以下文件可能需要修改：

#### analysis/visualization/ 下的文件
```python
# 如果有相对导入，需要调整
# 从: from model.tree_wrapper import TreeWrapper
# 到: from model.tree_wrapper import TreeWrapper  # 不变，因为从项目根执行
```

#### 运行方式调整
```bash
# 原来:
python visualize_tree_rules.py

# 现在:
python analysis/visualization/visualize_tree_rules.py
# 或者
cd analysis/visualization && python visualize_tree_rules.py
```

### 9. 创建快捷脚本（可选）

为常用命令创建快捷脚本：

```bash
# 创建 scripts/ 目录
mkdir -p scripts

# 创建快捷脚本
cat > scripts/visualize.sh <<'EOF'
#!/bin/bash
python analysis/visualization/visualize_tree_rules.py "$@"
EOF

cat > scripts/analyze_tactical.sh <<'EOF'
#!/bin/bash
python analysis/visualization/analyze_rules_tactical.py "$@"
EOF

cat > scripts/export_report.sh <<'EOF'
#!/bin/bash
python analysis/visualization/export_rules_report.py "$@"
EOF

chmod +x scripts/*.sh
```

使用示例：
```bash
# 使用快捷脚本
./scripts/visualize.sh --rule 1
./scripts/analyze_tactical.sh --max-rules 10
./scripts/export_report.sh --output report.txt
```

## 整理后的目录结构

```
viper-verifiable-rl-impl/
├── README.md                   # 项目主文档
├── .gitignore
├── .clinerules                 # 项目规则
├── requirements.txt
├── main.py
├── 1805.08328.pdf             # VIPER论文
│
├── analysis/                   # 分析工具
│   ├── visualization/          # 可视化工具
│   │   ├── visualize_tree_rules.py
│   │   ├── export_rules_report.py
│   │   ├── analyze_rules_tactical.py
│   │   └── test_visualization.py
│   ├── export/                 # 导出工具
│   │   ├── export_tree_json.py
│   │   ├── export_tree_text.py
│   │   ├── extract_tree_rules.py
│   │   └── summarize_rules.py
│   └── comparison/             # 对比分析
│       ├── compare_rule_orderings.py
│       └── analyze_viper_results.py
│
├── evaluation/                 # 评估脚本
│   ├── battle_nn_vs_tree.py
│   ├── battle_single_tree.py
│   └── evaluate_rules.py
│
├── train/                      # 训练脚本
│   ├── train_parallel_viper.py
│   ├── train_viper_improved.py
│   └── run_viper_from_ppo.sh
│
├── test/                       # 测试脚本
│   ├── test_minmax.py
│   ├── test_delta_selfplay.py
│   ├── test_rule_priority.py
│   ├── test_trees.py
│   └── example_use_regression_rules.py
│
├── docs/                       # 文档
│   ├── guides/                 # 使用指南
│   │   ├── visualization/      # 可视化文档
│   │   │   ├── VISUALIZATION_README.md
│   │   │   ├── 使用说明.md
│   │   │   ├── 战术分析说明.md
│   │   │   └── 工具总结.md
│   │   ├── VIPER_LEARNING_GUIDE.md
│   │   ├── VIPER_MASKABLE_PPO_GUIDE.md
│   │   ├── VIPER_QUICK_REFERENCE.md
│   │   ├── VIPER_SAMPLE_TRAJECTORY_GUIDE.md
│   │   ├── VIPER_TECHNICAL_ANALYSIS.md
│   │   ├── TEST_TREES_README.md
│   │   ├── README_PARALLEL_TRAINING.md
│   │   ├── RULE_PRIORITY_USAGE.md
│   │   └── SUMMARY_RULE_CHANGES.md
│   └── examples/               # 示例输出
│       ├── example_output.txt
│       ├── example_report.txt
│       ├── tactical_analysis_example.txt
│       ├── tictactoe_rules_50_10.txt
│       └── tictactoe_rules_100_10.txt
│
├── scripts/                    # 快捷脚本（可选）
│   ├── visualize.sh
│   ├── analyze_tactical.sh
│   └── export_report.sh
│
├── model/                      # 模型代码
├── gym_env/                    # 环境定义
├── log/                        # 日志和结果
├── archive/                    # 归档
├── debug/                      # 调试
└── verify/                     # 验证
```

## 执行步骤

### 一键执行脚本

创建一个自动整理脚本：

```bash
# 创建 reorganize.sh
cat > reorganize.sh <<'EOF'
#!/bin/bash
set -e

echo "开始整理文件结构..."

# 1. 创建目录
echo "创建目录结构..."
mkdir -p analysis/visualization
mkdir -p analysis/export
mkdir -p analysis/comparison
mkdir -p docs/guides/visualization
mkdir -p docs/examples
mkdir -p evaluation
mkdir -p train
mkdir -p scripts

# 2. 移动可视化工具
echo "移动可视化工具..."
mv -n visualize_tree_rules.py analysis/visualization/ 2>/dev/null || true
mv -n export_rules_report.py analysis/visualization/ 2>/dev/null || true
mv -n analyze_rules_tactical.py analysis/visualization/ 2>/dev/null || true
mv -n test_visualization.py analysis/visualization/ 2>/dev/null || true

# 3. 移动导出工具
echo "移动导出工具..."
mv -n export_tree_json.py analysis/export/ 2>/dev/null || true
mv -n export_tree_text.py analysis/export/ 2>/dev/null || true
mv -n extract_tree_rules.py analysis/export/ 2>/dev/null || true
mv -n summarize_rules.py analysis/export/ 2>/dev/null || true

# 4. 移动对比工具
echo "移动对比工具..."
mv -n compare_rule_orderings.py analysis/comparison/ 2>/dev/null || true
mv -n analyze_viper_results.py analysis/comparison/ 2>/dev/null || true

# 5. 移动评估脚本
echo "移动评估脚本..."
mv -n battle_nn_vs_tree.py evaluation/ 2>/dev/null || true
mv -n battle_single_tree.py evaluation/ 2>/dev/null || true
mv -n evaluate_rules.py evaluation/ 2>/dev/null || true

# 6. 移动训练脚本
echo "移动训练脚本..."
mv -n train_parallel_viper.py train/ 2>/dev/null || true
mv -n train_viper_improved.py train/ 2>/dev/null || true
mv -n run_viper_from_ppo.sh train/ 2>/dev/null || true

# 7. 移动测试脚本
echo "移动测试脚本..."
mv -n test_minmax.py test/ 2>/dev/null || true
mv -n test_delta_selfplay.py test/ 2>/dev/null || true
mv -n test_rule_priority.py test/ 2>/dev/null || true
mv -n test_trees.py test/ 2>/dev/null || true
mv -n example_use_regression_rules.py test/ 2>/dev/null || true

# 8. 移动文档
echo "移动文档..."
mv -n VISUALIZATION_README.md docs/guides/visualization/ 2>/dev/null || true
mv -n 使用说明.md docs/guides/visualization/ 2>/dev/null || true
mv -n 战术分析说明.md docs/guides/visualization/ 2>/dev/null || true
mv -n 工具总结.md docs/guides/visualization/ 2>/dev/null || true

mv -n example_output.txt docs/examples/ 2>/dev/null || true
mv -n example_report.txt docs/examples/ 2>/dev/null || true
mv -n tactical_analysis_example.txt docs/examples/ 2>/dev/null || true
mv -n tictactoe_rules_*.txt docs/examples/ 2>/dev/null || true

mv -n VIPER_LEARNING_GUIDE.md docs/guides/ 2>/dev/null || true
mv -n VIPER_MASKABLE_PPO_GUIDE.md docs/guides/ 2>/dev/null || true
mv -n VIPER_QUICK_REFERENCE.md docs/guides/ 2>/dev/null || true
mv -n VIPER_SAMPLE_TRAJECTORY_GUIDE.md docs/guides/ 2>/dev/null || true
mv -n VIPER_TECHNICAL_ANALYSIS.md docs/guides/ 2>/dev/null || true
mv -n TEST_TREES_README.md docs/guides/ 2>/dev/null || true
mv -n README_PARALLEL_TRAINING.md docs/guides/ 2>/dev/null || true
mv -n RULE_PRIORITY_USAGE.md docs/guides/ 2>/dev/null || true
mv -n SUMMARY_RULE_CHANGES.md docs/guides/ 2>/dev/null || true

echo "✓ 文件整理完成！"
echo ""
echo "现在根目录只保留："
ls -1 | grep -E '(README\.md|\.gitignore|\.clinerules|requirements\.txt|main\.py|\.pdf|reorganize\.sh)'
echo ""
echo "其他文件已按功能分类到相应目录"
EOF

chmod +x reorganize.sh
```

### 执行整理

```bash
# 在服务器的项目根目录执行
./reorganize.sh
```

### 验证整理结果

```bash
# 检查根目录（应该很干净）
ls -la

# 检查分析工具
ls analysis/visualization/
ls analysis/export/
ls analysis/comparison/

# 检查文档
ls docs/guides/
ls docs/examples/
```

## 注意事项

1. **备份**: 整理前先备份或提交到git
   ```bash
   git status
   git add .
   git commit -m "backup: before reorganization"
   ```

2. **测试**: 整理后测试关键脚本是否能正常运行
   ```bash
   python analysis/visualization/visualize_tree_rules.py --help
   ```

3. **更新README**: 整理后更新主README.md，说明新的目录结构

4. **Git追踪**: 使用 `git mv` 而不是 `mv` 可以保留文件历史
   ```bash
   git mv visualize_tree_rules.py analysis/visualization/
   ```

## 完成后的使用方式

```bash
# 可视化
python analysis/visualization/visualize_tree_rules.py --rule 1

# 战术分析
python analysis/visualization/analyze_rules_tactical.py --max-rules 10

# 生成报告
python analysis/visualization/export_rules_report.py --output report.txt

# 或使用快捷脚本
./scripts/visualize.sh --rule 1
./scripts/analyze_tactical.sh --max-rules 10
```

## 清理计划

整理完成后可以删除：
- `FILE_REORGANIZATION_PLAN.md` (本文件)
- `reorganize.sh` (整理脚本)

或者保留在 `docs/` 目录作为记录。
