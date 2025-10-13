#!/bin/bash

# 井字棋VIPER训练 - 改进版本
# 解决过拟合/策略不匹配问题

set -e  # 遇到错误立即退出

echo "=================================================="
echo "井字棋 VIPER 改进训练流程"
echo "=================================================="
echo ""

# 配置参数（优化后的推荐配置）
ORACLE_PATH="log/oracle_TicTacToe_selfplay.zip"
ENV_NAME="TicTacToe-v0"
N_ITER=60
TOTAL_TIMESTEPS=100000
MAX_LEAVES=100
MAX_DEPTH=15
EXPLORATION_STRATEGY="decay"
CCP_ALPHA=0.0005
MIN_SAMPLES_SPLIT=5
MIN_SAMPLES_LEAF=2
VERBOSE=1

# 检查Oracle是否存在
if [ ! -f "$ORACLE_PATH" ]; then
    echo "⚠️  错误：找不到Oracle模型 $ORACLE_PATH"
    echo ""
    echo "请先训练selfplay神经网络："
    echo "  python train_selfplay.py --total-timesteps 200000 --output $ORACLE_PATH"
    echo ""
    exit 1
fi

echo "✓ 找到Oracle模型: $ORACLE_PATH"
echo ""

# 显示配置
echo "训练配置："
echo "  环境: $ENV_NAME"
echo "  迭代次数: $N_ITER"
echo "  总时间步: $TOTAL_TIMESTEPS"
echo "  最大叶子数: $MAX_LEAVES"
echo "  最大深度: $MAX_DEPTH"
echo "  探索策略: $EXPLORATION_STRATEGY"
echo "  数据增强: 启用"
echo ""

read -p "按Enter开始训练，或Ctrl+C取消..."
echo ""

# 开始训练
echo "=================================================="
echo "开始训练改进版VIPER..."
echo "=================================================="
echo ""

python train_viper_improved.py \
  --env-name "$ENV_NAME" \
  --oracle-path "$ORACLE_PATH" \
  --n-iter "$N_ITER" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --max-leaves "$MAX_LEAVES" \
  --max-depth "$MAX_DEPTH" \
  --use-augmentation \
  --exploration-strategy "$EXPLORATION_STRATEGY" \
  --ccp-alpha "$CCP_ALPHA" \
  --min-samples-split "$MIN_SAMPLES_SPLIT" \
  --min-samples-leaf "$MIN_SAMPLES_LEAF" \
  --verbose "$VERBOSE"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ 训练失败，退出码: $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "=================================================="
echo "✓ 训练完成！"
echo "=================================================="
echo ""

# 查找生成的模型文件
VIPER_MODEL=$(ls -t log/viper_${ENV_NAME}*.joblib 2>/dev/null | head -1)

if [ -z "$VIPER_MODEL" ]; then
    echo "⚠️  未找到生成的VIPER模型"
    exit 1
fi

echo "生成的模型: $VIPER_MODEL"
echo ""

# 询问是否进行对战测试
read -p "是否进行对战测试？(y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "=================================================="
    echo "开始对战测试..."
    echo "=================================================="
    echo ""

    python battle_nn_vs_tree.py \
      --oracle-path "$ORACLE_PATH" \
      --viper-path "$VIPER_MODEL" \
      --mode all \
      --n-games 200 \
      --seed 42

    BATTLE_EXIT_CODE=$?

    if [ $BATTLE_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "❌ 对战测试失败"
        exit $BATTLE_EXIT_CODE
    fi
fi

echo ""
echo "=================================================="
echo "全部完成！"
echo "=================================================="
echo ""
echo "模型位置: $VIPER_MODEL"
echo ""
echo "下一步操作："
echo "  1. 查看决策树结构："
echo "     python export_tree_text.py --tree-path $VIPER_MODEL"
echo ""
echo "  2. 分析策略："
echo "     python analyze_strategy.py --tree-path $VIPER_MODEL"
echo ""
echo "  3. 单独对战测试："
echo "     python battle_nn_vs_tree.py --viper-path $VIPER_MODEL --mode both --verbose"
echo ""
