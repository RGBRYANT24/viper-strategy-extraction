#!/bin/bash

# VIPER from MaskablePPO - 快速运行脚本
# 使用训练好的 PPO 模型提取决策树

set -e  # 遇到错误立即退出

echo "=========================================="
echo "VIPER from MaskablePPO - Quick Start"
echo "=========================================="
echo ""

# 配置参数
ORACLE_PATH="${1:-log/oracle_TicTacToe_ppo_aggressive.zip}"
OUTPUT_PATH="${2:-log/viper_TicTacToe_from_ppo.joblib}"
TOTAL_TIMESTEPS="${3:-50000}"
N_ITER="${4:-10}"
MAX_DEPTH="${5:-10}"
MAX_LEAVES="${6:-50}"

echo "配置:"
echo "  Oracle:     $ORACLE_PATH"
echo "  Output:     $OUTPUT_PATH"
echo "  Timesteps:  $TOTAL_TIMESTEPS"
echo "  Iterations: $N_ITER"
echo "  Max Depth:  $MAX_DEPTH"
echo "  Max Leaves: $MAX_LEAVES"
echo ""

# 检查 Oracle 是否存在
if [ ! -f "$ORACLE_PATH" ]; then
    echo "❌ Oracle 模型不存在: $ORACLE_PATH"
    echo ""
    echo "可用的 PPO 模型:"
    ls -lh log/oracle_TicTacToe_ppo*.zip 2>/dev/null || echo "  (没有找到)"
    echo ""
    echo "训练新的 Oracle:"
    echo "  python train/train_delta_selfplay_ppo.py --total-timesteps 200000 --output $ORACLE_PATH"
    echo ""
    exit 1
fi

echo "✓ Oracle 模型存在"
echo ""

# 步骤 1: 训练 VIPER 决策树
echo "=========================================="
echo "步骤 1: 训练 VIPER 决策树"
echo "=========================================="
echo ""

python train/viper_maskable_ppo.py \
    --oracle-path "$ORACLE_PATH" \
    --output "$OUTPUT_PATH" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --n-iter "$N_ITER" \
    --max-depth "$MAX_DEPTH" \
    --max-leaves "$MAX_LEAVES" \
    --opponent-type minmax \
    --test \
    --verbose 1

if [ $? -ne 0 ]; then
    echo "❌ VIPER 训练失败"
    exit 1
fi

echo ""
echo "✓ VIPER 训练完成"
echo ""

# 步骤 2: 详细评估
echo "=========================================="
echo "步骤 2: 详细评估"
echo "=========================================="
echo ""

RULES_PATH="${OUTPUT_PATH%.joblib}_rules.txt"

python evaluation/evaluate_viper_tree.py \
    --model-path "$OUTPUT_PATH" \
    --opponent both \
    --n-episodes 100 \
    --export-rules "$RULES_PATH" \
    --visualize

if [ $? -ne 0 ]; then
    echo "❌ 评估失败"
    exit 1
fi

echo ""
echo "✓ 评估完成"
echo ""

# 总结
echo "=========================================="
echo "完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  决策树模型: $OUTPUT_PATH"
echo "  决策规则:   $RULES_PATH"
echo ""
echo "查看规则:"
echo "  cat $RULES_PATH"
echo ""
echo "重新评估:"
echo "  python evaluation/evaluate_viper_tree.py --model-path $OUTPUT_PATH --opponent both"
echo ""
