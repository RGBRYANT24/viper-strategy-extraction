#!/bin/bash
# 使用MinMax对手训练TicTacToe神经网络和决策树

echo "=========================================="
echo "训练方案：使用MinMax对手"
echo "=========================================="

# 设置参数
ENV_NAME="TicTacToe-v0"
TIMESTEPS=100000  # 增加训练步数，因为MinMax更难对付
N_ENV=8
N_ITER=80
SEED=42

echo ""
echo "步骤 1/3: 训练神经网络（Oracle）对战MinMax"
echo "----------------------------------------"
python main.py train-oracle \
    --env-name $ENV_NAME \
    --total-timesteps $TIMESTEPS \
    --n-env $N_ENV \
    --seed $SEED \
    --tictactoe-opponent minmax \
    --tictactoe-minmax-depth 9 \
    --oracle-path log/oracle_${ENV_NAME}_minmax.zip

echo ""
echo "步骤 2/3: 测试神经网络性能"
echo "----------------------------------------"
python main.py test-oracle \
    --env-name $ENV_NAME \
    --n-env 1 \
    --seed $SEED \
    --tictactoe-opponent minmax \
    --oracle-path log/oracle_${ENV_NAME}_minmax.zip

echo ""
echo "步骤 3/3: 使用VIPER提取决策树"
echo "----------------------------------------"
python main.py train-viper \
    --env-name $ENV_NAME \
    --n-iter $N_ITER \
    --n-env $N_ENV \
    --seed $SEED \
    --tictactoe-opponent minmax \
    --oracle-path log/oracle_${ENV_NAME}_minmax.zip

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "神经网络模型: log/oracle_${ENV_NAME}_minmax.zip"
echo "决策树模型: log/viper_${ENV_NAME}_all-leaves_${N_ITER}.joblib"
echo ""
echo "现在可以运行对战测试："
echo "python battle_nn_vs_tree.py --mode all --n-games 100 \\"
echo "    --oracle-path log/oracle_${ENV_NAME}_minmax.zip \\"
echo "    --viper-path log/viper_${ENV_NAME}_all-leaves_${N_ITER}.joblib"
