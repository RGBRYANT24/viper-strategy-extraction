# 快速命令列表

## 上传代码到服务器

```bash
# 在本地执行（替换user@server为你的服务器地址）
# 方法1：上传整个目录
rsync -avz /Users/adrin/Projects/NOGGINGS/VIPER/viper-verifiable-rl-impl/ user@server:~/viper/

# 方法2：只上传修复后的文件（更快）
scp /Users/adrin/Projects/NOGGINGS/VIPER/viper-verifiable-rl-impl/train_viper_improved.py user@server:~/viper/train_viper_improved.py
```

## 在服务器上执行

```bash
# 1. 登录并进入目录
ssh user@server
cd ~/viper
conda activate your_env  # 或 source venv/bin/activate

# 2. 检查Oracle存在
ls -lh log/oracle_TicTacToe_selfplay.zip

# 3. 训练改进版VIPER（推荐配置）
# 配置说明：
# - max-leaves=100: 允许更大的树（原来50太小）
# - max-depth=15: 增加深度限制
# - n-iter=60: 减少迭代次数（80太多，可能过拟合）
# - ccp-alpha=0.0005: 减少正则化（允许更复杂的树）
python train_viper_improved.py \
  --env-name TicTacToe-v0 \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --n-iter 60 \
  --total-timesteps 100000 \
  --max-leaves 100 \
  --max-depth 15 \
  --use-augmentation \
  --exploration-strategy decay \
  --ccp-alpha 0.0005 \
  --min-samples-split 5 \
  --min-samples-leaf 2 \
  --verbose 1

# 4. 对战评估（文件名会根据实际生成的模型调整）
# 先找到最新的viper模型
ls -lt log/viper_*.joblib | head -1

# 然后评估（替换下面的文件名）
python battle_nn_vs_tree.py \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --viper-path log/viper_TicTacToe-v0_all-leaves_100.joblib \
  --mode all \
  --n-games 200

# 5. 导出决策树规则
python export_tree_text.py \
  --tree-path log/viper_TicTacToe-v0_all-leaves_100.joblib
```

## 或者使用一键脚本

```bash
# 在服务器上
cd ~/viper
chmod +x run_improved_viper.sh
./run_improved_viper.sh
```

## 下载结果到本地（在本地执行）

```bash
scp user@server:~/viper/log/viper_*.joblib ./
scp user@server:~/viper/decision_tree.txt ./
```

## 重要修复：视角转换问题

如果出现决策树大量非法移动，需要修复视角转换：

```bash
# 在服务器上，上传修复后的battle_nn_vs_tree.py
# 然后重新测试：
python battle_nn_vs_tree.py \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --viper-path log/viper_TicTacToe-v0_100_15.joblib \
  --mode both \
  --n-games 200
```
