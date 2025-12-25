# 立即执行的调试命令

## 问题现状
- `test_tree_perspective.py` 显示两种方法都能得到合法动作
- 但实际对战时决策树作为O有40次非法移动
- 说明问题在特定的游戏状态下

## 立即执行

### 在本地上传调试脚本
```bash
scp /Users/adrin/Projects/NOGGINGS/VIPER/viper-verifiable-rl-impl/debug_illegal_moves.py user@server:~/viper/
```

### 在服务器上运行调试
```bash
ssh user@server
cd ~/viper
conda activate your_env

# 运行详细调试
python debug_illegal_moves.py
```

## 预期输出

脚本会：
1. ✅ 显示第一次非法移动发生时的完整棋盘状态
2. ✅ 显示决策树翻转前后的预测
3. ✅ 显示该位置的实际值
4. ✅ 统计多局游戏中非法移动的模式

## 根据输出判断

### 情况1: 决策树总是预测已被占据的位置
**可能原因**：决策树训练数据有问题，学到了错误的策略

**解决**：重新训练，确保数据质量

### 情况2: 视角翻转没有生效
**可能原因**：`battle_nn_vs_tree.py` 中的调用没有正确传递 `player_id`

**解决**：检查第368行的调用代码

### 情况3: 某些特定棋盘状态导致非法预测
**可能原因**：决策树在训练时没有见过这些状态

**解决**：增加训练数据多样性，重新训练

## 快速验证修复是否生效

如果你已经上传了修复后的 `battle_nn_vs_tree.py`，直接测试：

```bash
# 清除缓存
rm -rf __pycache__
find . -name "*.pyc" -delete

# 小规模测试
python battle_nn_vs_tree.py \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --viper-path log/viper_TicTacToe-v0_100_15.joblib \
  --mode both \
  --n-games 5 \
  --debug

# 查看输出中是否还有非法移动
```

## 关键检查点

运行 `debug_illegal_moves.py` 后，查看输出中的：

```
步骤 X: 决策树 (O) 的回合
当前棋盘:
  ...
合法动作: [...]

决策树预测 (player_id=-1):
  原始棋盘: [...]
  翻转棋盘: [...]
  不翻转预测: X
  翻转后预测: Y
```

- 如果"不翻转预测"是非法的，但"翻转后预测"是合法的 → 视角转换代码正确，但没有被调用
- 如果两个都是非法的 → 决策树模型质量问题
- 如果两个都是合法的但实际仍报非法 → 环境状态不一致

立即运行 `python debug_illegal_moves.py` 并把输出发给我！
