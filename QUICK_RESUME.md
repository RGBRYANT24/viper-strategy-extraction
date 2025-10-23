# 快速恢复工作

## 一句话总结
使用 MaskablePPO 替代 MaskedDQN，因为 DQN 的 ε-greedy exploration 会绕过 mask 导致非法移动。

## 下次开始的命令

```bash
# 给 Claude Code 的 Prompt
我需要继续 TicTacToe Action Masking 训练工作。
已验证 MaskablePPO 与 VIPER 兼容，需要：
1. 修改 train/oracle.py 添加 MaskablePPO 配置
2. 创建测试脚本 test_maskable_ppo_viper.py
详见 MASKED_TRAINING_GUIDE.md
```

## 当前问题
- DQN 训练时 `ep_rew_mean = -6.18`（有 -10 惩罚）
- 原因：ε-greedy 的 `action_space.sample()` 绕过了 MaskedDQNPolicy

## 解决方案
使用 MaskablePPO（已验证兼容 VIPER）

## 下一步
1. 修改 `train/oracle.py` 第75行后添加配置
2. 创建 `test_maskable_ppo_viper.py`
3. 运行训练验证

## 文档
- 完整指南: [MASKED_TRAINING_GUIDE.md](MASKED_TRAINING_GUIDE.md)
- 快速开始: [QUICKSTART_MASKED_DQN.md](QUICKSTART_MASKED_DQN.md)（需更新为PPO）
