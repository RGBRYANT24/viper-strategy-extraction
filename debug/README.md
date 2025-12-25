# Debug Tools - 调试工具集

这个文件夹包含用于调试和分析神经网络、决策树对战的各种工具。

## 📁 文件列表

| 文件 | 说明 |
|------|------|
| `battle_detailed.py` | 详细对战日志工具，输出每一步的完整信息 |
| `README.md` | 本文件，工具集概览 |
| `BATTLE_DETAILED_GUIDE.md` | battle_detailed.py 快速参考指南 |
| `EXAMPLE_OUTPUT.md` | 输出示例和解读说明 |
| `ILLEGAL_MOVES_ANALYSIS.md` | 非法移动分析功能详细说明 |

**📖 文档快速链接：**
- [快速开始指南](BATTLE_DETAILED_GUIDE.md) - 常用命令和参数速查
- [输出示例](EXAMPLE_OUTPUT.md) - 查看完整输出示例和统计总结
- [非法移动分析](ILLEGAL_MOVES_ANALYSIS.md) - 理解非法移动分析功能（重要！）

---

## 🚀 快速开始

### battle_detailed.py - 详细对战调试

**功能：** 输出每一步的棋盘状态、视角转换、动作预测等详细信息

**最简单的用法：**
```bash
cd /path/to/viper-verifiable-rl-impl
python debug/battle_detailed.py
```

**常用命令：**
```bash
# 调试非法移动
python debug/battle_detailed.py --mode nn-vs-tree

# 测试随机性
python debug/battle_detailed.py --n-games 5 --epsilon 0.2

# 验证确定性
python debug/battle_detailed.py --seed 42
```

**详细文档：** [BATTLE_DETAILED_GUIDE.md](BATTLE_DETAILED_GUIDE.md)

---

## 📚 工具详细介绍

### 1. battle_detailed.py

**用途：** 调试对战过程中的问题，特别是：
- ❌ 非法移动问题
- 🔄 视角转换问题
- 🎯 策略确定性问题
- 🎲 随机探索效果

**输出内容：**
- 每一步的原始棋盘状态
- 视角转换后的输入（如果适用）
- 合法动作列表
- 模型预测的动作
- 动作是否合法
- 游戏结束原因
- **多局统计总结（n_games > 1时自动显示）**
  - 胜负统计和胜率
  - 非法移动统计
  - 智能分析和建议
- **非法移动详细分析（有非法移动时自动显示）**
  - 记录所有非法移动的完整信息
  - **自动判断是否所有非法移动都发生在相同局面**
  - 按局面分组展示详情
  - 智能诊断和修复建议

**参数：**
- `--mode`: 对战模式（nn-vs-tree, tree-vs-nn, both）
- `--n-games`: 对战局数
- `--epsilon`: 随机探索概率 (0.0-1.0)
- `--seed`: 随机种子
- `--oracle-path`: 神经网络模型路径
- `--viper-path`: 决策树模型路径

**使用场景：**

| 场景 | 命令 | 目的 |
|------|------|------|
| 调试第一次非法移动 | `python debug/battle_detailed.py --n-games 1` | 找到问题发生在哪一步 |
| 验证确定性 | `python debug/battle_detailed.py --seed 42` | 检查是否每次都走同样的棋 |
| 测试随机性 | `python debug/battle_detailed.py --epsilon 0.3 --n-games 10` | 看随机探索能否避免问题 |
| 对比先后手 | `python debug/battle_detailed.py --mode both` | 比较作为X和O的表现 |

---

## 🔍 调试典型问题

### 问题1: 决策树总是非法移动

**检查步骤：**
```bash
# 1. 查看详细对局过程
python debug/battle_detailed.py --mode nn-vs-tree --n-games 1

# 2. 重点关注输出中的：
#    - "视角转换: YES/NO"
#    - "转换后数组"
#    - "合法动作"
#    - "预测动作"
```

**可能原因：**
- 视角转换未正确应用
- 决策树在训练时未见过后手场景
- 模型输出的动作索引有误

### 问题2: 每次对局结果都一样

**验证方法：**
```bash
# 使用固定种子运行多次
python debug/battle_detailed.py --seed 42
python debug/battle_detailed.py --seed 42
# 应该得到完全相同的输出
```

**解决方案：**
```bash
# 加入随机探索
python debug/battle_detailed.py --epsilon 0.2 --n-games 10
```

### 问题3: 神经网络表现异常

**对比测试：**
```bash
# 测试先手
python debug/battle_detailed.py --mode nn-vs-tree

# 测试后手
python debug/battle_detailed.py --mode tree-vs-nn

# 对比两次的"视角转换"和"转换后数组"是否正确
```

---

## 📊 输出示例解读

```
当前玩家: 决策树 (O) (player_id=-1)
当前棋盘数组: [ 0.  0.  0.  0.  1.  0.  0.  0.  0.]   ← 环境视角的棋盘
合法动作: [0 1 2 3 5 6 7 8]                         ← 可以落子的位置
视角转换: YES (因为是O玩家)                          ← 是否进行了视角转换
转换后数组: [-0. -0. -0. -0. -1. -0. -0. -0. -0.]   ← 模型实际看到的输入
[决策树] 预测动作: 4                                ← 模型的预测
动作 4 是否合法: False                              ← 检查结果（❌非法）
⚠️  非法移动！位置 4 已被占用或超出范围              ← 错误提示
```

**关键检查点：**
1. ✅ `视角转换` 是否符合预期（O玩家应该是YES）
2. ✅ `转换后数组` 的值是否正确（应该翻转符号）
3. ✅ `预测动作` 是否在 `合法动作` 列表中
4. ✅ 如果非法，查看该位置当前是什么状态

---

## 💡 使用技巧

### 1. 保存输出到文件
```bash
python debug/battle_detailed.py --mode both --n-games 5 > output.log 2>&1
```

### 2. 搜索特定信息
```bash
# 查找所有非法移动
grep "非法移动" output.log

# 查找视角转换
grep "视角转换" output.log

# 统计非法移动次数
grep -c "非法移动" output.log
```

### 3. 批量测试
```bash
# 运行100局并统计
for i in {1..100}; do
  python debug/battle_detailed.py --seed $i --n-games 1
done | grep "非法移动" | wc -l
```

---

## 🔗 相关文件

- `../battle_nn_vs_tree.py` - 批量对战工具（统计信息，无详细日志）
- `../gym_env/tictactoe.py` - 井字棋环境实现
- `../gym_env/tictactoe_selfplay.py` - 自我对弈环境

---

## 📝 贡献指南

添加新的调试工具时：

1. 在此文件夹创建新的 `.py` 文件
2. 在文件顶部添加详细的文档字符串
3. 更新本 README.md 添加工具说明
4. 如果工具较复杂，创建对应的使用指南文档

**命名规范：**
- 工具脚本: `<功能>_<描述>.py`（如 `battle_detailed.py`）
- 使用指南: `<工具名大写>_GUIDE.md`（如 `BATTLE_DETAILED_GUIDE.md`）
