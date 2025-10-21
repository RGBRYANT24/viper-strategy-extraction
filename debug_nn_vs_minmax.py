"""
调试神经网络后手 vs MinMax先手的详细对战过程
"""
import numpy as np
from battle_nn_vs_tree import (
    TicTacToeBattleEnv,
    NeuralNetPlayer,
    MinMaxPlayer
)

# 创建玩家
print("加载模型...")
minmax_player = MinMaxPlayer(depth_limit=9)
nn_player = NeuralNetPlayer(
    "log/oracle_TicTacToe_selfplay.zip",
    model_type='auto',
    debug=True,
    use_q_masking=True
)

# 创建环境
env = TicTacToeBattleEnv()

# 开始一局游戏：MinMax先手（X），神经网络后手（O）
print("\n" + "="*70)
print("对战：MinMax先手 (X) vs 神经网络后手 (O)")
print("="*70)

obs = env.reset()
player1 = minmax_player  # X
player2 = nn_player       # O

step = 0
while not env.done and step < 9:
    step += 1

    # 当前玩家
    if env.current_player == 1:
        current_agent = player1
        agent_name = "MinMax (X)"
        player_id = 1
    else:
        current_agent = player2
        agent_name = "神经网络 (O)"
        player_id = -1

    print(f"\n{'='*70}")
    print(f"步骤 {step}: {agent_name} 的回合")
    print(f"{'='*70}")
    print(f"当前棋盘状态:")
    env.render()
    print(f"观察向量: {obs}")
    print(f"合法动作: {np.where(obs == 0)[0]}")

    # 获取动作
    if isinstance(current_agent, NeuralNetPlayer):
        print(f"\n调用神经网络predict(obs, player_id={player_id})...")
        action = current_agent.predict(obs, player_id=player_id)
    else:
        action = current_agent.predict(obs)

    print(f"\n{agent_name} 选择动作: {action}")

    # 执行动作
    obs, reward, done, info = env.step(action)

    if done:
        print(f"\n{'='*70}")
        print("游戏结束！")
        print(f"{'='*70}")
        env.render()

        if 'illegal_move' in info:
            print(f"结果: 非法移动！玩家 {info['player']} 输了")
        elif 'winner' in info:
            winner_name = "MinMax (X)" if info['winner'] == 1 else "神经网络 (O)"
            print(f"结果: {winner_name} 获胜！")
        elif 'draw' in info:
            print(f"结果: 平局")

        break

print("\n" + "="*70)
print("分析总结")
print("="*70)
print("如果神经网络总是输，可能的原因：")
print("1. 训练时没有足够的后手经验（selfplay应该有，但可能不平衡）")
print("2. 视角转换有问题（需要验证）")
print("3. Q值选择策略有问题")
print("="*70)
