import numpy as np
import gymnasium as gym
import gym_env

print("=" * 60)
print("测试1: 基本游戏流程（先手）")
print("=" * 60)

env = gym.make('TicTacToe-v0', opponent_type='minmax', play_as_o_prob=1.0)
obs, info = env.reset()

# print(f"玩家角色: {'O (后手)' if env.unwrapped.play_as_o else 'X (先手)'}")
print(f"初始观察:\n{obs.reshape(3, 3)}")
print(f"物理棋盘:\n{env.unwrapped.board.reshape(3, 3)}")
print()

# # 玩一步
# legal_actions = np.where(obs == 0)[0]
# action = legal_actions[0]
# print(f"玩家选择动作: {action}")

# obs, reward, done, truncated, info = env.step(action)

done = False

while not done:
    player_action = input("input action")
    player_action = int(player_action)
    obs, reward, done, truncated, info = env.step(player_action)
    print(f"执行后观察:\n{obs.reshape(3, 3)}")
    print(f"物理棋盘:\n{env.unwrapped.board.reshape(3, 3)}")
    print(f"奖励: {reward}, 游戏结束: {done}")
    if done:
        break

# print(f"执行后观察:\n{obs.reshape(3, 3)}")
# print(f"物理棋盘:\n{env.board.reshape(3, 3)}")
# print(f"奖励: {reward}, 游戏结束: {done}")

# print("\n" + "=" * 60)
# print("测试2: 后手测试")
# print("=" * 60)

# env2 = gym.make('TicTacToe-v0', opponent_type='random', play_as_o_prob=1.0)
# obs2, info2 = env2.reset()

# print(f"玩家角色: {'O (后手)' if env2.play_as_o else 'X (先手)'}")
# print(f"初始观察:\n{obs2.reshape(3, 3)}")
# print(f"物理棋盘:\n{env2.board.reshape(3, 3)}")
# print()

# # 验证：后手时，对手应该先下了一步
# assert np.sum(np.abs(env2.board)) == 1, "后手时对手应该下了一步"
# assert np.sum(np.abs(obs2)) == 1, "观察中应该看到对手的一步"
# assert np.any(obs2 == -1), "观察中对手应该是-1"

# print("✓ 后手初始化正确")

# print("\n" + "=" * 60)
# print("测试3: 视角转换验证")
# print("=" * 60)

# # 创建一个固定的棋盘状态
# env3 = gym.make('TicTacToe-v0', opponent_type='random', play_as_o_prob=0.0)
# env3.reset()
# env3.board = np.array([1, 0, -1, 0, 1, 0, 0, 0, -1], dtype=np.float32)
# env3.play_as_o = False  # 先手

# obs_x = env3._get_observation()
# print("玩家是X（先手）时:")
# print(f"  物理棋盘:\n{env3.board.reshape(3, 3)}")
# print(f"  观察:\n{obs_x.reshape(3, 3)}")
# assert np.array_equal(obs_x, env3.board), "先手时观察应该等于物理棋盘"
# print("  ✓ 先手视角正确")

# env3.play_as_o = True  # 后手
# obs_o = env3._get_observation()
# print("\n玩家是O（后手）时:")
# print(f"  物理棋盘:\n{env3.board.reshape(3, 3)}")
# print(f"  观察:\n{obs_o.reshape(3, 3)}")
# assert np.array_equal(obs_o, -env3.board), "后手时观察应该是物理棋盘的相反"
# print("  ✓ 后手视角正确")

# print("\n" + "=" * 60)
# print("✅ 所有基本测试通过！")
# print("=" * 60)