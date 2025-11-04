"""
课程学习训练：在 Delta-Uniform Self-play 基础上混入关键决策场景

策略：
1. 保持原有的 Delta-Uniform Self-play 不变
2. 以一定概率从关键决策场景开始
3. 使用原环境的奖励机制（+1/-1/0/-10）
4. 支持加载已有模型继续训练
5. 逐步调整关键场景比例（课程）
"""

import argparse
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces
import torch
from collections import deque
import sys
import os
from datetime import datetime
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym_env
from gym_env.tictactoe_delta_selfplay import TicTacToeDeltaSelfPlayEnv
from gym_env.policies import RandomPlayerPolicy, MinMaxPlayerPolicy
from train.train_delta_selfplay_ppo import PolicySnapshot, mask_fn


class WeightedSelfPlayEnv(TicTacToeDeltaSelfPlayEnv):
    """支持加权采样的自我对弈环境"""
    def __init__(self, random_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.random_weight = random_weight

    def _sample_opponent(self):
        """加权采样对手"""
        all_opponents = []
        weights = []

        # 基准策略池（Random 有更高权重）
        for policy in self.baseline_pool:
            all_opponents.append(policy)
            if isinstance(policy, RandomPlayerPolicy):
                weights.append(self.random_weight)  # Random 对手权重更高
            else:
                weights.append(1.0)

        # 学习策略池（正常权重）
        if self.learned_pool is not None and len(self.learned_pool) > 0:
            for policy in list(self.learned_pool):
                all_opponents.append(policy)
                weights.append(1.0)

        if len(all_opponents) == 0:
            return None

        # 归一化权重并采样
        weights = np.array(weights)
        probs = weights / weights.sum()
        idx = np.random.choice(len(all_opponents), p=probs)

        return all_opponents[idx]


class CriticalScenarioGenerator:
    """
    生成所有 Win/Lose 关键决策场景
    使用枚举方法，与 exhaustive_win_lose_test.py 逻辑一致
    """

    WIN_COMBINATIONS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
        [0, 4, 8], [2, 4, 6]              # 对角线
    ]

    def __init__(self):
        self.scenarios = self._generate_all_scenarios()
        print(f"[CriticalScenarioGenerator] 生成了 {len(self.scenarios)} 个关键场景")

    def _generate_all_scenarios(self):
        """
        枚举所有 3^9 种棋盘，筛选出合法的关键场景
        与 exhaustive_win_lose_test.py 逻辑一致
        """
        from itertools import product

        scenarios = []

        # 枚举所有棋盘
        for config in product([-1, 0, 1], repeat=9):
            board = np.array(config, dtype=np.float32)

            num_x = int(np.sum(board == 1))
            num_o = int(np.sum(board == -1))

            # 基本合法性检查
            if num_x < num_o - 1 or num_x > num_o + 1:
                continue

            # 检查是否已经有人获胜
            if self._check_winner(board, 1) or self._check_winner(board, -1):
                continue

            # 我们需要找"现在轮到X下"的局面
            x_to_move = (num_x == num_o) or (num_x == num_o - 1)

            if not x_to_move:
                continue

            # 检查X是否即将获胜
            x_winning_moves = self._is_one_move_to_win(board, 1)

            # 检查O是否即将获胜（X需要防守）
            o_winning_moves = self._is_one_move_to_win(board, -1)

            # 收集获胜场景
            if len(x_winning_moves) > 0:
                scenarios.append({
                    'board': board.copy(),
                    'type': 'win'
                })

            # 收集防守场景
            if len(o_winning_moves) > 0:
                scenarios.append({
                    'board': board.copy(),
                    'type': 'defend'
                })

        return scenarios

    def _is_one_move_to_win(self, board, player):
        """
        检查某个玩家是否只差一步就能获胜
        返回所有可能的获胜位置列表
        """
        results = []
        for combo in self.WIN_COMBINATIONS:
            player_count = sum(1 for pos in combo if board[pos] == player)
            empty_count = sum(1 for pos in combo if board[pos] == 0)

            # 该组合中有2个该玩家的棋子，且有1个空位
            if player_count == 2 and empty_count == 1:
                empty_pos = [pos for pos in combo if board[pos] == 0][0]
                results.append(empty_pos)

        return results

    def _check_winner(self, board, player):
        """检查某个玩家是否获胜"""
        for combo in self.WIN_COMBINATIONS:
            if all(board[pos] == player for pos in combo):
                return True
        return False

    def get_random_scenario(self):
        """随机获取一个场景"""
        idx = np.random.randint(len(self.scenarios))
        return self.scenarios[idx]


class CurriculumSelfPlayEnv(WeightedSelfPlayEnv):
    """
    课程学习混合环境：继承 WeightedSelfPlayEnv

    以一定概率从关键场景开始，然后继续正常的 Self-play
    保持原有的奖励机制和对手采样策略
    """

    def __init__(self,
                 baseline_pool,
                 learned_pool,
                 scenario_generator,
                 critical_prob=0.3,
                 play_as_o_prob=0.5,
                 random_weight=2.0,
                 sampling_strategy='uniform'):
        """
        Args:
            baseline_pool: 基准策略池
            learned_pool: 学习策略池
            scenario_generator: 关键场景生成器
            critical_prob: 从关键场景开始的概率（0-1）
            play_as_o_prob: 作为O方的概率
            random_weight: Random对手权重
            sampling_strategy: 采样策略
        """
        # 调用父类初始化（WeightedSelfPlayEnv）
        super().__init__(
            baseline_pool=baseline_pool,
            learned_pool=learned_pool,
            play_as_o_prob=play_as_o_prob,
            sampling_strategy=sampling_strategy,
            random_weight=random_weight
        )

        self.scenario_generator = scenario_generator
        self.critical_prob = critical_prob

        # 统计
        self.critical_episodes = 0
        self.normal_episodes = 0

    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)

        # 决定是否从关键场景开始
        if np.random.random() < self.critical_prob:
            # 从关键场景开始
            self.critical_episodes += 1
            return self._reset_from_critical_scenario(seed, options)
        else:
            # 正常 Self-play
            self.normal_episodes += 1
            return super().reset(seed=seed, options=options)

    def _reset_from_critical_scenario(self, seed=None, options=None):
        """从关键场景开始"""
        # 获取随机关键场景
        scenario = self.scenario_generator.get_random_scenario()

        # 设置棋盘为关键场景
        self.board = scenario['board'].copy()
        self.done = False

        # 选择对手（后续步骤会用到）
        self.opponent = self._sample_opponent()

        # 决定玩家角色
        self.play_as_o = (np.random.random() < self.play_as_o_prob)

        if self.play_as_o:
            # 如果玩家是 O，让对手先走（但从当前场景开始，不需要）
            # 由于场景已经设置好，直接返回即可
            # 需要注意的是场景是为 X 设计的，所以如果是 O 则需要反转
            self.board = -self.board.copy()

        return self.board.copy(), {}


class CurriculumCallback(BaseCallback):
    """课程学习回调：逐步调整关键场景比例"""

    def __init__(self, envs, curriculum_schedule, verbose=1):
        """
        Args:
            envs: 环境列表
            curriculum_schedule: [(steps, critical_prob), ...]
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.envs = envs
        self.curriculum_schedule = sorted(curriculum_schedule, key=lambda x: x[0])
        self.current_stage = 0

    def _on_step(self) -> bool:
        # 检查是否需要调整课程
        if self.current_stage < len(self.curriculum_schedule):
            target_steps, target_prob = self.curriculum_schedule[self.current_stage]

            if self.num_timesteps >= target_steps:
                # 更新所有环境的关键场景比例
                for env_idx in range(self.envs.num_envs):
                    env = self.envs.envs[env_idx]
                    # 获取实际的环境（可能被包装）
                    actual_env = env
                    while hasattr(actual_env, 'env'):
                        actual_env = actual_env.env

                    if hasattr(actual_env, 'critical_prob'):
                        actual_env.critical_prob = target_prob

                if self.verbose > 0:
                    print(f"\n{'='*70}")
                    print(f"[课程学习] 第 {self.num_timesteps} 步")
                    print(f"调整关键场景比例: {target_prob*100:.0f}%")
                    print(f"{'='*70}\n")

                self.current_stage += 1

        return True


def main():
    parser = argparse.ArgumentParser(description='课程学习：混合 Self-play + 关键决策场景')

    # 模型参数
    parser.add_argument("--model", type=str, default=None,
                       help="要加载的模型路径（不指定则从零开始训练）")

    # 训练参数
    parser.add_argument("--total-timesteps", type=int, default=150000,
                       help="总训练步数")
    parser.add_argument("--n-env", type=int, default=8)
    parser.add_argument("--update-interval", type=int, default=10000)
    parser.add_argument("--max-pool-size", type=int, default=20)

    # 课程学习参数（关键！）
    parser.add_argument("--initial-critical-prob", type=float, default=0.2,
                       help="初始关键场景比例（推荐0.2-0.3）")
    parser.add_argument("--max-critical-prob", type=float, default=0.5,
                       help="最大关键场景比例（推荐0.5-0.7，更激进用0.7）")
    parser.add_argument("--final-critical-prob", type=float, default=None,
                       help="最终关键场景比例（默认回到initial值，可设置为0.3-0.4保持较高）")

    # PPO 参数
    parser.add_argument("--ent-coef", type=float, default=0.05,
                       help="熵系数（保持原有值或略低）")
    parser.add_argument("--play-as-o-prob", type=float, default=0.5)
    parser.add_argument("--random-weight", type=float, default=2.0)
    parser.add_argument("--use-minmax", action="store_true")

    # 输出
    parser.add_argument("--output-dir", type=str, default="log")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()

    # 设置默认 final_critical_prob
    if args.final_critical_prob is None:
        args.final_critical_prob = args.initial_critical_prob

    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.model:
        model_basename = os.path.basename(args.model).replace('.zip', '')
        output_filename = f"{model_basename}_curriculum_cr{int(args.max_critical_prob*100)}_steps{args.total_timesteps//1000}k_{timestamp}.zip"
    else:
        output_filename = f"oracle_TicTacToe_ppo_curriculum_cr{int(args.max_critical_prob*100)}_steps{args.total_timesteps//1000}k_{timestamp}.zip"
    output_path = os.path.join(args.output_dir, output_filename)

    print("=" * 70)
    print("课程学习训练：混合 Self-play + 关键决策场景")
    print("=" * 70)
    if args.model:
        print(f"加载模型: {args.model}")
    else:
        print(f"模式: 从零开始训练")
    print(f"输出路径: {output_path}")
    print(f"总步数: {args.total_timesteps}")
    print(f"熵系数: {args.ent_coef}")
    print(f"关键场景比例: {args.initial_critical_prob*100:.0f}% → {args.max_critical_prob*100:.0f}% → {args.final_critical_prob*100:.0f}%")
    print(f"✓ 保持 Delta-Uniform Self-play 策略")
    print()

    # 创建或加载模型
    if args.model:
        # 加载已有模型
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"模型文件不存在: {args.model}")

        print("加载模型...")
        model = MaskablePPO.load(args.model)
        print("✓ 模型加载成功")

        # 调整参数
        original_ent_coef = model.ent_coef
        model.ent_coef = args.ent_coef
        print(f"✓ 熵系数: {original_ent_coef} → {args.ent_coef}")
    else:
        # 从零创建新模型（先创建临时环境）
        print("从零创建新模型...")
        model = None  # 稍后在创建环境后再创建
    print()

    # 初始化策略池
    temp_env = TicTacToeDeltaSelfPlayEnv()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    baseline_policies = [RandomPlayerPolicy(obs_space, act_space)]
    if args.use_minmax:
        baseline_policies.append(MinMaxPlayerPolicy(obs_space, act_space))

    learned_policy_pool = deque(maxlen=args.max_pool_size)

    # 将当前模型加入策略池（如果有）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is not None:
        initial_snapshot = PolicySnapshot(model.policy, device=device)
        learned_policy_pool.append(initial_snapshot)

    print(f"策略池初始化完成")
    print(f"  基准策略: {len(baseline_policies)} 个")
    print(f"  学习策略: {len(learned_policy_pool)} 个")
    print()

    # 创建关键场景生成器
    print("生成关键决策场景...")
    scenario_generator = CriticalScenarioGenerator()
    print()

    # 创建课程学习环境
    def make_curriculum_env():
        env = CurriculumSelfPlayEnv(
            baseline_pool=baseline_policies,
            learned_pool=learned_policy_pool,
            scenario_generator=scenario_generator,
            critical_prob=args.initial_critical_prob,
            play_as_o_prob=args.play_as_o_prob,
            random_weight=args.random_weight,
            sampling_strategy='uniform'
        )
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env

    envs = DummyVecEnv([make_curriculum_env for _ in range(args.n_env)])

    # 如果没有加载模型，现在创建
    if model is None:
        print("创建新的 MaskablePPO 模型...")
        model = MaskablePPO(
            policy='MlpPolicy',
            env=envs,
            learning_rate=1e-3,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            policy_kwargs={'net_arch': [128, 128]},
            verbose=args.verbose,
            seed=args.seed
        )
        print("✓ 新模型创建成功")
    else:
        model.set_env(envs)

    print(f"✓ 创建了 {args.n_env} 个课程学习环境\n")

    # 定义课程计划（更激进、更持久）
    curriculum_schedule = [
        (0, args.initial_critical_prob),  # 开始
        (args.total_timesteps // 6, args.max_critical_prob * 0.7),  # 17%
        (args.total_timesteps // 3, args.max_critical_prob),  # 33%: 达到最高
        (args.total_timesteps * 2 // 3, args.max_critical_prob),  # 67%: 保持最高
        (args.total_timesteps * 5 // 6, args.final_critical_prob),  # 83%: 回归到final值
    ]

    print("课程计划:")
    for steps, prob in curriculum_schedule:
        progress = steps / args.total_timesteps * 100
        print(f"  {steps:7d} 步 ({progress:5.1f}%): 关键场景 {prob*100:5.0f}%")
    print()

    if args.final_critical_prob > args.initial_critical_prob:
        print(f"注意：最终比例({args.final_critical_prob*100:.0f}%)高于初始比例，保持较强的关键场景训练")
    elif args.final_critical_prob == args.initial_critical_prob:
        print(f"注意：最终回归到初始比例({args.final_critical_prob*100:.0f}%)，平衡训练")
    print()

    curriculum_callback = CurriculumCallback(
        envs=envs,
        curriculum_schedule=curriculum_schedule,
        verbose=args.verbose
    )

    # 开始训练
    print("=" * 70)
    print("开始课程学习训练")
    print("=" * 70)
    print()

    steps_trained = 0
    update_count = 0

    while steps_trained < args.total_timesteps:
        steps_this_round = min(args.update_interval, args.total_timesteps - steps_trained)

        print(f"[轮次 {update_count + 1}] 训练 {steps_this_round} 步...")
        model.learn(
            total_timesteps=steps_this_round,
            reset_num_timesteps=False,
            callback=curriculum_callback,
            log_interval=100
        )

        steps_trained += steps_this_round
        update_count += 1

        print(f"\n[更新] 已训练 {steps_trained}/{args.total_timesteps} 步")

        # 更新策略池
        current_snapshot = PolicySnapshot(model.policy, device=device)
        learned_policy_pool.append(current_snapshot)
        print(f"策略池大小: {len(learned_policy_pool)}/{args.max_pool_size}\n")

    # 保存模型
    os.makedirs(args.output_dir, exist_ok=True)
    model.save(output_path)

    print("\n" + "=" * 70)
    print("课程学习训练完成！")
    print("=" * 70)
    print(f"模型已保存: {output_path}")
    print()
    print("建议测试:")
    print(f"python evaluation/exhaustive_win_lose_test.py --model {output_path}")
    print()


if __name__ == "__main__":
    main()
