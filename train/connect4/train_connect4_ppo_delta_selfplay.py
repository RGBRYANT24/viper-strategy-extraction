"""
使用 MaskablePPO 训练 connect4 Delta-Uniform Self-Play
支持 action masking, 解决 Q值污染问题
"""

import argparse
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import torch
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym_env
from gym_env.connect4_delta_selfplay import Connect4DeltaSelfPlayEnv
from gym_env.policies import RandomPlayerPolicy, MinMaxPlayerPolicy
from gymnasium import spaces


class PolicySnapshot:
    """
    策略快照包装器
    使用 state_dict 保存策略，避免 deepcopy 问题
    """
    def __init__(self, policy, device='cpu'):
        """
        Args:
            policy: MaskablePPO 的 policy 对象
            device: 运行设备
        """
        self.device = device
        # 保存策略的状态字典（克隆以确保独立性）
        # 对于小模型，可以直接保存在原设备上，但克隆到 CPU 更通用
        self.policy_state_dict = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        # 保存策略的类和参数（用于重建）
        self.policy_class = type(policy)
        self.observation_space = policy.observation_space
        self.action_space = policy.action_space

        # 保存网络架构参数（关键！）
        self.policy_kwargs = {}
        if hasattr(policy, 'features_extractor_class'):
            self.policy_kwargs['features_extractor_class'] = policy.features_extractor_class
        if hasattr(policy, 'features_extractor_kwargs'):
            self.policy_kwargs['features_extractor_kwargs'] = policy.features_extractor_kwargs
        if hasattr(policy, 'net_arch'):
            self.policy_kwargs['net_arch'] = policy.net_arch
        if hasattr(policy, 'activation_fn'):
            self.policy_kwargs['activation_fn'] = policy.activation_fn
        if hasattr(policy, 'ortho_init'):
            self.policy_kwargs['ortho_init'] = policy.ortho_init

        # 创建策略实例并加载状态
        self._policy = None
        self._create_policy()

    def _create_policy(self):
        """创建策略实例并加载状态"""
        # 创建新的策略实例，传递所有保存的参数
        self._policy = self.policy_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=lambda _: 0.0,  # 不需要优化器，所以学习率为0
            **self.policy_kwargs  # 传递网络架构等参数
        )
        # 加载状态字典
        self._policy.load_state_dict(self.policy_state_dict)
        self._policy.to(self.device)
        self._policy.eval()  # 设置为评估模式

    def predict(self, observation, deterministic=False, action_masks=None):
        """
        预测动作（兼容 MaskablePPO 接口）

        Args:
            observation: 观察值
            deterministic: 是否使用确定性策略
            action_masks: 动作掩码 (可选)，shape=(n_actions,), 1=合法, 0=非法

        Returns:
            action: 动作
            state: 状态（用于兼容性，PPO不使用）
        """
        with torch.no_grad():
            # 转化为张量
            if not isinstance(observation, torch.Tensor):
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                obs_tensor = observation.unsqueeze(0).to(self.device)

            # 如果提供了 action_masks，应用 mask
            if action_masks is not None:
                # 将 mask 转换为张量
                if not isinstance(action_masks, torch.Tensor):
                    mask_tensor = torch.as_tensor(action_masks, dtype=torch.bool).unsqueeze(0).to(self.device)
                else:
                    mask_tensor = action_masks.unsqueeze(0).to(self.device)

                # 应用 mask：将非法动作的 logits 设为 -inf
                # 注意：distribution 是 Categorical，需要修改其 logits
                if hasattr(distribution.distribution, 'logits'):
                    logits = distribution.distribution.logits.clone()
                    logits[~mask_tensor] = -float('inf')
                    # 重新创建分布
                    from torch.distributions import Categorical
                    distribution.distribution = Categorical(logits=logits)

            # 采样动作
            if deterministic:
                action = distribution.mode()
            else:
                action = distribution.sample()

            return action.cpu().numpy()[0], None

def mask_fn_connect4(env):
    """
    返回 action mask

    Args:
        env: 环境实例

    Returns:
        mask: numpy array of shape (7,), 1=legal, 0=illegal

    connect4 的 action mask 应该是对于当前列满了才会为 0 其他情况都是 1
    """
    # 获取棋盘状态
    if hasattr(env, 'board'):
        board = env.board
    else:
        # 如果是包装过的环境
        board = env.env.board

    mask = (board[-1] == 0).astype(np.int8)
    return mask
    
