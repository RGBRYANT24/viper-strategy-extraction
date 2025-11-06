import numpy as np
import torch
import gymnasium as gym
from sklearn.tree import DecisionTreeClassifier
from sb3_contrib import MaskablePPO
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env.tictactoe import TicTacToeEnv

def test_env():
    env = gym.make('TicTacToe-v0', opponent_type='random')
    obs,_ = env.reset()
    print('Initial Observation:\n', obs.reshape(3,-1))

    action = 4
    obs, reward, done, truncated, info = env.step(action)
    print(f'\nAfter taking action {action}:')
    print('Observation:\n', obs.reshape(3,-1))
    print('Reward:', reward)
    print('Done:', done)
    print('Truncated:', truncated)
    print('Info:', info)
    return env, obs


def load_oracle(oracle_path):
    env = gym.make('TicTacToe-v0', opponent_type='random')
    oracle = MaskablePPO.load(oracle_path, env=env)
    obs, _ = env.reset()
    mask = (obs == 0).astype(bool)
    print('mask', mask)
    mask_tensor = torch.tensor(mask).unsqueeze(0)
    print('mask_tensor', mask_tensor)
    action, _ = oracle.predict(obs,deterministic=True, action_masks=mask_tensor)
    print('Predicted action by oracle:', action)
    return oracle

def sample_trajectory(orcale, env, n_steps, use_oracle=True):
    """采样 n_steps 个样本

    Returns:
        dataset: List of (observation, action, weight)
    """
    pass


def compute_criticality(env, model, observation):
    if isinstance(model, MaskablePPO):
        # For policy gradient methods we use the max entropy formulation
        # to get Q(s, a) \approx log pi(a|s)
        # See Ziebart et al. 2008
        assert isinstance(env.action_space,
                          gym.spaces.Discrete), "Only discrete action spaces supported for loss function"
        possible_actions = np.arange(env.action_space.n)
        mask = (observation == 0).astype(bool)
        mask_tensor = torch.tensor(mask).unsqueeze(0)
        # print('mask_tensor', mask_tensor)
        possible_actions = possible_actions[mask]
        # print('possible_actions after masking', possible_actions)
        obs_tensor = torch.as_tensor(observation).unsqueeze(0).to(model.device)
        # print('obs_tensor', obs_tensor)

        # MaskablePPO's evaluate_actions expects:
        # - obs: torch.Tensor (not a dict)
        # - actions: torch.Tensor
        # - action_masks: Optional[torch.Tensor] (as third parameter)

        # print("--- 检查张量 ---")
        # print("观测张量:", obs_tensor)
        # print("Mask 张量:", mask_tensor)
        # print("-----------------")

        log_probs = []
        for action in possible_actions:
            action_tensor = torch.tensor([action]).to(model.device)
            # print('action_tensor', action_tensor)
            _, log_prob, _ = model.policy.evaluate_actions(obs_tensor, action_tensor, action_masks=mask_tensor)
            log_probs.append(log_prob.detach().cpu().numpy().flatten())
            # print('log_prob for action', action, ':', log_prob.detach().cpu().numpy().flatten())

        log_probs = np.array(log_probs).T
        # print('log_probs array:', log_probs)
        # print('Max log prob:', log_probs.max(axis=1))
        # print('Min log prob:', log_probs.min(axis=1))
        # print('Criticality (max - min):', log_probs.max(axis=1) - log_probs.min(axis=1))
        return log_probs.max(axis=1) - log_probs.min(axis=1)
    
    # with torch.no_grad():
        #     # extract_features 只需要观察张量
        #     features = model.policy.extract_features(obs_tensor)
            
        #     # 获取潜在向量
        #     latent_pi, latent_vf = model.policy.mlp_extractor(features)
            
        #     # 获取动作分布
        #     distribution = model.policy._get_action_dist_from_latent(latent_pi)

        #     logits = model.policy.action_net(latent_pi)  # [batch, num_actions]
        #     print("原始 logits:", logits)

        #     # 应用掩码：将不可行的动作设置为极小值
        #     masked_logits = logits.clone()
        #     masked_logits[~mask_tensor] = -1e8
        #     print("掩码后的 logits:", masked_logits)

        #     log_probs = torch.log_softmax(masked_logits, dim=-1)
        #     print("掩码后的 log 概率:", log_probs)

        #     max_log_prob = torch.max(log_probs, dim=1)[0]
        #     print("最大 log 概率:", max_log_prob.cpu().numpy())
            
            # 获取所有动作的概率
            # all_probs = distribution.distribution.probs
            # print("所有动作的 (Masked) 概率:", all_probs.cpu().numpy())

        # log_probs = []
        # for action in possible_actions:
        #     action_te

        # with torch.no_grad():
        #     # 将观察转换为张量
        #     if not isinstance(observation, torch.Tensor):
        #         obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        #     else:
        #         obs_tensor = observation.unsqueeze(0).to(self.device)
        
    

    

def test_oracle():
    env = gym.make('TicTacToe-v0', opponent_type='random')
    oracle = MaskablePPO.load("log/oracle_TicTacToe_ppo_aggressive.zip", env=env)
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    while not done:
        mask = (obs == 0).astype(bool)
        print('mask', mask, 'mask_shape', mask.shape)

        action,_ = oracle.predict(obs, deterministic=True, action_masks=torch.tensor(mask).unsqueeze(0))

        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        print('truncated', truncated)
        done = done or truncated

        print('Action taken:', action, 'reward:', reward)

    print('Episode reward:', episode_reward)
    env.close()


def get_oracle_action_logits(oracle, obs):
    """获取原始logits（不要mask！）"""
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_tensor = obs_tensor.to(oracle.device)

        distribution = oracle.policy.get_distribution(obs_tensor)
        logits = distribution.distribution.logits.cpu().numpy()[0]

        # ✅ 直接返回原始logits
        return logits

def test_samsample_trajectoryple(oracle, policy, env, n_steps, beta=0.5, use_criticality=True):
    env = gym.make('TicTacToe-v0', opponent_type='minimax')

    dataset = []
    n_steps = 10
    trajectory = []

    obs, _ = env.reset()
    policy = policy or oracle

    while len(trajectory) < n_steps:
        print('Current observation:\n', obs.reshape(3,-1))

        # mixed strategy
        use_oracle_for_action = np.random.binomial(1, beta) == 1
        # use_oracle_for_action = True
        active_policy = oracle if use_oracle_for_action else policy
        action = None

        # mask
        mask = (obs == 0).astype(bool)

        # 获取神经网络的输出 logits
        oracle_logits = get_oracle_action_logits(oracle, obs)
        # oracle_logits = logits.clone()
        print('oracle_logits', oracle_logits)
        masked_logits = np.full(9, -np.inf)

        # 应用掩码：将不可行的动作设置为极小值

        if isinstance(active_policy, MaskablePPO):
            # MaskablePPO
            masked_logits[mask] = oracle_logits[mask]
            action = np.argmax(masked_logits)
            # print('Action chosen by MaskablePPO:', action)
        else:
            # Decision Tree
            logits = active_policy.predict(obs.reshape(1,-1))[0]
            # 选择最高logit的合法动作
            masked_logits[mask] = logits[mask]
            action = np.argmax(masked_logits)

        # 采取行动并获取下一个状态
        # 使用active_policy采取行动 不是oracle
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result

        state_loss = compute_criticality(env, oracle, obs)

        # # 将 logits tensor 转换为 numpy 数组
        # oracle_logits = logits.squeeze().detach().cpu().numpy()  # shape: (num_actions,)
        # 提取权重标量
        weight = state_loss[0] if isinstance(state_loss, np.ndarray) else state_loss

        print('observation:', obs)
        print('oracle_logits shape:', oracle_logits.shape, 'values:', oracle_logits)
        print('weight:', weight)
        print('trajectory this time:', (obs.copy(), oracle_logits, weight))

        # 存储 (观察向量, logits向量, 权重标量) 用于回归树训练
        trajectory.append((obs.copy(), oracle_logits, weight))

        obs = next_obs

        if done:
            obs, _ = env.reset()

    print('Sampled trajectory:', trajectory)

    return trajectory







if __name__ == "__main__":
    env, obs = test_env()
    model = load_oracle("log/oracle_TicTacToe_ppo_aggressive.zip")
    # compute_criticality(env, model, obs)
    # test_oracle()
    test_samsample_trajectoryple(model, policy=[], env=env, n_steps=5, beta=1.0, use_criticality=True)
