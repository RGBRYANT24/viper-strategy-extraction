import numpy as np
import torch
import gymnasium as gym
from sklearn.tree import DecisionTreeClassifier
from sb3_contrib import MaskablePPO
import joblib
import sys
import os
from stable_baselines3.common.utils import obs_as_tensor

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
        print('mask_tensor', mask_tensor)
        possible_actions = possible_actions[mask]
        print('possible_actions after masking', possible_actions)
        obs_tensor = torch.as_tensor(observation).unsqueeze(0).to(model.device)
        print('obs_tensor', obs_tensor)

        obs_dict = {
        "observation": obs_tensor,
        "action_mask": mask_tensor
        }

        # 2. 把这个字典转换成模型需要的张量
        # obs_as_tensor 会自动处理字典、批次和设备
        obs_tensor_dict = obs_as_tensor(obs_dict, model.device)
    
        print("--- 检查张量 ---")
        print("观测张量:", obs_tensor_dict["observation"])
        print("Mask 张量:", obs_tensor_dict["action_mask"])
        print("-----------------")

        with torch.no_grad():
            # extract_features 只需要观察张量
            features = model.policy.extract_features(obs_tensor)
            
            # 获取潜在向量
            latent_pi, latent_vf = model.policy.mlp_extractor(features)
            
            # 获取动作分布
            distribution = model.policy._get_action_dist_from_latent(latent_pi)

            logits = model.policy.action_net(latent_pi)  # [batch, num_actions]
            print("原始 logits:", logits)

            # 应用掩码：将不可行的动作设置为极小值
            masked_logits = logits.clone()
            masked_logits[~mask_tensor] = -1e8
            print("掩码后的 logits:", masked_logits)
            
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
    

    


if __name__ == "__main__":
    env, obs = test_env()
    model = load_oracle("log/oracle_TicTacToe_ppo_aggressive.zip")
    compute_criticality(env, model, obs)

