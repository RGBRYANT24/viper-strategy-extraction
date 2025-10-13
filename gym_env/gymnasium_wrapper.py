"""
Wrapper to convert Gymnasium environments to Gym-compatible format
This allows using Gymnasium envs with stable-baselines3 that expects old Gym API
"""

import gym
import numpy as np


class CompatibleBox(gym.spaces.Box):
    """
    Gym Box space that can compare equal with Gymnasium Box spaces
    """
    def __init__(self, *args, gymnasium_space=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._gymnasium_space = gymnasium_space

    def __eq__(self, other):
        # First try standard comparison
        if super().__eq__(other):
            return True

        # If that fails, check if we're comparing with our stored gymnasium space
        if hasattr(self, '_gymnasium_space') and other is self._gymnasium_space:
            return True

        # Also check if the other space has the same properties
        if hasattr(other, 'shape') and hasattr(other, 'dtype'):
            return (np.array_equal(self.low, other.low) and
                    np.array_equal(self.high, other.high) and
                    self.shape == other.shape and
                    self.dtype == other.dtype)

        return False


class CompatibleDiscrete(gym.spaces.Discrete):
    """
    Gym Discrete space that can compare equal with Gymnasium Discrete spaces
    """
    def __init__(self, n, gymnasium_space=None):
        super().__init__(n)
        self._gymnasium_space = gymnasium_space

    def __eq__(self, other):
        # First try standard comparison
        if super().__eq__(other):
            return True

        # If that fails, check if we're comparing with our stored gymnasium space
        if hasattr(self, '_gymnasium_space') and other is self._gymnasium_space:
            return True

        # Also check if the other space has the same n
        if hasattr(other, 'n'):
            return self.n == other.n

        return False


class GymnasiumToGymWrapper(gym.Wrapper):
    """
    Wraps a Gymnasium environment to make it compatible with stable-baselines3

    This wrapper converts gymnasium.spaces to gym.spaces while keeping
    the modern API (reset returns obs+info, step returns 5 values).

    stable-baselines3 >= 1.6.0 uses the new API:
    - reset() returns (obs, info)
    - step() returns (obs, reward, terminated, truncated, info)
    """

    def __init__(self, env):
        # Don't call super().__init__ because env is gymnasium, not gym
        self.env = env
        self.observation_space = self._convert_space(env.observation_space)
        self.action_space = self._convert_space(env.action_space)
        self.metadata = env.metadata
        self.reward_range = env.reward_range

    def _convert_space(self, space):
        """Convert gymnasium space to gym space with proper equality checking"""
        # Import the specific space type
        space_type = type(space).__name__

        if space_type == 'Box':
            return CompatibleBox(
                low=space.low,
                high=space.high,
                shape=space.shape,
                dtype=space.dtype,
                gymnasium_space=space
            )
        elif space_type == 'Discrete':
            return CompatibleDiscrete(n=space.n, gymnasium_space=space)
        elif space_type == 'MultiBinary':
            return gym.spaces.MultiBinary(n=space.n)
        elif space_type == 'MultiDiscrete':
            return gym.spaces.MultiDiscrete(nvec=space.nvec)
        else:
            # For other spaces, try to use as-is
            return space

    def reset(self, **kwargs):
        """Reset the environment - convert gymnasium API to gym API"""
        obs, info = self.env.reset(**kwargs)
        # stable-baselines3 >= 1.6.0 expects (obs, info) from Monitor wrapper
        return obs, info

    def step(self, action):
        """Step the environment - gymnasium already returns the correct format"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        # stable-baselines3 >= 1.6.0 expects (obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    def render(self, mode='human', **kwargs):
        """Render the environment"""
        return self.env.render()

    def close(self):
        """Close the environment"""
        return self.env.close()

    def seed(self, seed=None):
        """Set random seed"""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        # Gymnasium uses reset(seed=...) instead
        return None

    def __getattr__(self, name):
        """Forward all other attributes to the wrapped environment"""
        return getattr(self.env, name)
