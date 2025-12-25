import gym
from gym import register
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from gym_env.pong_wrapper import PongWrapper

# Try to import gymnasium for TicTacToe
try:
    import gymnasium
    has_gymnasium = True
except ImportError:
    has_gymnasium = False

register(
    id='ToyPong-v0',
    entry_point='gym_env.toy_pong:ToyPong',
    kwargs={'args': None}
)

# Register TicTacToe with gymnasium if available
if has_gymnasium:
    from gymnasium.envs.registration import register as gymnasium_register
    gymnasium_register(
        id='TicTacToe-v0',
        entry_point='gym_env.tictactoe:TicTacToeEnv',
        kwargs={'opponent_type': 'random', 'minmax_depth': 9}
    )

    # Register TicTacToe Self-Play environment
    gymnasium_register(
        id='TicTacToe-SelfPlay-v0',
        entry_point='gym_env.tictactoe_selfplay:TicTacToeSelfPlayEnv',
        kwargs={'opponent_policy': None}
    )

    # Register TicTacToe Delta-Uniform Self-Play environment
    gymnasium_register(
        id='TicTacToe-DeltaSelfPlay-v0',
        entry_point='gym_env.tictactoe_delta_selfplay:TicTacToeDeltaSelfPlayEnv',
        kwargs={'baseline_pool': None, 'learned_pool': None, 'play_as_o_prob': 0.5}
    )


def make_env(args, test_viper=False):
    if args.env_name == "PongNoFrameskip-v4":
        env = make_atari_env("PongNoFrameskip-v4", n_envs=args.n_env)
        if test_viper is True:
            return PongWrapper(env, return_extracted_obs=True)
        return VecFrameStack(PongWrapper(env), n_stack=4)
    if args.env_name == "CartPole-v1":
        return DummyVecEnv([lambda: gym.make(args.env_name) for _ in range(args.n_env)])
    elif args.env_name == "ToyPong-v0":
        return DummyVecEnv([lambda: Monitor(gym.make(args.env_name, args=args)) for _ in range(args.n_env)])
    elif args.env_name == "TicTacToe-v0":
        if not has_gymnasium:
            raise ImportError("TicTacToe environment requires gymnasium. Please install it with: pip install gymnasium")
        opponent_type = getattr(args, 'tictactoe_opponent', 'random')
        # Use gymnasium directly (stable-baselines3 >= 1.6 supports gymnasium)
        # Import gymnasium's Monitor and DummyVecEnv
        from gymnasium.wrappers import RecordEpisodeStatistics
        try:
            # Try to use gymnasium's DummyVecEnv if available
            from stable_baselines3.common.vec_env import DummyVecEnv as SB3DummyVecEnv
            def make_tictactoe_env():
                env = gymnasium.make(args.env_name, opponent_type=opponent_type)
                env = RecordEpisodeStatistics(env)  # Gymnasium's equivalent of Monitor
                return env
            return SB3DummyVecEnv([make_tictactoe_env for _ in range(args.n_env)])
        except:
            # Fallback: use gymnasium env directly without vectorization if n_env=1
            if args.n_env == 1:
                env = gymnasium.make(args.env_name, opponent_type=opponent_type)
                return RecordEpisodeStatistics(env)
            raise

    raise NotImplementedError(f"Environment {args.env_name} not implemented")
