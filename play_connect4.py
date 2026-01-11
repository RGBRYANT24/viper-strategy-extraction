import gymnasium as gym
import gym_env
import sys
import numpy as np

def main():
    print("Welcome to Connect4!")
    print("You are playing against the AI.")
    
    opponent_type = 'random'
    if len(sys.argv) > 1:
        opponent_type = sys.argv[1]
    
    print(f"Opponent type: {opponent_type}")
    
    try:
        env = gym.make('Connect4-v0', opponent_type=opponent_type)
    except Exception as e:
        print(f"Error creating environment: {e}")
        return

    obs, _ = env.reset()
    env.render()
    
    done = False
    while not done:
        # Check if it's our turn
        # The environment handles the opponent's move inside step()
        # But we need to know if we are Player 1 (X) or Player 2 (O)
        # The environment logic is:
        # If play_as_o is False (default for random/minmax usually, unless updated):
        # We are X (1). Opponent is O (-1).
        
        # Let's just ask for input.
        print("\nEnter column (0-6):")
        try:
            action = int(input())
            if action < 0 or action > 6:
                print("Invalid column. Please enter 0-6.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        env.render()
        
        if 'illegal_move' in info:
            print("Illegal move! Try again.")
            # If illegal move, the game ends in the current implementation?
            # Looking at connect4.py:
            # if not self._is_valid_action(action):
            #      self.done = True
            #      return ..., -10, True, False, {'illegal_move': True}
            # Yes, it ends. We should probably prevent that in this test script if we interpret 'illegal move' as just 'retry' for the human, 
            # but the env is strict. Let's just restart if illegal.
            if done: 
                print("Game Over due to illegal move.")
                break
        
        if done:
            if reward == 1:
                print("You Won!")
            elif reward == -1:
                print("You Lost!")
            elif reward == 0:
                print("Draw!")
            else:
                print(f"Game Over. Reward: {reward}")
            
            # Print winner info if available
            if 'winner' in info:
                print(f"Winner: {info['winner']}")

if __name__ == "__main__":
    main()
