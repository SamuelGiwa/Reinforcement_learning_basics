import gym
from gym import spaces
import numpy as np

class CustomGridEnv(gym.Env):
    """Custom Grid World Environment"""

    def __init__(self):
        super().__init__()

        # Action space: [0 = Up, 1 = Down, 2 = Left, 3 = Right]
        self.action_space = spaces.Discrete(4)

        # state space: 5x5 grid (agent's position)
        self.grid_size = 6
        
        self.observation_space = spaces.Discrete(self.grid_size^2)  # 5x5 = 25 states

        # grid size
        

        # start and goal positions
        self.start_pos = (0, 0)  # Top-left
        self.goal_pos = (5, 5)   # Bottom-right

        # Initialize state
        self.agent_pos = self.start_pos
        

    def reset(self):
        """Resets the environment to the initial state."""
        self.agent_pos = self.start_pos  # Reset agent to start position
        return self._get_state()  # Return the initial state

    def step(self, action):
        """Takes a step in the environment."""
        x, y = self.agent_pos

        # Define movement based on action
        if action == 0:  # Move Up
            x = max(0, x - 1)
        elif action == 1:  # Move Down
            x = min(5, x + 1)
        elif action == 2:  # Move Left
            y =  max(0, y - 1)
        elif action == 3:  # Move Right
            y = min(5, y + 1)
        # Update agent position
        self.agent_pos = (x, y)

        # Define reward and check if episode is done
        reward = 1 if self.agent_pos == self.goal_pos else -0.1  # Small penalty for each step
        done = self.agent_pos == self.goal_pos  # Episode ends if goal is reached

        return self._get_state(), reward, done, {}  # Return next state, reward, done flag, info

    def _get_state(self):
        """Returns a unique state index for the agent's position."""
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]

    def render(self, mode="human"):
        """Renders the grid."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:, :] = "."
        grid[self.goal_pos] = "G"  # Goal
        grid[self.agent_pos] = "A"  # Agent
        print("\n".join([" ".join(row) for row in grid]) + "\n")

    def close(self):
        """Cleanup resources if needed."""
        pass

# Register environment (optional if using Gym)
# import gym
# from gym.envs.registration import register

# register(
#     id="CustomGrid-v0",
#     entry_point=CustomGridEnv
# )

# Test the environment
env = CustomGridEnv()

# state = env.reset()
env.render()

for _ in range(100):  
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    env.render()
    if done:
        print("Goal reached!")
        break

env.close()