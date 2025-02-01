import gym
import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import display, clear_output

# interactive mode for real-time updating
plt.ion()

# environment
env = gym.make("MountainCar-v0", render_mode="rgb_array")

frames = []  # Store frames for visualization

obs, info = env.reset()
done = False
max_frames = 2000  

while not done and len(frames) < max_frames:
    frame = env.render()  # Capture frame
    frames.append(frame)
    action = env.action_space.sample()  # Take a random action
    obs, reward, done, truncated, info = env.step(action)

env.close()

# # animation
fig, ax = plt.subplots()
for frame in frames:
    ax.clear()
    ax.imshow(frame)
    ax.axis("off")
    display(fig)
    plt.pause(0.05)  
    clear_output(wait=True)

plt.ioff()  
plt.show()