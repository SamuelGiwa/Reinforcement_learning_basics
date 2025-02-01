import gym
env = gym.make('CartPole-v1')
observation, info = env.reset()

for step_index in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info  = env.step(action)
    print("Step {}:".format(step_index))
    print("action: {}".format(action))
    print("observation: {}".format(observation))
    print("reward: {}".format(reward))
    print("done: {}".format(terminated))
    print("truncated: {}".format(truncated))
    print("info: {}".format(info))
    
    if terminated or truncated:
        print("Episode finished after {} steps".format(step_index + 1))
        break
env.close()