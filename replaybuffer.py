import random
from collections import deque, namedtuple

# Creating the Transition object as a  global variable
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.buffer)

# Example
if __name__ == "__main__":
    # Initialize replay buffer with capacity of 5
    replay_buffer = ReplayBuffer(capacity=5)

    # Add transitions to the buffer
    replay_buffer.push(1, 'a', 10, 2, False)
    replay_buffer.push(2, 'b', 20, 3, False)
    replay_buffer.push(3, 'c', 30, 4, True)
    replay_buffer.push(4, 'd', 40, 5, False)
    replay_buffer.push(5, 'e', 50, 6, True)

    # Print the current contents of the replay buffer
    print("Replay Buffer Contents:")
    for transition in replay_buffer.buffer:
        print(transition)

    # Sampling a batch of 3 transitions
    batch_size = 3
    batch = replay_buffer.sample(batch_size)

    # Printing the sampled batch
    print("\nSampled Batch:")
    print(f"States: {batch.state}")
    print(f"Actions: {batch.action}")
    print(f"Rewards: {batch.reward}")
    print(f"Next States: {batch.next_state}")
    print(f"Dones: {batch.done}")