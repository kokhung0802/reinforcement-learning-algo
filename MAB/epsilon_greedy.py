"""
With epsilon-greedy, we select the best arm with a probability 1-epsilon and we select a
random arm with a probability epsilon.

action = arm of slot machine

"""
import gym_bandits
import gym
import numpy as np 


# create a simple 2-armed bandit environment
env = gym.make("BanditTwoArmedHighLowFixed-v0")

# store the number of times an arm is pulled
count = np.zeros(2)
# store the sum of rewards of each arm
sum_rewards = np.zeros(2)
# store average rewards of each arm
Q = np.zeros(2)
num_rounds = 100
epsilon = 0.5


def epsilon_greedy(epsilon):
    if np.random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q)


if __name__ == "__main__":
    for i in range(num_rounds):
        arm = epsilon_greedy(epsilon)
        next_state, reward, done, info = env.step(arm)

        count[arm] += 1
        sum_rewards[arm] += reward
        Q[arm] = sum_rewards[arm] / count[arm]
    
    print(Q)
    print('The optimal arm is arm {}'.format(np.argmax(Q)+1))

