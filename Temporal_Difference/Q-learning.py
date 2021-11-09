import gym
import numpy as np 
import pandas as pd 
import random


env = gym.make('FrozenLake-v1')

learning_rate = 0.85
discount_rate = 0.9
epsilon = 0.8
num_episodes = 5000
num_timesteps = 1000


def epsilon_greedy(state, epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        # return action in the action saoce that has the highest Q-value
        return max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])


if __name__ == "__main__":
    Q = {}
    # initialize each Q function value with 0.0
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            Q[(state, action)] = 0.0

    for i in range(num_episodes):
        # Initialize the state by resetting the environment
        state = env.reset()
        for step in range(num_timesteps):
            action = epsilon_greedy(state, epsilon)
            state_, reward, done, _ = env.step(action)
            
            # select the action a′ that has the maximum Q value in the next state s′ using greedy policy
            action_ = np.argmax([Q[(state_, action)] for action in range(env.action_space.n)])

            Q[(state, action)] += learning_rate * (reward + discount_rate*Q[(state_, action_)] - Q[(state, action)])

            state = state_

            if done:
                break

    # evaluate Q function
    df = pd.DataFrame(list(Q.items()), columns=['state-action', 'value'])
    print(df)
