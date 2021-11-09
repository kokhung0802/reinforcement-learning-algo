import gym
import numpy


# create the Frozen Lake environment
env = gym.make('FrozenLake-v1')

def value_iteration(env):
    num_iterations = 1000
    # threshold number for checking the convergence of the value function
    threshold = 1e-20
    # discount factor
    gamma = 1.0

    # initialize the value table by setting the value of all states to zero
    value_table = np.zeros(env.observation_space.n)

    for i in range(num_iterations):
        updated_value_table = np.copy(value_table)
        for s in range(env.observation_space.n):
            Q_values = 

