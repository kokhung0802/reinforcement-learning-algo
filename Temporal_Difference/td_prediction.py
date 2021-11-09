import gym
import pandas as pd 


env = gym.make('FrozenLake-v1')

learning_rate = 0.85
discount_rate = 0.90
num_episodes = 50000
num_timesteps = 1000


def random_policy():
    return env.action_space.sample()


if __name__ == "__main__":
    value = {}
    # initialize the value of all the states to 0.0
    for state in range(env.observation_space.n):
        value[state] = 0.0

    for i in range(num_episodes):
        state = env.reset()
        for step in range(num_timesteps):
            # generate an action
            action = random_policy()
            # Perform the selected action and store the next state information
            state_, reward, done, _ = env.step(action)
            # Compute the value of the state
            value[state] += learning_rate * (reward + discount_rate * value[state_] - value[state])
            state = state_
            if done:
                break   
        
    # evaluate our value function
    df = pd.DataFrame(list(value.items()), columns=['state', 'value'])
    print(df)
