import gym
import pandas as pd 
from collections import defaultdict


# max number of timesteps in an episode
num_timesteps = 100
# number of episodes we want to generate
num_iterations = 5000
# define a dict where values are float
total_return = defaultdict(float)
# define a dict where values are int
N = defaultdict(int)

# Create a blackjack environment
env = gym.make('Blackjack-v1')


def policy(state):
    # if total card larger than 19 do not take anymore card (0), else take card (1)
    return 0 if state[0] > 19 else 1

# generate an episode using the given policy
def generate_episode(policy):
    episode = []
    # state: (total value of player, total value of dealer, any reusable ace [True/False])
    state = env.reset()

    for t in range(num_timesteps):
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))

        if done:
            break
        
        state = next_state
    
    return episode


if __name__ == "__main__":
    # for each episode
    for i in range(num_iterations):
        # generate an episode using the policy function
        # e.g: [((17, 5, False), 1, 0.0), ((19, 5, False), 1, -1.0)]
        episode = generate_episode(policy)
        states, actions, rewards = zip(*episode)
        # for each step in the episode
        for t, state in enumerate(states):
            R = sum(rewards[t:])
            total_return[state] += R 
            # Update the number of times the state is visited in the episode
            N[state] += 1

    # Convert the total_returns dictionary into a data frame
    total_return = pd.DataFrame(total_return.items(), columns=['state', 'total_return'])
    # Convert the counter N dictionary into a data frame
    N = pd.DataFrame(N.items(), columns=['state', 'N'])
    # Merge the two data frames on states
    df = pd.merge(total_return, N, on="state")
    # compute the value of the state as the average return
    df['value'] = df['total_return'] / df['N']
    print(df.head(10))




