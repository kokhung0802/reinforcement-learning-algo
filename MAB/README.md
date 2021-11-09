## About the bandit environment
2-armed bandit environment 

action space: 2 (there are two arms)

check the probability distribution of the arm

print(env.p_dist)

output: [0.8, 0.2]

with arm 1, we win the game 80% of the time and with arm 2, we win the game 20% of the time

Our goal is to find out whether pulling arm 1 or arm 2 makes us win the game most of the time.

----------------------------------------------------------------------

## Creating a bandit in the Gym

The Gym does not come with a prepackaged bandit environment.

We will use the open-source version of the bandit environment provided by Jesse Cooper.

`````````````
git clone https://github.com/JKCooper2/gym-bandits
cd gym-bandits
pip install -e .

`````````````

gym_bandits provides several versions of the bandit environment. We can examine
the different bandit versions at https://github.com/JKCooper2/gym-bandits


-----------------------------------------------------------------------
## Exploration strategies
- Epsilon-greedy
- Softmax exploration
- Upper confidence bound
- Thomson sampling



## Reference
1. Use Case for pip install -e
https://stackoverflow.com/questions/42609943/what-is-the-use-case-for-pip-install-e

