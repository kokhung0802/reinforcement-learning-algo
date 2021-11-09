import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam


class DQN:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.replay_buffer = deque(maxlen=1000000)
    self.gamma = 0.95
    self.epsilon = 1.0
    self.update_rate = 5
    self.main_network = self.build_network()
    self.target_network = self.build_network()
    self.target_network.set_weights(self.main_network.get_weights())

  def build_network(self):
    model = Sequential()
    model.add(Input(shape=(self.state_size,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(self.action_size, activation='linear'))

    model.compile(loss='mse', optimizer=Adam())
    return model

  def store_transition(self, state, action, reward, next_state, done):
    self.replay_buffer.append((state, action, reward, next_state, done))

  def epsilon_greedy(self, state):
    if random.uniform(0,1) < self.epsilon:
      return np.random.randint(self.action_size)
    else:
      Q_values = self.main_network.predict(state)
      return np.argmax(Q_values[0])

  def train(self, batch_size):
    minibatch = random.sample(self.replay_buffer, batch_size)
    for state, action, reward, next_state, done in minibatch:
      if not done:
        target_Q = (reward + self.gamma*np.amax(self.target_network.predict(next_state)))
      else:
        target_Q = reward
      Q_values = self.main_network.predict(state)
      Q_values[0][action] = target_Q
      self.main_network.fit(state, Q_values, verbose=0)
    self.epsilon *= EXPLORATION_DECAY
    self.epsilon = max(self.epsilon, EXPLORATION_MIN)

  def update_target_network(self):
    self.target_network.set_weights(self.main_network.get_weights())


if __name__ == '__main__':
  env = gym.make('CartPole-v1')
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  num_episodes = 200
  num_timesteps = 20000
  dqn = DQN(state_size, action_size)
  done = False
  time_step = 0
  batch_size = 20
  return_list = []
  EXPLORATION_MAX = 1.0
  EXPLORATION_DECAY = 0.995
  EXPLORATION_MIN = 0.01

  model2 = dqn.build_network()

  for i in range(num_episodes):
  Return = 0
  state = env.reset()

  for t in range(num_timesteps):
    time_step += 1
    if time_step % dqn.update_rate == 0:
      dqn.update_target_network()
    state = np.reshape(state, [1, state_size])
    action = dqn.epsilon_greedy(state)
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, state_size])
    reward = reward if not done else -reward
    dqn.store_transition(state, action, reward, next_state, done)
    state = next_state
    Return += reward

    if done:
      return_list.append(Return)
      print(f'Episode: {i}, Return: {Return}')

    if len(dqn.replay_buffer) > batch_size:
      dqn.train(batch_size)
