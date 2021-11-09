import numpy as np 
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym
import tensorflow_probability as tfp 


class ActorCritic(Model):
    def __init_-(self, action_dim):
        super().__init__()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.critic = Dense(1, activation=None)
        self.actor = Dense(action_dim, activation=None)

    def call(self, input_data):
        x = self.fc1(input_data)
        x1 = self.fc2(x)
        actor = self.actor(x1)
        critic = self.critic(x1)
        return critic, actor 


class Agent:
    def __init__(self, action_dim=4, discount_rate=0.99):
        self.discount_rate = discount_rate
        self.opt = Adam(learning_rate=1e-4)
        self.actor_critic = ActorCritic(action_dim)

        