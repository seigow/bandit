#  coding:utf-8

import numpy as np
from policy import Policy

class EpsilonGreedy(Policy):
    def __init__(self,n_arms,epsilon):
        self.n = n_arms # num of arms
        self.epsilon = epsilon # exploration rate
        self.drawn_counts = np.zeros(self.n,dtype=int) # num of draws
        self.expected_reward = np.zeros(self.n) # reward probability for each arm

    def select_arm(self):
        if np.random.random() > self.epsilon:
            return np.argmax(self.expected_reward)
        else:
            return np.random.randint(self.n)

    def update_states(self,arm,reward):
        self.drawn_counts[arm] += 1
        n = self.drawn_counts[arm]
        self.expected_reward[arm] += (reward - self.expected_reward[arm]) / n

class DecayEpsilonGreedy(Policy):
    def __init__(self,n_arms,epsilon_decay=50):
        self.n = n_arms # num of arms
        self.drawn_counts = [0] * self.n # num of draws
        self.expected_reward = [0.] * self.n # reward probability for each arm
        self.decay = epsilon_decay

    def get_epsilon(self):
        total = np.sum(self.drawn_counts)
        return float(self.decay) / (total+float(self.decay))

    def select_arm(self):
        epsilon = self.get_epsilon()
        if np.random.random() > epsilon:
            return np.argmax(self.expected_reward)
        else:
            return np.random.randint(self.n)

    def update_states(self,arm,reward):
        self.drawn_counts[arm] += 1
        n = self.drawn_counts[arm]
        self.expected_reward[arm] += (reward - self.expected_reward[arm]) / n
