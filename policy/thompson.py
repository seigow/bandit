# coding:utf-8

# Thompson Sampling for Bernoulli bandits

import numpy as np
from policy import Policy

class BinaryThompsonSampling(Policy):
    def __init__(self,n_arms):
        self.n = n_arms
        self.alpha = [1] * self.n
        self.beta = [1] * self.n

    def sampling(self):
        return np.random.beta(self.alpha,self.beta)

    def select_arm(self):
        i = np.argmax(self.sampling())
        return i

    def update_states(self,arm,reward):
        if reward:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
