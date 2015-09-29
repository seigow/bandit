# coding:utf-8

# Thompson Sampling for Bernoulli bandits

import numpy as np
from policy import Policy

class BinaryMultiplayTS(Policy):
    def __init__(self,n_arms,n_plays):
        self.n_arms = n_arms
        self.n_plays = n_plays
        self.alpha = [1] * self.n_arms
        self.beta = [1] * self.n_arms

    def sampling(self):
        return np.random.beta(self.alpha,self.beta)

    def select_arm(self):
        thetas = self.sampling()
        arms = np.array([],dtype=int)
        for _ in range(self.n_plays):
          i = np.argmax(thetas)
          thetas[i] = -1
          arms = np.append(arms,i)
        return arms

    def update_states(self,arm,reward):
        if reward:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
