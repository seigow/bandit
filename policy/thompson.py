# coding:utf-8

# Thompson Sampling for Bernoulli bandits

import numpy as np
from policy import Policy

class BinaryThompsonSampling(Policy):
    def __init__(self,n_arms):
        self.n = n_arms
        self.alpha = np.array([1] * self.n)
        self.beta = np.array([1] * self.n)

    def sampling(self):
        return np.random.beta(self.alpha,self.beta)

    def select_arm(self):
        return np.argmax(self.sampling())

    def update_states(self,arm,reward):
        if reward:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

class GaussianThompsonSampling(Policy):
    def __init__(self,n_arms):
        self.n = n_arms
        self.k = np.zeros(n_arms)
        self.mu = np.zeros(n_arms)

    def sampling(self):
        return np.random.normal(self.mu,1/(self.k+1))

    def select_arm(self):
        return np.argmax(self.sampling())

    def update_states(self,arm,reward):
        self.mu[arm] = (self.mu[arm]*self.k[arm]+reward)/(self.k[arm]+2)
        self.k[arm] += 1
