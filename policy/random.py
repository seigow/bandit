# coding:utf-8

import numpy as np
from policy import Policy

class RandomPolicy(Policy):
    def __init__(self,n_arms):
        self.n_arms = n_arms

    def select_arm(self):
        return np.random.uniform(0,n_arms-1)

    def update_states(self):
        pass
