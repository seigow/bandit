# coding:utf-8

import numpy as np
from arm import Arm

class GaussianArm(Arm):
    def __init__(self,mu,variance):
        self.mu = mu
        self.variance = variance

    def pull(self):
        return np.random.normal(self.mu,self.variance)

    def get_expected_reward(self):
        return self.mu
