# coding:utf-8

import numpy as np
from arm import Arm

class BernoulliArm(Arm):
    def __init__(self,prob):
        self.prob = prob

    def pull(self):
        return np.random.binomial(1,self.prob)

    def get_expected_reward(self):
        return self.prob
