# coding:utf-8

import numpy as np
from arm import Arm

class GaussianArm(Arm):
    def __init__(self,mu,variance,minval=0,maxval=1):
        self.mu = mu
        self.variance = variance
        self.minval = minval
        self.maxval = maxval

    def pull(self):
        val = np.random.normal(self.mu,self.variance)
        if self.minval<val<self.maxval:
            return val
        elif val>self.minval:
            return self.minval
        else:
            return self.maxval

    def get_expected_reward(self):
        return self.mu
