import numpy as np
import scipy.optimize as optimize
from policy import Policy

def get_kl(p,q):
    d = p * np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
    return d

def get_klprime(p,q):
    return (q-p)/(q*(1-q))

class KLUCB(Policy):
    def __init__(self,n_arms):
        # num of times an arm has been selected
        self.N = np.zeros(n_arms,dtype=int)
        self.n_arms = n_arms
        # total reward of an arm
        self.S = np.zeros(n_arms)

    def get_klucb(self,k,t):
        # can get by Newton Iteration or dichotomic search
        # 0 <= p <= 1, p <= q <= 1
        DELTA = 1.0e-8 # how to set this value???
        EPS = 1e-12 # how to set this value???
        logndn = np.log(t)/self.N[k]
        p = max(self.S[k]/self.N[k],DELTA)
        if p>=1: return 1
        q = p + DELTA
        for _ in range(20):
            f = logndn - get_kl(p,q)
            if f**2<EPS:
                break
            fprime = - get_klprime(p,q)
            q = min(1-DELTA,max(q - f/fprime,p+DELTA))
        return q

    def select_arm(self):
        t = sum(self.N)
        ucb = np.array([.0]*self.n_arms)
        for k in range(self.n_arms):
            if 0==self.N[k]:
                self.N[k] += 1
                return k
            ucb[k] = self.get_klucb(k,t)
        arm = np.argmax(ucb)
        self.N[arm] += 1
        return arm

    def update_states(self,arm,reward):
        self.N[arm] += 1
        self.S[arm] += reward
