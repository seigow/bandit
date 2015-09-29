# coding:utf-8

from abc import ABCMeta,abstractmethod

class Policy(metaclass=ABCMeta):
    @abstractmethod
    def select_arm(self):
        pass

    @abstractmethod
    def update_states(self):
        pass
