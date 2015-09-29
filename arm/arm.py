# coding: utf-8

from abc import ABCMeta,abstractmethod

class Arm(metaclass=ABCMeta):
    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def get_expected_reward(self):
        pass
