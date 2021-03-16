from abs.abs_ele import AbsEle
from abs.abs_layer import Linear

from torch import Tensor

from typing import Union, List
from abc import ABC, abstractmethod


class AbstractEnvironment(ABC):

    @abstractmethod
    def step(self, state: Union[AbsEle, Tensor],
             action: Union[AbsEle, Tensor]) -> AbsEle:
        pass


class AbstractLinearEnvironment(AbstractEnvironment):
    """
    Abstract linear environment
    """

    def __init__(self, A: Tensor, B: Tensor, timestep=0.01):
        self.A = Linear(A.shape[0], A.shape[1], bias=False)
        self.A.weight.data = A
        for parameter in self.A.parameters():
            parameter.requires_grad = False

        self.B = Linear(B.shape[0], B.shape[1], bias=False)
        self.B.weight.data = B
        for parameter in self.B.parameters():
            parameter.requires_grad = False

        self.timestep = timestep

    def step(self, state: Union[AbsEle, Tensor],
             action: Union[AbsEle, Tensor]) -> AbsEle:
        return state + self.timestep * (self.A(state) + self.B(action))


class AbstractPiecewiseLinearEnvironment(AbstractEnvironment):
    def __init__(self, linear_envs: List[AbstractLinearEnvironment], conditions):
        self.linear_envs = linear_envs
        self.conditions = conditions

    def step(self, state: Union[AbsEle, Tensor],
             action: Union[AbsEle, Tensor]) -> AbsEle:
        for i, cond in enumerate(self.conditions):
            if cond(state):
                return self.linear_envs[i].step(state, action)

        return self.linear_envs[-1].step(state, action)

    def add(self, linear_env: AbstractLinearEnvironment, condition):
        self.linear_envs.append(linear_env)
        self.conditions.append(condition)
