""" Redefined NN modules that take these abstract elements as inputs. Note that activation layers are defined by
    each abstract domain in individual. It may have different approximations.
"""

import sys
from pathlib import Path
from typing import Union

import torch
from torch import Tensor, nn

sys.path.append(str(Path(__file__).resolve().parent.parent))

from abs.abs_ele import AbsEle


class Linear(nn.Linear):
    """ Linear layer with the ability to take approximations rather than concrete inputs. """

    def forward(self, e: Union[AbsEle, Tensor], *args, **kwargs) -> Union[AbsEle, Tensor]:
        """ I have to implement the forward computation by myself, because F.linear() may apply optimization using
            torch.addmm() which requires inputs to be tensors.
        """
        if not isinstance(e, AbsEle):
            return super().forward(e)

        output = e.matmul(self.weight.t())
        if self.bias is not None:
            output += self.bias
        return output
    pass


class Normalize(nn.Module):
    """ Normalize following a fixed mean/variance.
        This class was originally written to comply with the BatchNorm trained networks in PLDI'19 experiments.
        However, it turned out the BatchNorm mean/variance collected from there is problematic. So we end up not using
        the extracted parameters, but directly call TF networks as oracles.
    """

    def __init__(self, beta, gamma, mean, variance, epsilon=1e-5):
        """
        :param epsilon: 1e-5 is the default value in tflearn implementation,
                        this value is used to avoid devide 0, and does not change in training.
        """
        assert (variance >= 0).all() and epsilon > 0
        super().__init__()
        self.beta = torch.from_numpy(beta)
        self.gamma = torch.from_numpy(gamma)
        self.mean = torch.from_numpy(mean)

        # somehow it needs to change variance first, otherwise it becomes ndarray again
        self.variance = variance + epsilon
        self.variance = torch.from_numpy(variance)
        return

    def forward(self, x: Union[AbsEle, Tensor]) -> Union[AbsEle, Tensor]:
        x_hat = (x - self.mean) / torch.sqrt(self.variance)
        return x_hat * self.gamma + self.beta
    pass
