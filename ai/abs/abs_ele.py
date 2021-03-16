""" Abstract domain elements. """

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parent.parent))

from abs.abs_utils import valid_lb_ub


class AbsEle(ABC):
    """ The abstract element propagated throughout the layers.
        FIXME maybe optimize using CPP extensions in the future.
    """

    @classmethod
    @abstractmethod
    def by_intvl(cls, lb: Tensor, ub: Tensor, *args, **kwargs) -> 'AbsEle':
        """ Transform the Lower Bounds and Upper Bounds to abstract elements propagated throughout the layers. """
        raise NotImplementedError()

    def __getitem__(self, key):
        """ It may only need to compute some rows but not all in the abstract element. Select those rows from here. """
        raise NotImplementedError()

    @abstractmethod
    def size(self):
        """ Return the size of any concretized data point from this abstract element. """
        raise NotImplementedError()

    @abstractmethod
    def dim(self):
        """ Return the number of dimensions for any concretized data point from this abstract element. """
        raise NotImplementedError()

    @abstractmethod
    def lb(self) -> Tensor:
        """ Lower Bound. """
        raise NotImplementedError()

    @abstractmethod
    def ub(self) -> Tensor:
        """ Upper Bound. """
        raise NotImplementedError()

    def gamma(self) -> Tuple[Tensor, Tensor]:
        """ Transform the abstract elements back into Lower Bounds and Upper Bounds. """
        lb = self.lb()
        ub = self.ub()
        assert valid_lb_ub(lb, ub)
        return lb, ub

    # ===== Below are pre-defined operations that every abstract element must support. =====

    @abstractmethod
    def view(self, *shape) -> 'AbsEle':
        raise NotImplementedError()

    @abstractmethod
    def contiguous(self) -> 'AbsEle':
        raise NotImplementedError()

    def squeeze(self, dim=None) -> 'AbsEle':
        shape = list(self.size())
        if dim is not None and shape[dim] != 1:
            # nothing to squeeze
            return self

        if dim is None:
            while 1 in shape:
                shape.remove(1)
        else:
            shape = shape[:dim] + shape[dim+1:]
        return self.view(*shape)

    def unsqueeze(self, dim) -> 'AbsEle':
        if dim < 0:
            # following pytorch doc
            dim = dim + self.dim() + 1

        shape = list(self.size())
        shape = shape[:dim] + [1] + shape[dim:]
        return self.view(*shape)

    @abstractmethod
    def transpose(self, dim0, dim1) -> 'AbsEle':
        raise NotImplementedError()

    @abstractmethod
    def matmul(self, other: Tensor) -> 'AbsEle':
        raise NotImplementedError()

    @abstractmethod
    def __add__(self, other) -> 'AbsEle':
        raise NotImplementedError()

    def to_dense(self) -> 'AbsEle':
        return self

    # ===== Below are pre-defined functions to compute distances of the abstract elements to certain property. =====

    def col_le_val(self, idx: int, threshold: float, mean: float = 0., range: float = 1.) -> Tensor:
        """ Return a distance tensor for 'idx-th column value <= threshold'.
            @mean and @range are for de-normalization since it is about absolute value.
        """
        t = self.ub()[..., idx]
        threshold = (threshold - mean) / range
        d = t - threshold
        return F.relu(d)

    def col_ge_val(self, idx: int, threshold: float, mean: float = 0., range: float = 1.) -> Tensor:
        """ Return a distance tensor for 'idx-th column value >= threshold'.
            @mean and @range are for de-normalization since it is about absolute value.
        """
        t = self.lb()[..., idx]
        threshold = (threshold - mean) / range
        d = threshold - t
        return F.relu(d)

    def cols_not_max(self, *idxs: int) -> Tensor:
        """ Return a distance tensor for 'Forall idx-th column value is not maximal among all'.

            <Loss definition>: Intuitively, always-not-max => exists col . target < col is always true.
            Therefore, target_col.UB() - other_col.LB() should < 0, if not, that is the distance.
            As long as some of the others < 0, it's OK (i.e., min).
        """
        raise NotImplementedError()

    def cols_is_max(self, *idxs: int) -> Tensor:
        """ Return a distance tensor for 'Exists idx-th column value is the maximal among all.

            <Loss definition>: Intuitively, some-is-max => exists target . target > all_others is always true.
            Therefore, other_col.UB() - target_col.LB() should < 0, if not, that is the distance.
            All of the others should be accounted (i.e., max).
        """
        raise NotImplementedError()

    def cols_not_min(self, *idxs: int) -> Tensor:
        """ Return a distance tensor for 'Forall idx-th column value is not minimal among all'.

            <Loss definition>: Intuitively, always-not-min => exists col . col < target is always true.
            Therefore, other_col.UB() - target_col.LB() should < 0, if not, that is the distance.
            As long as some of the others < 0, it's OK (i.e., min).
        """
        raise NotImplementedError()

    def cols_is_min(self, *idxs: int) -> Tensor:
        """ Return a distance tensor for 'Exists idx-th column value is the minimal among all.

            <Loss definition>: Intuitively, some-is-min => exists target . target < all_others is always true.
            Therefore, target_col.UB() - other_col.LB() should < 0, if not, that is the distance.
            All of the others should be accounted (i.e., max).
        """
        raise NotImplementedError()

    def worst_of_labels_predicted(self, labels: Tensor) -> Tensor:
        """ Return the worst case output tensor for 'Forall batched input, their prediction should match the corresponding label'.

            <Loss definition>: Intuitively, this is specifying a label_is_max for every input abstraction.
        :param label: same number of batches as self
        """
        raise NotImplementedError()

    def worst_of_labels_not_predicted(self, labels: Tensor) -> Tensor:
        """ Return the worst case output tensor for 'Forall batched input, none of their prediction matches the corresponding label'.

            <Loss definition>: Intuitively, this is specifying a label_not_max for every input abstraction.
        :param label: same number of batches as self
        """
        raise NotImplementedError()

    # ===== Finally, some utility functions shared by different domains. =====

    def _idxs_not(self, *idxs: int) -> List[int]:
        """ Validate and get other column indices that are not specified. """
        col_size = self.size()[-1]
        assert len(idxs) > 0 and all([0 <= i < col_size for i in idxs])
        assert len(set(idxs)) == len(idxs)  # no duplications
        others = [i for i in range(col_size) if i not in idxs]
        assert len(others) > 0
        return others
    pass


class PtwiseEle(AbsEle):
    """ 'Ptwise' if the abstract domain is non-Relational in each field.
        As a consequence, their loss functions are purely based on LB/UB tensors.
    """

    def cols_not_max(self, *idxs: int) -> Tensor:
        # FIXME Not considering corner case when target == col?
        others = self._idxs_not(*idxs)
        others = self.lb()[..., others]

        res = []
        for i in idxs:
            target = self.ub()[..., [i]]
            diff = target - others  # will broadcast
            diff = F.relu(diff)
            mins, _ = torch.min(diff, dim=-1)
            res.append(mins)
        return sum(res)

    def cols_is_max(self, *idxs: int) -> Tensor:
        # FIXME Not considering corner case when target == col?
        others = self._idxs_not(*idxs)
        others = self.ub()[..., others]

        res = []
        for i in idxs:
            target = self.lb()[..., [i]]
            diffs = others - target  # will broadcast
            diffs = F.relu(diffs)
            res.append(diffs)

        if len(idxs) == 1:
            all_diffs = res[0]
        else:
            all_diffs = torch.stack(res, dim=-1)
            all_diffs, _ = torch.min(all_diffs, dim=-1)  # it's OK to have either one to be max, thus use torch.min()

        # then it needs to surpass everybody else, thus use torch.max() for maximum distance
        diffs, _ = torch.max(all_diffs, dim=-1)
        return diffs

    def cols_not_min(self, *idxs: int) -> Tensor:
        # FIXME Not considering corner case when target == col?
        others = self._idxs_not(*idxs)
        others = self.ub()[..., others]

        res = []
        for i in idxs:
            target = self.lb()[..., [i]]
            diffs = others - target  # will broadcast
            diffs = F.relu(diffs)
            mins, _ = torch.min(diffs, dim=-1)
            res.append(mins)
        return sum(res)

    def cols_is_min(self, *idxs: int) -> Tensor:
        # FIXME Not considering corner case when target == col?
        others = self._idxs_not(*idxs)
        others = self.lb()[..., others]

        res = []
        for i in idxs:
            target = self.ub()[..., [i]]
            diffs = target - others  # will broadcast
            diffs = F.relu(diffs)
            res.append(diffs)

        if len(idxs) == 1:
            all_diffs = res[0]
        else:
            all_diffs = torch.stack(res, dim=-1)
            all_diffs, _ = torch.min(all_diffs, dim=-1)  # it's OK to have either one to be min, thus use torch.min()

        # then it needs to surpass everybody else, thus use torch.max() for maximum distance
        diffs, _ = torch.max(all_diffs, dim=-1)
        return diffs

    def worst_of_labels_predicted(self, labels: Tensor) -> Tensor:
        full_lb = self.lb()
        full_ub = self.ub()
        res = []
        for i in range(len(labels)):
            cat = labels[i]
            piece_outs_lb = full_lb[[i]]
            piece_outs_ub = full_ub[[i]]

            # default lb-ub or ub-lb doesn't know that target domain has distance 0, so specify that explicitly
            lefts = piece_outs_ub[..., :cat]
            rights = piece_outs_ub[..., cat+1:]
            target = piece_outs_lb[..., [cat]]

            full = torch.cat((lefts, target, rights), dim=-1)
            diffs = full - target  # will broadcast
            # no need to ReLU here, negative values are also useful
            res.append(diffs)

        res = torch.cat(res, dim=0)
        return res

    def worst_of_labels_not_predicted(self, labels: Tensor) -> Tensor:
        full_lb = self.lb()
        full_ub = self.ub()
        res = []
        for i in range(len(labels)):
            cat = labels[i]
            piece_outs_lb = full_lb[[i]]
            piece_outs_ub = full_ub[[i]]

            # default lb-ub or ub-lb doesn't know that target domain has distance 0, so specify that explicitly
            lefts = piece_outs_lb[..., :cat]
            rights = piece_outs_lb[..., cat+1:]
            target = piece_outs_ub[..., [cat]]

            full = torch.cat((lefts, target, rights), dim=-1)
            diffs = target - full  # will broadcast
            # no need to ReLU here, negative values are also useful
            res.append(diffs)

        res = torch.cat(res, dim=0)
        raise NotImplementedError('To use this as distance, it has to have target category not being max, ' +
                                  'thus use torch.min(dim=-1) then ReLU().')
        return res
    pass


class AbsData(Dataset):
    """ Storing the split LB/UB boxes/abstractions. """
    def __init__(self, boxes_lb: Tensor, boxes_ub: Tensor, boxes_extra: Tensor = None):
        assert valid_lb_ub(boxes_lb, boxes_ub)
        self.boxes_lb = boxes_lb
        self.boxes_ub = boxes_ub
        self.boxes_extra = boxes_extra
        return

    def __len__(self):
        return len(self.boxes_lb)

    def __getitem__(self, idx):
        if self.boxes_extra is None:
            return self.boxes_lb[idx], self.boxes_ub[idx]
        else:
            return self.boxes_lb[idx], self.boxes_ub[idx], self.boxes_extra[idx]
    pass
