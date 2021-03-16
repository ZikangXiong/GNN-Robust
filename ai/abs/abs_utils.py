""" Utilities for all over the project. """

from typing import Tuple, Union, Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def valid_lb_ub(lb: Union[float, Tensor], ub: Union[float, Tensor], eps: float = 1e-8) -> bool:
    """ To be valid:
        (1) Size ==
        (2) LB <= UB
    :param eps: added for numerical instability.
    """
    if isinstance(lb, float) and isinstance(ub, float):
        return lb <= ub + eps

    if lb.size() != ub.size():
        return False

    # '<=' will return a uint8 tensor of 1 or 0 for each element, it should have all 1s.
    return True # (lb <= ub + eps).all()


def cat0(*ts: Tensor) -> Tensor:
    """ Usage: simplify `torch.cat((ts1, ts2), dim=0)` to `cat0(ts1, ts2)`. """
    return torch.cat(ts, dim=0)


def divide_pos_neg(ws: Tensor) -> Tuple[Tensor, Tensor]:
    """
    :return: positive part and negative part of the original tensor, 0 filled elsewhere.
    """
    pos_weights = F.relu(ws)
    neg_weights = F.relu(ws * -1) * -1
    return pos_weights, neg_weights


def total_area(lb: Tensor, ub: Tensor, eps: float = 1e-8, by_batch: bool = False) -> float:
    """ Return the total area constrained by LB/UB. Area = \Sum_{batch}{ \Prod{Element} }.
    :param lb: <Batch x ...>
    :param ub: <Batch x ...>
    :param by_batch: if True, return the areas of individual abstractions
    """
    assert valid_lb_ub(lb, ub)
    diff = ub - lb
    diff += eps  # some dimensions may be degenerated, then * 0 becomes 0.

    while diff.dim() > 1:
        diff = diff.prod(dim=-1)

    if by_batch:
        return diff
    else:
        return diff.sum().item()


def sample_points(lb: Tensor, ub: Tensor, K: int) -> Tensor:
    """ Uniformly sample K points for each region.
    :param lb: Lower bounds, batched
    :param ub: Upper bounds, batched
    :param K: how many pieces to sample
    """
    assert valid_lb_ub(lb, ub)
    assert K >= 1

    repeat_dims = [1] * (len(lb.size()) - 1)
    base = lb.repeat(K, *repeat_dims)  # repeat K times in the batch, preserving the rest dimensions
    width = (ub - lb).repeat(K, *repeat_dims)

    coefs = torch.rand_like(base)
    pts = base + coefs * width
    return pts


def sample_regions(lb: Tensor, ub: Tensor, K: int, depth: int) -> Tuple[Tensor, Tensor]:
    """ Uniformly sample K sub-regions with fixed width boundaries for each sub-region.
    :param lb: Lower bounds, batched
    :param ub: Upper bounds, batched
    :param K: how many pieces to sample
    :param depth: bisecting original region width @depth times for sampling
    """
    assert valid_lb_ub(lb, ub)
    assert K >= 1 and depth >= 1

    repeat_dims = [1] * (len(lb.size()) - 1)
    base = lb.repeat(K, *repeat_dims)  # repeat K times in the batch, preserving the rest dimensions
    orig_width = ub - lb

    try:
        piece_width = orig_width / (2 ** depth)
        # print('Piece width:', piece_width)
        avail_width = orig_width - piece_width
    except RuntimeError as e:
        print('Numerical error at depth', depth)
        raise e

    piece_width = piece_width.repeat(K, *repeat_dims)
    avail_width = avail_width.repeat(K, *repeat_dims)

    coefs = torch.rand_like(base)
    lefts = base + coefs * avail_width
    rights = lefts + piece_width
    return lefts, rights


def gen_grid_pts(base_lb: Tensor, base_ub: Tensor, pts_each: int) -> Tensor:
    """ Generate points in the meshgrid of entire region.
    :param pts_each: points on each dimension
    :return: pts_each x pts_each ... x pts_each (dim of them) x state, structure preserved
    """
    assert base_lb.dim() == 1 and base_ub.dim() == 1 and len(base_lb) == len(base_ub)

    xi_pivots = []
    for i in range(len(base_lb)):
        pivots = torch.linspace(base_lb[i], base_ub[i], pts_each)
        xi_pivots.append(pivots)

    mg = torch.meshgrid(*xi_pivots)
    grid_pts = torch.stack(mg, dim=-1)
    return grid_pts


def gen_vertices(base_lb: Tensor, base_ub: Tensor) -> Tensor:
    """ Generate the vertices of a hyper-rectangle bounded by LB/UB.
    :return: N x State tensor, all vertices flattened
    """
    assert base_lb.dim() == 1 and base_ub.dim() == 1 and len(base_lb) == len(base_ub)

    # basically, a cartesian product of LB/UB on each dimension
    lbub = torch.stack((base_lb, base_ub), dim=-1)  # Dim x 2
    vtx = torch.cartesian_prod(*list(lbub))
    return vtx


def gen_edge_pts(base_lb: Tensor, base_ub: Tensor, pts_each: int) -> Tensor:
    """ Generate points on the edges, with each edge equally divided.
    :param pts_each: points on each edge
    :return: N x State tensor, all states flattened
    """
    assert base_lb.dim() == 1 and base_ub.dim() == 1 and len(base_lb) == len(base_ub)
    assert pts_each >= 2, 'at least two points on each edge for the vertices'

    vtx = gen_vertices(base_lb, base_ub)
    if pts_each == 2:
        return vtx

    def _diff_at_one_idx(vi: Tensor, vj: Tensor, eps: float = 1e-5) -> Optional[Tensor]:
        """ If vi and vj only differs at one dimension, return that dim, otherwise return None. """
        diff = vi - vj
        bits = (diff.abs() > eps).nonzero().squeeze(dim=1)
        if len(bits) == 1:
            return bits
        else:
            return None

    # generate edge points for each pair of contiguous vertices (i.e., different only in one dimension),
    edge_pts = [vtx]  # store all vertices first
    for i in range(len(vtx)):
        for j in range(i + 1, len(vtx)):
            vi, vj = vtx[i], vtx[j]
            idx = _diff_at_one_idx(vi, vj)
            if idx is None:
                continue

            idx = idx.item()
            mn, mx = min(vi[idx], vj[idx]), max(vi[idx], vj[idx])
            pivots = torch.linspace(mn.item(), mx.item(), pts_each)
            pts = vi.repeat(pts_each, 1)  # use expand() doesn't work here, need repeat()
            pts[:, idx] = pivots
            edge_pts.append(pts[1:-1])  # exclude the two vertices which are already stored

    edge_pts = torch.cat(edge_pts, dim=0)
    return edge_pts


def gen_edges(base_lb: Tensor, base_ub: Tensor) -> Tuple[Tensor, Tensor]:
    """ Generate edges of the hyperrectangle.
    :param base_lb: unbatched
    :param base_ub: unbatched
    :return: batched edges
    """
    assert base_lb.dim() == base_ub.dim() == 1, 'should be unbatched'

    tot_dim = len(base_lb)
    vtxs = gen_vertices(base_lb, base_ub)
    edge_lbs, edge_ubs = [], []
    for i in range(tot_dim):
        # checking the two contiguous vertices that differ only in dimension d
        for v1_idx in range(len(vtxs)):
            v2_idx = flip_bit(v1_idx, tot_dim, i)
            if v1_idx > v2_idx:
                # only check once, when v1_idx < v2_idx
                continue

            v1, v2 = vtxs[v1_idx], vtxs[v2_idx]
            if valid_lb_ub(v1, v2):
                edge_lbs.append(v1)
                edge_ubs.append(v2)
            else:
                assert valid_lb_ub(v2, v1)
                edge_lbs.append(v2)
                edge_ubs.append(v1)

    edge_lbs = torch.stack(edge_lbs, dim=0)
    edge_ubs = torch.stack(edge_ubs, dim=0)
    return edge_lbs, edge_ubs


def flip_bit(v: int, d: int, i: int):
    """ Given an integer v (with d binary bits), flip its value on the i-th largest dimension (zero-based). """
    fmt = f'0{d}b'  # 'b' for binary numbers, 'd' for total amount of bits, '0' for adding 0s to fill the prefix
    vb = list(format(v, fmt))
    mapping = {
        '0': '1',
        '1': '0'
    }
    vb[i] = mapping[vb[i]]
    vb = ''.join(vb)
    return int(vb, 2)
