from abs.deeppoly import Ele, ReLU

import numpy as np
import scipy.sparse as sp
import torch as th
from torch import nn


def normalize_A(adj):
    adj_norm = None
    N = adj.shape[0]
    if sp.isspmatrix(adj):
        adj_tilde = adj + sp.eye(N)
        degs_inv = np.power(adj_tilde.sum(0), -0.5)
        adj_norm = adj_tilde.multiply(degs_inv).multiply(degs_inv.T)
    elif isinstance(adj, np.ndarray):
        adj_tilde = adj + np.eye(N)
        degs_inv = np.power(adj_tilde.sum(0), -0.5)
        adj_norm = np.multiply(np.multiply(adj_tilde, degs_inv[None, :]), degs_inv[:, None])

    return adj_norm


def perturb(A_sub_hat, center_node, Q, q, dataset):
    # loc_pert = q * np.abs((np.mean(A_sub_hat[center_node].numpy()))) / np.exp(5)
    # glo_pert = (Q - q) * np.abs(np.mean((A_sub_hat + A_sub_hat @ A_sub_hat).numpy()) / np.exp(5))
    # # if dataset == "pubmed":
    # #     ep = (loc_pert + glo_pert) * 100
    # # elif dataset == "citeseer":
    # #     ep = (loc_pert + glo_pert) / 100
    # # else:
    #
    # ep = (loc_pert + glo_pert)

    if dataset == "citeseer":
        ep = q ** 5 / 10000
    elif dataset == "cora_ml":
        ep = q ** 1 / 100
    elif dataset == "pubmed":
        ep = q ** 5 / 1000

    return Ele.by_intvl(A_sub_hat, A_sub_hat + ep)


def gcn_forward(A_hat, X, weights, i=None):
    W1, b1, W2, b2 = weights

    W1 = th.from_numpy(W1)
    b1 = th.from_numpy(b1)
    W2 = th.from_numpy(W2)
    b2 = th.from_numpy(b2)

    relu = ReLU()
    logits = A_hat @ (relu(A_hat @ (X @ W1) + b1) @ W2) + b2

    if i is not None:
        logits = logits[i]

    return logits


two_hop = None


def two_hop_subgraph(A, X, z, i):
    global two_hop
    if two_hop is None:
        two_hop = A + A @ A
    ind = two_hop[i].nonzero()[1]

    center_node = np.where(ind == i)[0]
    if len(center_node) == 0:
        center_node = None
    else:
        center_node = center_node.item()

    return A[ind][:, ind], X[ind], z[ind], center_node


def cs(logits_lb, logits_ub, cls):
    cs_lb = logits_lb[cls] - np.max(np.concatenate([logits_ub[:cls], logits_ub[cls + 1:]]))
    cs_ub = logits_ub[cls] - np.min(np.concatenate([logits_lb[:cls], logits_lb[cls + 1:]]))

    return cs_lb, cs_ub


def certify(target_node,
            A,
            X,
            weight_list,
            z,
            local_changes,
            global_changes,
            dataset):
    A_sub, X_sub, z_sub, center_node_ind = two_hop_subgraph(A, X, z, target_node)
    if center_node_ind is None:
        # unsure
        return (-0.1, 0.1)

    A_sub_hat = normalize_A(A_sub)
    X_sub = th.from_numpy(X_sub.todense()).float()
    A_sub_hat = th.from_numpy(A_sub_hat.todense()).float()

    A_sub_hat_abs = perturb(A_sub_hat, center_node_ind, global_changes, local_changes, dataset)
    res = gcn_forward(A_sub_hat_abs, X_sub, weights=weight_list)
    lb = res.lb().numpy()[center_node_ind]
    ub = res.ub().numpy()[center_node_ind]

    return cs(lb, ub, z_sub[center_node_ind])


def _tc1():
    from robust_gcn_structure import loader
    from robust_gcn_structure.certification import gcn_forward as gf
    A, X, z = loader.load_dataset("citeseer")
    A_hat = normalize_A(A)
    weights = loader.load_network("citeseer")

    res = gf(A_hat, X, weights)
    print(np.sum(np.argmax(res, axis=1).squeeze() == z) / len(res))

    A_sub, X_sub, z_sub, i = two_hop_subgraph(A, X, z, 1995)
    print(i)
    A_sub_hat = normalize_A(A_sub)
    res = gf(A_sub_hat, X_sub, weights)
    print(np.sum(np.argmax(res, axis=1).squeeze() == z_sub) / len(res))


def _tc2():
    from robust_gcn_structure import loader
    A, X, z = loader.load_dataset("citeseer")
    weights = loader.load_network("citeseer")
    A_sub, X_sub, z_sub, center_node_ind = two_hop_subgraph(A, X, z, 789)

    A_sub_hat = normalize_A(A_sub)
    X_sub = th.from_numpy(X_sub.todense()).float()
    A_sub_hat = th.from_numpy(A_sub_hat.todense()).float()
    A_sub_hat = Ele.by_intvl(lb=A_sub_hat + 0.01, ub=A_sub_hat - 0.01)
    res = gcn_forward(A_sub_hat, X_sub, weights)
    lb = res.lb().numpy()[center_node_ind]
    ub = res.ub().numpy()[center_node_ind]
    print(lb.argmax())
    print(ub.argmax())
    print(z_sub[center_node_ind])

    print(cs(lb, ub, z_sub[center_node_ind]))


def _tc3():
    from robust_gcn_structure import loader
    A, X, z = loader.load_dataset("citeseer")
    weights = loader.load_network("citeseer")

    for i in range(50):
        # A_sub, X_sub, z_sub, center_node_ind = two_hop_subgraph(A, X, z, i)
        #
        # A_sub_hat = normalize_A(A_sub)
        # X_sub = th.from_numpy(X_sub.todense()).float()
        # A_sub_hat = th.from_numpy(A_sub_hat.todense()).float()
        #
        # A_sub_hat_abs = perturb(A_sub_hat, center_node_ind, 10, 5)
        # res = gcn_forward(A_sub_hat_abs, X_sub, weights)
        # lb = res.lb().numpy()[center_node_ind]
        # ub = res.ub().numpy()[center_node_ind]

        # print(cs(lb, ub, z_sub[center_node_ind]))

        print(certify(i, A, X, weights, z, 5, 10))


if __name__ == '__main__':
    _tc3()
