from robust_gcn_structure.utils import load_npz
import torch

import os
ROOT = os.path.split(os.path.dirname(__file__))[0]


def load_dataset(dataset):
    A, X, z = load_npz(f'{ROOT}/datasets/{dataset}.npz')
    A = A + A.T
    A[A > 1] = 1
    A.setdiag(0)

    X = (X > 0).astype("float32")
    z = z.astype("int64")

    return A, X, z


def load_network(dataset, robust_gcn=False):
    weight_path = f"{ROOT}/pretrained_weights/{dataset}"
    if robust_gcn:
        weight_path = f"{weight_path}_robust_gcn.pkl"
    else:
        weight_path = f"{weight_path}_gcn.pkl"

    state_dict = torch.load(weight_path, map_location="cpu")

    weights = [v for k, v in state_dict.items() if "weight" in k and "conv" in k]
    biases = [v for k, v in state_dict.items() if "bias" in k and "conv" in k]

    W1, W2 = [w.cpu().detach().numpy() for w in weights]
    b1, b2 = [b.cpu().detach().numpy() for b in biases]

    shapes = [x.shape[0] for x in biases]
    num_hidden = len(shapes) - 1
    if num_hidden > 1:
        raise NotImplementedError("Only one hidden layer is supported.")

    weight_list = [W1, b1, W2, b2]

    return weight_list


if __name__ == '__main__':
    load_dataset("citeseer")
    load_network("citeseer")
