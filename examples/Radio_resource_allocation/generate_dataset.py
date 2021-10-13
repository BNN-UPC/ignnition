"""Radio Resource Allocation dataset obtention and processing.

This script generates the radio resource management synthetic dataset into Networkx graphs, to use
in iGNNition example.
For the script to work, the following additional packages are needed:
  - tqdm -> https://github.com/tqdm/tqdm

References:
 - Y. Shen, Y. Shi, J. Zhang and K. B. Letaief,
   "Graph Neural Networks for Scalable Radio Resource Management: Architecture Design and
   Theoretical Analysis", in IEEE Journal on Selected Areas in Communications, vol. 39, no. 1,
   pp. 101-115, Jan. 2021 doi: 10.1109/JSAC.2020.3036965.
 - Y. Shi, J. Zhang and K. B. Letaief,
   "Group Sparse Beamforming for Green Cloud-RAN," in IEEE Transactions on Wireless Communications,
   vol. 13, no. 5, pp. 2809-2823, May 2014, doi: 10.1109/TWC.2014.040214.131770.
"""
import json
import math
import networkx as nx
import numpy as np
import os
import shutil
from itertools import product
from pathlib import Path
from tqdm.auto import tqdm

# Generation options
empty_dirs = True
random_seed = 20210205
root_path = Path(__file__).parent
raw_dir = root_path / Path("data/raw")
train_dir = root_path / Path("data/train")
train_samples = 1000
validation_dir = root_path / Path("data/validation")
validation_samples = 100
# Computed
total_samples = train_samples + validation_samples
rng = np.random.default_rng(seed=random_seed)

# Dataset options, please see the referenced papers for more details
n_links = 10
field_length = 1000
shortest_directLink_length = 2
longest_directLink_length = 65
shortest_crossLink_length = 1
carrier_f = 2.4e9
tx_height = 1.5
rx_height = 1.5
# Channel loss options
signal_lambda = 2.998e8 / carrier_f
Rbp = 4 * tx_height * rx_height / signal_lambda
Lbp = abs(20 * np.log10(np.power(signal_lambda, 2) / (8 * np.pi * tx_height * rx_height)))
antenna_gain_decibel = 2.5
# Network additional inputs
noise_power = 4e-12


def _empty_dirs(dirs=None):
    if dirs is None:
        return
    elif isinstance(dirs, (Path, str)):
        dirs = [Path(dirs)]
    for _dir in dirs:
        assert isinstance(_dir, Path)
        for file in [f for f in _dir.glob("*") if f.is_file()]:
            file.unlink()


def compute_losses(distances, add_shadowing=True, add_fast_fading=True):
    N = np.shape(distances)[-1]
    assert N == n_links
    # compute coefficient matrix for each Tx/Rx pair
    sum_term = 20 * np.log10(distances / Rbp)
    # adjust for longer path loss
    Tx_over_Rx = Lbp + 6 + sum_term + ((distances > Rbp).astype(int)) * sum_term
    # only add antenna gain for direct channel
    path_losses = -Tx_over_Rx + np.eye(N) * antenna_gain_decibel
    path_losses = np.power(10, (path_losses / 10))  # convert from decibel to absolute
    # Compute channel losses, if specified
    channel_losses = np.copy(path_losses)
    if add_shadowing:
        shadow_coefficients = rng.normal(loc=0, scale=8, size=np.shape(channel_losses))
        channel_losses = channel_losses * np.power(10.0, shadow_coefficients / 10)
    if add_fast_fading:
        fast_fadings = (
            np.power(rng.normal(loc=0, scale=1, size=np.shape(channel_losses)), 2) +
            np.power(rng.normal(loc=0, scale=1, size=np.shape(channel_losses)), 2)
        ) / 2
        channel_losses = channel_losses * fast_fadings
    # Join non-diagonal path with diagonal channel losses
    mask = np.eye(N)
    off_diag_path = path_losses - np.multiply(mask, path_losses)
    diag_channel = np.multiply(mask, channel_losses)
    path_losses = diag_channel + off_diag_path
    return path_losses, channel_losses


def generate_layout():
    """Generate a single graph."""
    N = n_links
    # first, generate transmitters' coordinates
    tx_xs = rng.uniform(low=0, high=field_length, size=[N, 1])
    tx_ys = rng.uniform(low=0, high=field_length, size=[N, 1])
    while True:  # loop until a valid layout generated
        # generate rx one by one rather than N together to ensure checking validity one by one
        rx_xs = []
        rx_ys = []
        for i in range(N):
            got_valid_rx = False
            while not got_valid_rx:
                pair_dist = rng.uniform(
                    low=shortest_directLink_length,
                    high=longest_directLink_length,
                )
                pair_angles = rng.uniform(low=0, high=np.pi * 2)
                rx_x = tx_xs[i] + pair_dist * np.cos(pair_angles)
                rx_y = tx_ys[i] + pair_dist * np.sin(pair_angles)
                if (
                    0 <= rx_x <= field_length
                    and 0 <= rx_y <= field_length
                ):
                    got_valid_rx = True
            rx_xs.append(rx_x)
            rx_ys.append(rx_y)
        # For now, assuming equal weights and equal power, so not generating them
        layout = np.concatenate((tx_xs, tx_ys, rx_xs, rx_ys), axis=1)
        distances = np.zeros([N, N])
        # compute distance between every possible Tx/Rx pair
        for rx_index in range(N):
            for tx_index in range(N):
                tx_coor = layout[tx_index][0:2]
                rx_coor = layout[rx_index][2:4]
                # according to paper notation convention
                # Hij is from jth transmitter to ith receiver
                distances[rx_index][tx_index] = np.linalg.norm(tx_coor - rx_coor)
        # Check whether a tx-rx link (potentially cross-link) is too close
        if np.min(distances) > shortest_crossLink_length:
            break
    return layout, distances


def generate_graphs(output_dir="data/raw", output_prefix="network", empty_dirs=False):
    """Generate all graphs for the dataset.

    This method generates Networkx graphs for the Radio Resource Allocation problem, from the module
    options, and saves them to the specified directory as JSON file, which will then be stacked to
    form the dataset.
    """
    if empty_dirs:
        _empty_dirs(output_dir)
    N = n_links
    print(f"Generating {total_samples} network graphs in {output_dir}.")
    for i in tqdm(range(total_samples)):
        layout, dist = generate_layout()
        path_loss, channel_loss = compute_losses(dist)
        assert np.shape(layout) == (N, 4)
        assert np.shape(dist) == np.shape(path_loss) == np.shape(channel_loss) == (N, N)
        diag_dist = np.diag(1/dist)
        diag_channel_loss = np.diag(channel_loss)
        adjacency = channel_loss - np.multiply(np.eye(N), channel_loss)  # Remove own pair
        weights = rng.uniform(size=N)  # Transceiver-receiver random weights
        weights = weights / weights.sum()  # Normalize weights
        wmmse_power = get_wmmse_power(channel_loss, noise_power)
        graph = nx.DiGraph()
        # We add as node attribute a placeholder per node power label to use as target
        # altough the loss we plan to use will be self-supervised.
        graph.add_nodes_from([
            (link_idx, {
                "entity": "transmitter_receiver_pair",
                "transceiver_x": layout[link_idx, 0],
                "transceiver_y": layout[link_idx, 1],
                "receiver_x": layout[link_idx, 2],
                "receiver_y": layout[link_idx, 3],
                "receiver_distance": diag_dist[link_idx],
                "channel_loss": diag_channel_loss[link_idx],
                "path_loss": path_loss[:, link_idx].tolist(),
                "power": 0,
                "weights": weights[link_idx],
                "wmmse_power": wmmse_power[link_idx],
            })
            for link_idx in range(N)
        ])
        graph.add_edges_from([
            (src, dst, {
                "transceiver_receiver_loss": adjacency[src, dst]
            })
            for src, dst in product(range(N), range(N)) if src != dst
        ])
        graph.graph["noise_power"] = noise_power
        filepath = Path(output_dir) / f"{output_prefix}_{i}.json"
        with filepath.open("w") as _f:
            json.dump(nx.readwrite.json_graph.node_link_data(graph), _f)
    print(f"Finished generating {total_samples} network graphs in {output_dir}.")


def get_wmmse_power(channel_loss, noise_power, max_iterations=100):
    """Get WMMSE optimimum power aproximation for given matrix of channel losses and noise power."""
    H = channel_loss
    K = np.shape(channel_loss)[0]
    P_ini = np.random.rand(K, 1)
    P_max = np.ones([K, 1])
    vnew = 0
    b = np.sqrt(P_ini)
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.matmul(np.square(H[i, :]), np.square(b)) + noise_power)
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew += math.log2(w[i])

    for _ in range(max_iterations):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(P_max)[i]) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / (np.matmul(np.square(H[i, :]), np.square(b)) + noise_power)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            vnew += math.log2(w[i])

        if vnew - vold <= 1e-3:
            break

    return np.round(np.squeeze(np.square(b)), 10)


def join_graphs_into_dataset(files, output_dir, output_file_name="data.json", empty_dirs=False):
    if empty_dirs:
        _empty_dirs(output_dir)
    graphs = [json.load(open(file, "r")) for file in files]
    with open(output_dir / output_file_name, "w") as fp:
        json.dump(graphs, fp)


def split_train_validation(
    raw_dir, train_dir, validation_dir, train_samples, validation_samples, empty_dirs=False
):
    if empty_dirs:
        _empty_dirs([train_dir, validation_dir])
    files = np.array(list(Path(raw_dir).glob("*.json")))
    assert (
        files.shape[0] >= train_samples + validation_samples
    ), "Train + Validation samples exceed number of files available."
    rng.shuffle(files)
    training_files = files[validation_samples:(train_samples + validation_samples)]
    validation_files = files[:validation_samples]
    print(f"Copying training graphs into {raw_dir / 'train'}")
    for file in training_files:
        shutil.copy(file, raw_dir / "train")
    print(f"Joining training graphs into {train_dir}")
    join_graphs_into_dataset(training_files, output_dir=train_dir)
    print(f"Copying validation graphs into {raw_dir / 'validation'}")
    for file in validation_files:
        shutil.copy(file, raw_dir / "validation")
    print(f"Joining validation graphs into {validation_dir}")
    join_graphs_into_dataset(validation_files, output_dir=validation_dir)


if __name__ == "__main__":
    for _dir in [raw_dir, train_dir, validation_dir, raw_dir / "train", raw_dir / "validation"]:
        os.makedirs(_dir, exist_ok=True)
    generate_graphs(output_dir=raw_dir, empty_dirs=empty_dirs)
    split_train_validation(
        raw_dir=raw_dir,
        train_dir=train_dir,
        validation_dir=validation_dir,
        train_samples=train_samples,
        validation_samples=validation_samples,
        empty_dirs=empty_dirs,
    )
