# QM9 dataset obtention and processing.
#
# This script downloads and transforms the molecules in the QM9 dataset into Networkx graphs, to use
# in iGNNition example.
# For the data transformation to work, additional cheminformatics packages are needed, such as:
# - pysmiles -> https://pypi.org/project/pysmiles/
# - rdkit -> https://www.rdkit.org/docs/GettingStartedInPython.html
#
# References:
# - L. Ruddigkeit, R. van Deursen, L. C. Blum, J.-L. Reymond,
#   Enumeration of 166 billion organic small molecules in the chemical universe database GDB-17,
#   J. Chem. Inf. Model. 52, 2864–2875, 2012.
# - R. Ramakrishnan, P. O. Dral, M. Rupp, O. A. von Lilienfeld,
#   Quantum chemistry structures and properties of 134 kilo molecules,
#   Scientific Data 1, 140022, 2014.
import json
import networkx as nx
import numpy as np
import os
import pysmiles
import requests
import shutil
import tarfile
import tempfile
from pathlib import Path
from rdkit import Chem

# Parameters to determine dataset generation
empty_dirs = True
limit = 2000  # Limit number of files to take from dataset. None to take all
qm9_url = "https://s3-eu-west-1.amazonaws.com/pstorage-npg-968563215/3195389/dsgdb9nsd.xyz.tar.bz2"
random_seed = 42
raw_dir = Path("data/raw")
train_dir = Path("data/train")
train_samples = 1000
validation_dir = Path("data/validation")
validation_samples = 100


def _empty_dirs(dirs=None):
    if dirs is None:
        return
    elif isinstance(dirs, (Path, str)):
        dirs = [Path(dirs)]
    for _dir in dirs:
        assert isinstance(_dir, Path)
        for file in [f for f in _dir.glob("*") if f.is_file()]:
            file.unlink()


def get_graph_from_molecule(molecule):
    # Parse molecule file
    na = int(molecule[0][0])
    coordinates = [
        [c[0], float(c[1].replace("*^", "e")), float(c[2].replace("*^", "e")),
            float(c[3].replace("*^", "e")), float(c[4].replace("*^", "e"))]
        for c in molecule[2:(na+2)]
    ]
    properties = dict(zip(
        [
            "id", "rotational_a", "rotational_b", "rotational_c", "dipole_moment", "polarizability",
            "homo_energy", "lumo_energy", "spatial_extent", "internal_energy_0k",
            "internal_energy_298k", "free_energy", "heat_capacity"],
        [float(e.replace("gdb", "").strip()) for e in molecule[1][:-1]]
    ))
    smiles = molecule[na+3][0]
    graph = pysmiles.read_smiles(smiles, explicit_hydrogen=True)
    mol = Chem.MolFromSmiles(smiles)
    # One-hot encode element
    nx.set_node_attributes(graph, {
        k: {
            "entity": "atom",
            "element_c": int(d["element"] == "C"),
            "element_f": int(d["element"] == "F"),
            "element_h": int(d["element"] == "H"),
            "element_n": int(d["element"] == "N"),
            "element_o": int(d["element"] == "O"),
            "acceptor": int(d["charge"] > 0),
            "donor": int(d["charge"] < 0),
        }
        for k, d in dict(graph.nodes(data=True)).items()
    })
    # Add Chem molecule attributes
    hybridizations = ["SP", "SP2", "SP3"]
    nx.set_node_attributes(graph, {
        **{
            atom.GetIdx(): {
                "aromatic": int(atom.GetIsAromatic()),
                "atomic_number": atom.GetAtomicNum(),
                "hybridization_null": int(str(atom.GetHybridization()) not in hybridizations),
                "hybridization_sp": int(str(atom.GetHybridization()) == hybridizations[0]),
                "hybridization_sp2": int(str(atom.GetHybridization()) == hybridizations[1]),
                "hybridization_sp3": int(str(atom.GetHybridization()) == hybridizations[2]),
                "hydrogen_count": atom.GetNumImplicitHs(),
            }
            for atom in mol.GetAtoms()
        }, **{
            k: {
                "aromatic": 0,
                "atomic_number": 1,
                "hybridization_null": 1,
                "hybridization_sp": 0,
                "hybridization_sp2": 0,
                "hybridization_sp3": 0,
                "hydrogen_count": 0
            }
            for k in range(mol.GetNumAtoms(), graph.number_of_nodes())
        }
    })
    # Set edge attributes
    nx.set_edge_attributes(graph, {
        (src, tgt): {
            "distance": np.sqrt(np.sum(np.square(
                np.array(coordinates[tgt][1:]) - np.array(coordinates[src][1:])
            ))),
            "order_1": int(d["order"] == 1),
            "order_1_5": int(d["order"] == 1.5),
            "order_2": int(d["order"] == 2),
            "order_3": int(d["order"] == 3),
        }
        for src, tgt, d in list(graph.edges(data=True))
    })
    # Add graph level targets
    for key in [k for k in properties if k != "id"]:
        graph.graph[key] = properties[key]
    # Turn into directed graph
    digraph = nx.DiGraph(graph)
    return digraph


def join_graphs_into_dataset(files, output_dir, output_file_name="data.json", empty_dirs=False):
    if empty_dirs:
        _empty_dirs(output_dir)
    graphs = [json.load(open(file, "r")) for file in files]
    with open(output_dir / output_file_name, "w") as fp:
        json.dump(graphs, fp)


def qm9_download_and_extract(
    url, empty_dirs=False, limit=None, output_dir="data/raw", output_prefix="mol", process_func=None
):
    """Download QM9 to temporary file and extract it to data/raw folder"""
    if process_func is None:
        process_func = get_graph_from_molecule
    if empty_dirs:
        _empty_dirs(output_dir)
    with tempfile.TemporaryFile() as fp:
        print("Downloading tar file containing molecules...")
        r = requests.get(url, allow_redirects=True)
        fp.write(r.content)
        fp.seek(0)
        tar = tarfile.open(fileobj=fp)
        elem = tar.next()
        i = 0
        print(f"Extracting & transforming molecule files to {output_dir}...")
        while(elem is not None and limit is not None and i < limit):
            file = tar.extractfile(elem)
            molecule = [l.split("\t") for l in file.read().decode("utf-8").split("\n")]
            graph = process_func(molecule)
            filepath = Path(output_dir) / f"{output_prefix}_{i}.json"
            with filepath.open("w") as _f:
                json.dump(nx.readwrite.json_graph.node_link_data(graph), _f)
            elem = tar.next()
            i += 1


def split_traing_validation(
    raw_dir, train_dir, validation_dir, train_samples, validation_samples, empty_dirs=False
):
    if empty_dirs:
        _empty_dirs([train_dir, validation_dir])
    files = np.array(list(Path(raw_dir).glob("*.json")))
    assert files.shape[0] > train_samples+validation_samples, \
        "Train + Validation samples exceed number of files available."
    np.random.shuffle(files)
    training_files = files[validation_samples:(train_samples + validation_samples)]
    validation_files = files[:validation_samples]
    print(f"Copying training graphs into {raw_dir / 'traing'}")
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
    np.random.seed(random_seed)
    for _dir in [raw_dir, train_dir, validation_dir, raw_dir / "train", raw_dir / "validation"]:
        os.makedirs(_dir, exist_ok=True)
    qm9_download_and_extract(url=qm9_url, limit=limit, output_dir=raw_dir, empty_dirs=empty_dirs)
    split_traing_validation(
        raw_dir=raw_dir, train_dir=train_dir, validation_dir=validation_dir,
        train_samples=train_samples, validation_samples=validation_samples, empty_dirs=empty_dirs
    )
