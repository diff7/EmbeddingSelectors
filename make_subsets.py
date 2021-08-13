import os

import numpy as np
from tqdm import tqdm

import apricot as apri

from acq.acq_funcs import CoreFuncs
from make_embeddings import get_embeddings_slow


def get_indices_FL(embeddings, size):
    a = apri.FacilityLocationSelection(size, "euclidean").fit(embeddings)
    return np.nonzero(a.mask)[0]


def get_indices_FS(embeddings, size):
    a = apri.FeatureBasedSelection(size, concave_func="sqrt").fit(embeddings)
    return np.nonzero(a.mask)[0]


def get_indices_GCUT(embeddings, size):
    a = apri.GraphCutSelection(size).fit(embeddings)
    return np.nonzero(a.mask)[0]


def get_indices_KC(embeddings, size):
    k_center_sampler = CoreFuncs(name="k_center", all_data_size=len(embeddings))
    return k_center_sampler.sample(size, embeddings)


def Facility_partial_fit(embeddings, size, split_k=10_000):
    a = apri.FacilityLocationSelection(size, max_reservoir_size=split_k)
    steps = len(embeddings)//split_k
    start = 0 
    for i in tqdm(range(steps)):
        a.partial_fit(embeddings[start:start+split_k])
        start+=split_k
        print(len((a.ranking)))
    return a.ranking

def GCUT_partial_fit(embeddings, size, split_k=10_000):
    a = apri.GraphCutSelection(size, max_reservoir_size=split_k)
    steps = len(embeddings)//split_k
    start = 0 
    for i in tqdm(range(steps)):
        a.partial_fit(embeddings[start:start+split_k])
        start+=split_k
        print(len((a.ranking)))
    return a.ranking


selectors = {
    "Fasilitylocation": Facility_partial_fit,
    "GraphCut": GCUT_partial_fit,
    #"FeatureSelection": get_indices_FS,
}


SUB_SIZE = 0.03

if __name__ == "__main__":
    FOLDER = "/home/dev/data_main/DIV2K/processed_50/LR/train/"
    files = sorted(
        [os.path.join(FOLDER, f) for f in os.listdir(FOLDER) if ".npy" not in f]
    )

    embeddings = get_embeddings_slow(
        model=None,
        data_txt_path=FOLDER,
        files=files,
        model_name="CIFAR_unsuper",
        resize=32,
        force_new=False,
    ).reshape(-1, 64)

    assert len(embeddings) == len(
        files
    ), "Embeddings size does not match file size"
    for name in selectors:
        print(f"SELECTING {name}")
        indices = selectors[name](embeddings, int(len(embeddings) * SUB_SIZE))

        with open(f"./subset_{name}.txt", "w") as f:
            for ind in indices:
                f.write(files[ind] + "\n")
