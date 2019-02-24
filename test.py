from skimage.io import imread, imshow, imsave
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from satellite.batch_manager import PatchSequence

if __name__ == "__main__":
    np.random.seed(0)
    parent_path = Path("data/parsed/")

    dataset_paths = []

    it = parent_path.iterdir()
    for _ in range(10):
        dataset_paths.append(next(it))

    seq = PatchSequence.from_path_list(dataset_paths, batch_size=16)
    #print(len(seq.sub_sequences))

    print(seq.cum_lengths)
    print(seq.sub_lengths)
    for i, e in enumerate(seq):
        for j, s in enumerate(e):
            print(i, j)
