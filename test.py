from skimage.io import imread, imshow, imsave
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from bisect import bisect

from keras.utils import Sequence

from satellite.batch_manager import SingleImagePatchSequence


class PatchSequence(Sequence):
    @classmethod
    def from_path_list(cls, path_list, batch_size=16):
        return cls([
            SingleImagePatchSequence(path, batch_size=batch_size)
            for path in path_list
        ])

    def __init__(self, sub_sequences):
        self.sub_sequences = sub_sequences
        self.sub_lengths = np.array([len(sub) for sub in sub_sequences])
        self.cum_lengths = np.cumsum(self.sub_lengths)
        self.total_length = self.cum_lengths[-1]

        self.cum_lengths[-1] = 0  # so -1 gives correct answer

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        sub_idx = bisect(self.cum_lengths, idx)

        sub_sub_idx = idx - self.cum_lengths[sub_idx - 1]

        return self.sub_sequences[sub_idx][sub_sub_idx]

    def on_epoch_end(self):
        for sub in self.sub_sequences:
            sub.on_epoch_end()


if __name__ == "__main__":
    np.random.seed(0)
    parent_path = Path("data/parsed/")

    dataset_paths = []

    it = parent_path.iterdir()
    for _ in range(5):
        dataset_paths.append(next(it))

    seq = PatchSequence.from_path_list(dataset_paths, batch_size=16)

    print(len(seq))
    x, y = seq[15]
    print(x.shape)
