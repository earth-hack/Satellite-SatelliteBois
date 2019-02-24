from bisect import bisect
import numpy as np

from keras.utils import Sequence, to_categorical


def pad(img, patch_shape, value=255):
    new_shape = (img.shape[0] + (-img.shape[0] % patch_shape[0]),
                 img.shape[1] + (-img.shape[1] % patch_shape[1]))
    new_shape = np.array(new_shape)

    pad_width = [[e // 2 + e % 2, e // 2] for e in new_shape - img.shape]

    return np.pad(img, pad_width, "constant", constant_values=value)


def preprocess_img(img, patch_shape, stride_shape):
    assert patch_shape == stride_shape
    # img = pad(img, new_shape, 255) # white borders
    img = pad(img, patch_shape, 1)  # black borders
    img = img / 255.
    return img


def preprocess_label(label, patch_shape, stride_shape):
    assert patch_shape == stride_shape
    label = pad(label, patch_shape, 0)
    label = np.clip(label, 0, 1)
    label = to_categorical(label, 2)
    return label


class SingleImagePatchSequence(Sequence):
    def __init__(self,
                 path,
                 batch_size,
                 patch_shape=(256, 256),
                 randomize=True):
        self.batch_size = batch_size
        self.patch_shape = patch_shape
        self.stride_shape = patch_shape

        self.randomize = randomize

        img = np.load(path / "img.npy")
        self.img = preprocess_img(img, patch_shape, patch_shape)

        label = np.load(path / "label.npy")
        self.label = preprocess_label(label, patch_shape, patch_shape)

        self.n_rows = self.img.shape[0] // patch_shape[0]
        self.n_cols = self.img.shape[1] // patch_shape[1]

        self.N = self.n_rows * self.n_cols

        self.idxs = np.arange(self.N)

        self.on_epoch_end()

        self.mapping = []
        k = 0
        for i in range(len(self)):
            _, y = self.get_absolute(i)

            if y.argmax(3).sum() > 0:
                self.mapping.append(i)

        if not self.mapping:
            self.N = 0
        else:
            self.N = len(self.mapping)

    def __len__(self):
        return (self.N + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        idx = self.mapping[idx]
        return self.get_absolute(idx)

    def get_absolute(self, idx):
        img_lis = []
        label_lis = []

        idx *= self.batch_size
        idxs = self.idxs[idx:idx + self.batch_size]

        i, j = np.divmod(idxs, self.n_cols)
        i *= self.patch_shape[0]
        j *= self.patch_shape[1]

        for k in range(idxs.shape[0]):
            img_lis.append(self.img[i[k]:i[k] + self.patch_shape[0], \
                j[k]:j[k] + self.patch_shape[1]])

            label_lis.append(self.label[i[k]:i[k] + self.patch_shape[0], \
                j[k]:j[k] + self.patch_shape[1]])

        return np.array(img_lis)[..., np.newaxis], np.array(
            label_lis)[..., np.newaxis]

    def on_epoch_end(self):
        if self.randomize:
            np.random.shuffle(self.idxs)


class PatchSequence(Sequence):
    @classmethod
    def from_path_list(cls, path_list, batch_size=16):
        return cls([
            SingleImagePatchSequence(path, batch_size=batch_size)
            for path in path_list
        ])

    def __init__(self, sub_sequences):
        self.sub_sequences = []
        for seq in sub_sequences:
            if len(seq) != 0:
                self.sub_sequences.append(seq)

        self.sub_lengths = np.array([len(sub) for sub in self.sub_sequences])
        self.cum_lengths = np.cumsum(self.sub_lengths)
        self.total_length = self.cum_lengths[-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        sub_idx = bisect(self.cum_lengths, idx)

        if sub_idx == 0:
            sub_sub_idx = idx
        else:
            sub_sub_idx = idx - self.cum_lengths[sub_idx - 1]

        return self.sub_sequences[sub_idx][sub_sub_idx]

    def on_epoch_end(self):
        for sub in self.sub_sequences:
            sub.on_epoch_end()
