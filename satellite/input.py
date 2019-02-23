from skimage.external.tifffile import imread
from pathlib import Path
import matplotlib.pyplot as plt  # noqa
from matplotlib.gridspec import GridSpec  # noqa
import numpy as np
from itertools import product
import re

from skimage import transform

import np_utils


def load_label(fname):
    '''
    extract features from a label directory.

    returns (img, coords, resolution)

    '''

    fname = str(fname)

    metadata = Path(f'{fname}.txg').read_text()
    digit = r'\d+.?\d+'

    coords = {}
    for k in [
            ' '.join(x) for x in product(['Upper', 'Lower'], ['Left', 'Right'])
    ]:
        pattern = f'{k}:\s+({digit})\s+E\s+({digit}) N'  # noqa
        g = re.search(pattern, metadata)
        x, y = (float(g.groups()[0]), float(g.groups()[1]))
        coords[k] = np.array((x, y))

    pattern = f'Resolution: ({digit}) .* ({digit}) .*'
    x, y = re.search(pattern, metadata).groups()
    resolution = np.array((float(x), float(y)))

    img = imread(f'{fname}.tif')

    return img, resolution, coords


def resize_and_localize_patch(satelite_feat, feat):
    '''
    given a feature, will resize the feature image and return a location on
    where it will be located.

    returns:
        transformed: np.array of shape (H, W)
        offsets: tuple, of shape (o_h, o_w)
    '''
    img, res, coords = satelite_feat
    p_img, p_res, pcoords = feat

    D = {k: (v - coords['Lower Left']) / res for k, v in coords.items()}
    E = {k: (v - coords['Lower Left']) / res for k, v in pcoords.items()}

    x = D['Lower Right'] - D['Lower Left']
    t = -np.arctan(x[0] / x[1])

    M = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    D = {k: np.matmul(M, v[::-1]) for k, v in D.items()}
    E = {k: np.matmul(M, v[::-1]) for k, v in E.items()}

    X = np.vstack(list(D.values()))
    offset = X.min(axis=0)
    D = {k: v - offset for k, v in D.items()}
    E = {k: v - offset for k, v in E.items()}

    x = np.vstack(list(E.values()))
    offset = x.min(axis=0)

    bottom_left = x[x.sum(axis=1).argmin()]
    top_right = x[x.sum(axis=1).argmax()]

    offset = np_utils.iceil(bottom_left)[::-1]  # height first.
    patch_size = top_right - bottom_left

    transformed = transform.resize(p_img, np_utils.iceil(patch_size))

    return transformed, offset


def get_image_and_labels(satelite_feat, patch_features):
    '''
    Given a satelite feature and patch features, will do proper resizing
    and stuff to generate an image with patches.

    returns
        img: np.array of shape (H, W)
        mask: np.array of shape (H, W) and dtype np.int32.
    '''
    img, res, coords = satelite_feat
    mask = np.zeros_like(img, dtype=np.int32)

    for i, (p_img, p_res, p_coords) in enumerate(patch_features, 1):
        feat = (p_img, p_res, p_coords)

        # TODO: add check on rotation here.
        patch_mask, (y0, x0) = resize_and_localize_patch(satelite_feat, feat)
        y1, x1 = y0 + patch_mask.shape[0], x0 + patch_mask.shape[1]
        try:
            mask[y0:y1, x0:x1] = np.ones_like(patch_mask) * i
        except ValueError:
            print('Out of bounds. Skipping')  # let's be pragmatic.

    return img, mask


def parse_dataset():
    parsed_data_dir = Path('../data/parsed')
    parsed_data_dir.mkdir(parents=True, exist_ok=True)
    for p in Path('../data/ql8tiff').iterdir():

        satelite_feat = load_label(p.joinpath('_' + p.stem + 'ql8'))

        q = Path('../data/sx8tiff').joinpath(p.parts[-1])

        patches = []
        for each in q.glob('*.tif'):
            each = each.with_suffix('')  # remove suffix
            patches.append(load_label(each))

        # is_inside(satelite_feat, patches[-1])
        img, labels = get_image_and_labels(satelite_feat, patches)

        parsed_data_dir.joinpath(p.stem).mkdir(exist_ok=True)
        np.save(parsed_data_dir.joinpath(p.stem, 'img.npy'), img)
        np.save(parsed_data_dir.joinpath(p.stem, 'label.npy'), labels)


parse_dataset()
