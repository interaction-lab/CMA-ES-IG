"""
Download MNIST and build the latent space the explorer searches.

    uv run prepare.py

Fits PCA (up to 16 components) to the 60,000 training digits.  Each component is scaled so
that 98% of digits fall inside [-1, 1] on it, because every query generator searches the box
[-1, 1]^d.  Decoding a latent point is the inverse PCA transform.  Writes data/latent.npz.
"""

import gzip
import os
import urllib.request

import numpy as np
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
FILES = {'images': 'train-images-idx3-ubyte.gz', 'labels': 'train-labels-idx1-ubyte.gz'}
MAX_DIM = 16


def fetch(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print(f'downloading {name}')
        urllib.request.urlretrieve(URL + name, path)
    with gzip.open(path, 'rb') as f:
        return f.read()


def main():
    os.makedirs(DATA, exist_ok=True)
    raw = fetch(FILES['images'])
    images = np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 28 * 28).astype(np.float32) / 255
    labels = np.frombuffer(fetch(FILES['labels']), np.uint8, offset=8)
    pca = PCA(n_components=MAX_DIM, random_state=0).fit(images)
    z = pca.transform(images)
    scale = np.percentile(np.abs(z), 98, axis=0)          # per-component box scaling
    np.savez_compressed(os.path.join(DATA, 'latent.npz'),
                        mean=pca.mean_.astype(np.float32),
                        components=pca.components_.astype(np.float32),
                        scale=scale.astype(np.float32),
                        latents=(z / scale).astype(np.float32),
                        labels=labels,
                        explained=pca.explained_variance_ratio_.astype(np.float32))
    print(f'wrote data/latent.npz: {len(images)} digits, {MAX_DIM} components '
          f'({pca.explained_variance_ratio_.sum():.0%} of pixel variance)')


if __name__ == '__main__':
    main()
