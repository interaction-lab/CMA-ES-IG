import pickle
import numpy as np

# Dimensionality reduction libraries
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import KernelPCA
import umap


def build_embedding_matrix(embedding_dict):
    """
    Given a dictionary {filename: embedding}, create:
      - mapping dictionary {filename: row_index}
      - stacked embedding matrix [num_speakers x embedding_dim]
    
    Args:
        embedding_dict (dict[str, np.ndarray]): 
            Keys are filenames, values are 1D numpy arrays (embeddings).

    Returns:
        mapping (dict[str, int]): filename -> row index
        matrix (np.ndarray): [num_speakers x embedding_dim]
    """
    mapping = {}
    embeddings = []

    for idx, (fname, emb) in enumerate(embedding_dict.items()):
        mapping[idx] = fname
        embeddings.append(np.array(emb, dtype=np.float32))

    matrix = np.stack(embeddings, axis=0)
    return mapping, matrix

def reduce_embeddings(matrix, reducer="pca", target_dim=8, **kwargs):
    """
    Reduce embeddings using a specified dimensionality reduction technique.

    Args:
        matrix (np.ndarray): [num_speakers, num_dims]
        reducer (str): which reduction method to use ("pca", "tsne", "umap")
        target_dim (int): reduced dimensionality
        **kwargs: extra keyword args passed to the reducer

    Returns:
        reduced_matrix (np.ndarray): [num_speakers, target_dim]
        model: fitted reduction model (so you can reuse/inspect it)
    """
    if reducer.lower() == "pca":
        model = PCA(n_components=target_dim, **kwargs)
    elif reducer.lower() == "tsne":
        model = TSNE(n_components=target_dim, **kwargs)
    elif reducer.lower() == "umap":
        model = umap.UMAP(n_components=target_dim, **kwargs)
    elif reducer.lower() == "kernelpca":
        model = KernelPCA(n_components=target_dim, **kwargs)
    elif reducer.lower() == "randomproj":
        model = GaussianRandomProjection(n_components=target_dim, **kwargs)

    reduced_matrix = model.fit_transform(matrix)
    return reduced_matrix, model


if __name__ == "__main__":
    speaker_info = pickle.load(open("./data/speaker_info.pkl", "rb"))
    # gpt_cond_latents = {k: v[0].squeeze().cpu().numpy() for k, v in speaker_info.items()}
    speaker_embeddings = {k: v[1].squeeze().cpu().numpy() for k, v in speaker_info.items()}

    spk_mapping, spk_matrix = build_embedding_matrix(speaker_embeddings)
    np.savez("./data/full_speaker_embeddings.npz", mapping=spk_mapping, matrix=spk_matrix)

    for reducer in ["pca", "umap", "kernelpca", "randomproj"]:
        for dim in [2, 4, 6, 8]:
            print(f"Reducing to {dim}d using {reducer}")
            reduced_matrix = reduce_embeddings(spk_matrix, reducer=reducer, target_dim=dim)[0]
            np.save(f"./data/embed_datasets/{dim}d_{reducer}.npy", reduced_matrix)

