import numpy as np 

def get_features(dim_embedding, emb_dir="./data/embeddings/"):
    data = np.load(f"{emb_dir}embeddings_driving_{dim_embedding}d.npz", allow_pickle=True)
    return data['embeddings'], data['filenames']