import numpy as np
import pandas as pd


def load_data(dim_embedding, type):
    df = np.load(f"./data/embed_datasets/{dim_embedding}d_{type}.npy")
    df = df.T
    # Normalize each row to unit vectors
    norms = np.linalg.norm(df, axis=1, keepdims=True)
    # Avoid division by zero for any zero vectors
    norms = np.where(norms == 0, 1, norms)
    df = df / norms
    return df


def calculate_reward_alignment(estimated_preference, true_preference):
    return np.dot(estimated_preference, true_preference) / (np.linalg.norm(estimated_preference) * np.linalg.norm(true_preference))

def calculate_regret(estimated_preference, true_preference, dataset):
    true_utilities = dataset @ true_preference
    estimated_utilities = dataset @ estimated_preference

    true_best = np.max(true_utilities)
    estimated_best = np.max(true_utilities[np.argmax(estimated_utilities)])

    regret = (true_best - estimated_best) / true_best
    return regret

def calculate_query_qualities(true_preference, query):
    utilities = query @ true_preference
    return np.max(utilities), np.mean(utilities), np.median(utilities), np.min(utilities)


def get_features(dim_embedding, emb_dir="./data/embeddings/", type='pca'):
    data = np.load(f"{emb_dir}/{dim_embedding}d_{type}.npy")
    return data, None # TODO: list out the file names for each voice