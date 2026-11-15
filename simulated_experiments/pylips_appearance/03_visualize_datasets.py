import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    face_info = np.load(open("./data/augmented_faces.npy", "rb"))
    random_indices = np.random.choice(face_info.shape[1], size=2, replace=False)
    dataset = face_info[:, random_indices]
    print(dataset.shape, random_indices)  # (num_faces, embedding_dim)
    plt.scatter(dataset[:, 0], dataset[:, 1])
    plt.show()