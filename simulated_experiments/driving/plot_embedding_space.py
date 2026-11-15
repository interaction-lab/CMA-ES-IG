import numpy as np
data = np.load('./data/embeddings/embeddings_driving_2d.npz', allow_pickle=True)

import matplotlib.pyplot as plt
plt.plot(data['embeddings'][:,0], data['embeddings'][:,1], 'o')
plt.show()