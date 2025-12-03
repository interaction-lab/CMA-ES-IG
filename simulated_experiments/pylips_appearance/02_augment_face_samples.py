import numpy as np
import os
from utils import face_params_to_vector, hex_to_rgb, vector_to_face_params, rgb_to_hex
from pylips.face import FacePresets
import random

def augment_colors(face_vectors):
    """
    Augment face vectors by randomly shifting color channels.
    Assumes color channels are contiguous in the vector.
    """
    augmented = []
    for vec in face_vectors:
        for exemplar in [FacePresets.high_contrast, FacePresets.default, FacePresets.chili, FacePresets.gingerbreadman, FacePresets.cutie]:
            
            # augment the colors in parameter space (one at a time)
            vec_aug = vec.copy()
            params = vector_to_face_params(vec_aug)
            for color_key in ['background_color', 'eyeball_color', 'iris_color',
                              'eyelid_color', 'nose_color', 'mouth_color', 'brow_color']:
                params[color_key] = exemplar[color_key]
                # Add the augmented vector to the list
                vec_aug = face_params_to_vector(params)
                augmented.append(vec_aug)

        # add the original face as well
        augmented.append(vec)

    return np.array(augmented)

if __name__ == "__main__":
    # Load the approved faces dataset
    faces = np.load('data/approved_faces.npy')  # Assuming shape (num_samples, vector_dim)
    print(f"Loaded {faces.shape[0]} approved faces.")

    data = augment_colors(faces)
    np.save('data/augmented_faces.npy', data)
    print(f"Saved {data.shape[0]} augmented faces to 'data/augmented_faces.npy'")