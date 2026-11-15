from pylips.speech import RobotFace
from pylips.face import FacePresets
import random
import numpy as np
from utils import random_color, random_face_params, hex_to_rgb, rgb_to_hex, face_params_to_vector, vector_to_face_params

def sample_faces_interactively(robot_face, num_samples=10, save_path="approved_faces.npy"):
    """
    Loop to generate random faces, ask for approval, and save approved ones to a NumPy array.

    Args:
        num_samples (int): Number of approved faces to collect.
        save_path (str): File path to save the approved face vectors.
    """
    # Check if file already exists and load previous faces
    import os
    if os.path.exists(save_path):
        existing_faces = np.load(save_path)
        approved_vectors = existing_faces.tolist()
        print(f"Loaded {len(approved_vectors)} existing faces from '{save_path}'")
    else:
        approved_vectors = []

    for i in range(num_samples):
        # Generate random face parameters
        face_params = random_face_params()
        robot_face.set_appearance(face_params)

        # Convert to vector
        face_vector = face_params_to_vector(face_params)

        # Optionally, print the parameters for review
        print("\nGenerated Face Parameters:")
        for k, v in face_params.items():
            print(f"{k}: {v}")

        # Ask user if this face is reasonable
        response = input("\nIs this face reasonable? (y/n): ").strip().lower()
        if response == 'y':
            approved_vectors.append(face_vector)
            print(f"Face accepted! Total approved: {len(approved_vectors)}")
        else:
            print("Face rejected.")

    # Convert to numpy array and save
    approved_array = np.stack(approved_vectors, axis=0)
    np.save(save_path, approved_array)
    print(f"\nSaved {len(approved_vectors)} total faces to '{save_path}' (including {len(approved_vectors) - (len(existing_faces) if 'existing_faces' in locals() else 0)} new faces)")
    return approved_array


# Example usage:
if __name__ == "__main__":
    face = RobotFace()
    approved_faces = sample_faces_interactively(face, num_samples=50, save_path="data/approved_faces.npy")
    print("Shape of approved faces array:", approved_faces.shape)

