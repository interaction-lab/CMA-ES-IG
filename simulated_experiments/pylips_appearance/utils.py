import random
import numpy as np

def random_color():
    """Generate a random hex color."""
    return "#{:02x}{:02x}{:02x}".format(
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

def random_face_params():
    """Generate a dictionary of randomized face parameters."""
    return {
        'background_color': random_color(),
        'eyeball_color': random_color(),
        'iris_color': random_color(),
        'eye_size': random.randint(100, 250),             # px
        'eye_height': random.randint(0, 150),           # px
        'eye_separation': random.randint(300, 700),      # px
        'iris_size': random.randint(10, 120),            # px
        'pupil_scale': round(random.uniform(0.1, 1.0), 2),
        'eye_shine': random.choice([True, False]),
        'eyelid_color': random_color(),
        'nose_color': random_color(),
        'nose_vertical_position': random.randint(-50, 100),
        'nose_width': random.randint(0, 100),
        'nose_height': random.randint(0, 100),
        'mouth_color': random_color(),
        'mouth_width': random.randint(200, 500),
        'mouth_height': random.randint(10, 50),
        'mouth_y': random.randint(50, 200),
        'mouth_thickness': random.randint(5, 55),
        'brow_color': random_color(),
        'brow_width': random.randint(80, 150),
        'brow_height': random.randint(150, 250),
        'brow_thickness': random.randint(15, 50)
    }

def hex_to_rgb(hex_color):
    """Convert a hex color string (#RRGGBB) to an RGB list of floats [0,1]."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return [r, g, b]

def rgb_to_hex(rgb):
    """Convert RGB list of floats [0,1] back to hex string."""
    r, g, b = [int(round(x * 255)) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"

def face_params_to_vector(params):
    """
    Convert a face_params dict into a 1D vector.
    Colors -> 3 floats each (0-1)
    Booleans -> 0 or 1
    Numbers -> kept as-is
    """
    vector = []

    # Convert colors
    color_keys = ['background_color', 'eyeball_color', 'iris_color',
                  'eyelid_color', 'nose_color', 'mouth_color', 'brow_color']
    for key in color_keys:
        vector.extend(hex_to_rgb(params[key]))

    # Numeric parameters
    numeric_keys = ['eye_size', 'eye_height', 'eye_separation', 'iris_size',
                    'pupil_scale', 'nose_vertical_position', 'nose_width', 'nose_height',
                    'mouth_width', 'mouth_height', 'mouth_thickness',
                    'brow_width', 'brow_height', 'brow_thickness']
    for key in numeric_keys:
        vector.append(params[key])

    # Boolean parameters
    bool_keys = ['eye_shine']
    for key in bool_keys:
        vector.append(int(params[key]))

    return np.array(vector, dtype=np.float32)


def vector_to_face_params(vector):
    """
    Convert a vector back into a face_params dictionary.
    Assumes the same order as face_params_to_vector:
    colors (7*3), numeric (14), boolean (1)
    """
    params = {}
    vector = vector.flatten()
    idx = 0

    # Colors
    color_keys = ['background_color', 'eyeball_color', 'iris_color',
                  'eyelid_color', 'nose_color', 'mouth_color', 'brow_color']
    for key in color_keys:
        params[key] = rgb_to_hex(vector[idx:idx+3])
        idx += 3

    # Numeric
    numeric_keys = ['eye_size', 'eye_height', 'eye_separation', 'iris_size',
                    'pupil_scale', 'nose_vertical_position', 'nose_width', 'nose_height',
                    'mouth_width', 'mouth_height', 'mouth_thickness',
                    'brow_width', 'brow_height', 'brow_thickness']
    for key in numeric_keys:
        params[key] = float(vector[idx])
        idx += 1

    # Boolean
    bool_keys = ['eye_shine']
    for key in bool_keys:
        params[key] = bool(round(vector[idx]))
        idx += 1

    return params




def calculate_reward_alignment(estimated_preference, true_preference):
    return np.dot(estimated_preference, true_preference) / (np.linalg.norm(estimated_preference) * np.linalg.norm(true_preference))

def calculate_regret(estimated_preference, true_preference, dataset):
    true_utilities = dataset @ true_preference
    estimated_utilities = dataset @ estimated_preference

    true_best = np.max(true_utilities)
    true_worst = np.min(true_utilities)
    estimated_best = true_utilities[np.argmax(estimated_utilities)]
    

    regret = (true_best - estimated_best) / (true_best - true_worst)
    # print(true_best, estimated_best, true_worst, regret)
    return regret

def calculate_query_qualities(true_preference, query):
    utilities = [calculate_reward_alignment(q, true_preference) for q in query]
    # print(utilities)
    return np.max(utilities), np.mean(utilities), np.median(utilities), np.min(utilities)

from sklearn.preprocessing import minmax_scale

def get_features(num_features, emb_dir="./data/embeddings/"):
    face_info = np.load(open(f"{emb_dir}/augmented_faces.npy", "rb"))
    random_indices = np.random.choice(face_info.shape[1], size=num_features, replace=False)
    dataset = face_info[:, random_indices]
    
    # Scale each row (feature) to [-1, 1] range using sklearn
    dataset = minmax_scale(dataset, feature_range=(-1, 1), axis=1)
    
    return dataset, None
