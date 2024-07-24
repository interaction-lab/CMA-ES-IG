import os
import json
import numpy as np
import pickle
from interpolator import create_linear_pose_interpolator, b_spline_interpolation, b_spline_from_control_points,linear_interpolation_from_control_points

def get_all_subfiles_os(directory):
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    return all_files


def convert_file_to_array(file):
    with open(files[4], 'r') as f:
        data = json.load(f)
    
    poses = []
    times = []

    for frame in data['frame_list']:
        if frame['millis'] > 4500:
            break
        times.append(frame['millis']/1000)
        
        pose = {}
        for position in frame['positions']:
            dof = position['dof']
            pos = (float(position['pos']) - 3) * 50
            if dof in ['base', 'tower_1', 'tower_2', 'tower_3']:
                pose[dof] = pos

        poses.append([pose['base'], pose['tower_1'], pose['tower_2'], pose['tower_3']])
    
    if len(times) > 10:

        times = times[::4]
        poses = poses[::4]

    times[0] = 0
    poses[0] = [0,70,50,50]

    times.append(5)
    poses.append([0,70,50,50])

    # print(times, poses)

    t, p = b_spline_interpolation(times, poses, 50)
    # print(poses)

    return p


# 'tower_1':(-40,140),
# 'tower_2':(-40,140),
# 'tower_3':(-40,140),
# 'base':(-140,140),

tower_keyframes = {
    "neutral": [70,50,50],
    "up": [120,120,120],
    "down": [0, 0, 0],
    "tilt_left": [100, 80, 20],
    "tilt_right": [80, 20, 100],
    "tilt_front": [0, 100, 100],
    "tilt_back": [100, 0, 0],
    "tilt_front_left": [0, 100, 0],
    "tilt_front_right": [0, 0, 100],
    "really_front": [-30,130,130]

}

def generate_random_behavior():
    num_points = np.random.randint(3, 10)
    tower_poses = []
    times = []


    for _ in range(num_points):
        # print(tower_keyframes.keys())
        label = np.random.choice(list(tower_keyframes.keys()))
        time = np.random.uniform(0.5, 4.5)

        times.append(time)
        tower_poses.append(tower_keyframes[label])

    times = sorted(times)

    times.append(5)
    tower_poses.append(tower_keyframes['neutral'])

    times[0] = 0
    tower_poses[0] = tower_keyframes['neutral']

    if np.random.rand() < 0.5:
        t, tower_p = b_spline_from_control_points(times, tower_poses, 50)
    else:
        t, tower_p = linear_interpolation_from_control_points(times, tower_poses, 50)

    num_points = np.random.randint(3, 5)
    base_poses = []
    times = []

    for _ in range(num_points):
        value = np.random.choice([-100, -40, 0, 40, 100])
        time = np.random.uniform(0.5, 4.5)

        times.append(time)
        base_poses.append([value])

    times = sorted(times)

    times.append(5)
    base_poses.append([0])

    times[0] = 0
    base_poses[0] = [0]


    # interpolator = create_linear_pose_interpolator(poses, times)
    if np.random.rand() < 0.5:
        t, base_p = b_spline_from_control_points(times, base_poses, 50)
    else:
        t, base_p = linear_interpolation_from_control_points(times, base_poses, 50)



    return np.hstack((base_p, tower_p))
   

        
        

    
if __name__ == '__main__':
    files = get_all_subfiles_os('./blossom-public/blossompy/src/sequences/woody')
    gestures = np.zeros((1500, 50, 4))
    
    # for i, file in enumerate(files):
    #     print(file, i)
    #     gestures[i] = convert_file_to_array(files[i])
    #     # print(gestures[i])

    for i in range(len(gestures)):
        behavior = generate_random_behavior()
        print(i)
        gestures[i] = behavior
        # break
    
    print(np.min(gestures, axis=(0,1)), np.max(gestures,axis=(0,1)))
    # print(gestures)

    np.save('blossom_gestures.npy', gestures)









