import numpy as np
from pose_interpolator import create_pose_interpolator






points1 = np.load('points1.npy', allow_pickle=True)
points2 = np.load('points2.npy', allow_pickle=True)

print(points1.shape, points2.shape)

def generate_handover():
    handover = np.zeros((50, 6))

    pose1 = [4.8, 2.9, 1.0, 4.2, 1.4, 1.3]
    pose2 = points1[np.random.choice(len(points1))] + .01* np.random.randn(6)
    pose2[-1] += .1* np.random.randn()
    pose3 = points2[np.random.choice(len(points2))] + .01* np.random.randn(6)
    pose3[-1] += .1* np.random.randn()

    interp = create_pose_interpolator([pose1, pose2, pose3])

    for i in range(50):
        handover[i] = interp(i/50)

    return handover



handovers = np.zeros((1500, 50, 6))

for i in range(1500):
    handovers[i] = generate_handover()

np.save('handovers.npy', handovers)
