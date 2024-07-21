import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def create_pose_interpolator(poses):
    """
    Takes a list of robot poses and corresponding quaternions, and returns a function in terms of t
    that smoothly interpolates between the points using a spline for positions and SLERP for quaternions.
    
    :param poses: List of robot poses. Each pose is a list or tuple of coordinates.
    :param quaternions: List of quaternions corresponding to the poses.
    :return: A function that takes a parameter t and returns the interpolated pose and quaternion.
    """
    # Assuming poses and quaternions are uniformly spaced in time
    num_poses = len(poses)
    t = np.linspace(0, 1, num_poses)
    poses = np.array(poses)
    
    # Create spline interpolators for each dimension of the pose
    pose_interpolators = [CubicSpline(t, poses[:, i]) for i in range(poses.shape[1])]
    
    
    def interpolated_pose(t_val):
        """
        Returns the interpolated pose and quaternion at time t_val.
        
        :param t_val: A value of t between 0 and 1
        :return: Interpolated pose as a numpy array and quaternion as a list
        """
        t_val = np.clip(t_val, 0, 1)
        # Interpolate the position using the splines
        interpolated_position = np.array([interpolator(t_val) for interpolator in pose_interpolators])
        
        return interpolated_position
    
    return interpolated_pose

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    ax = plt.figure().add_subplot(projection='3d')

    poses = np.array([
        [0, 0, 0], 
        [1, 2, 2], 
        [2, 3, -2], 
        [4, 2, 0]
    ])

    quaternions = [
        [0, 0, 0, 1], 
        [0.7071, 0, 0, 0.7071], 
        [1, 0, 0, 0], 
        [0.7071, 0, 0, -0.7071]
    ]

    pose_interpolator = create_pose_interpolator(poses)

    iposes= pose_interpolator(.1)

    print(iposes)
