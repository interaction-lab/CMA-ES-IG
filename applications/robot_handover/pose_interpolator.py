import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def create_pose_interpolator(poses, quaternions):
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
    quaternions = np.array(quaternions)
    
    # Create spline interpolators for each dimension of the pose
    pose_interpolators = [CubicSpline(t, poses[:, i]) for i in range(poses.shape[1])]
    
    # Create rotation objects for SLERP
    rotations = R.from_quat(quaternions)
    
    # Create SLERP interpolator
    slerp_interpolator = Slerp(t, rotations)
    
    def interpolated_pose(t_val):
        """
        Returns the interpolated pose and quaternion at time t_val.
        
        :param t_val: A value of t between 0 and 1
        :return: Interpolated pose as a numpy array and quaternion as a list
        """
        # Interpolate the position using the splines
        interpolated_position = np.array([interpolator(t_val) for interpolator in pose_interpolators])
        
        # Interpolate the quaternion using SLERP
        interpolated_quaternion = slerp_interpolator(t_val).as_quat()
        
        return interpolated_position, interpolated_quaternion
    
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

    pose_interpolator = create_pose_interpolator(poses, quaternions)

    iposes, quaternions = pose_interpolator(np.linspace(0, 1, 30))

    ax.scatter(poses[:, 0], poses[:, 1], zs=poses[:, 2])
    ax.plot(iposes[0, :], iposes[1,:], zs=iposes[2,:])
    plt.show()
