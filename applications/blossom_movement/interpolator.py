import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np
from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import splprep, splev
import numpy as np
from scipy.interpolate import splprep, splev
import numpy as np
from scipy.interpolate import BSpline

import numpy as np
from scipy.interpolate import interp1d

def linear_interpolation_from_control_points(times, poses, num_points=100):
    """
    Create a linear interpolation curve using the given poses as control points.

    Parameters:
    - times: list or array of time points
    - poses: list or array of poses to be used as control points
    - num_points: number of points to interpolate along the spline

    Returns:
    - interpolated_times: interpolated time points
    - interpolated_poses: interpolated poses
    """

    # Convert to numpy arrays
    times = np.array(times, dtype=float)
    poses = np.array(poses, dtype=float)

    # Check if the input dimensions are consistent
    if len(times) != poses.shape[0]:
        raise ValueError("The number of time points must match the number of poses.")

    # Normalize times to the range [0, 1]
    times_normalized = (times - times.min()) / (times.max() - times.min())

    # Create the linear interpolation function for each dimension
    interpolators = [interp1d(times_normalized, poses[:, dim], kind='linear') for dim in range(poses.shape[1])]

    # Create new time points for interpolation
    interpolated_times = np.linspace(0, 1, num_points)

    # Evaluate the interpolation at the new time points
    interpolated_poses = np.array([interpolator(interpolated_times) for interpolator in interpolators]).T

    return interpolated_times, interpolated_poses

def b_spline_from_control_points(times, poses, num_points=100, k=3):
    """
    Create a B-spline curve using the given poses as control points.

    Parameters:
    - times: list or array of time points
    - poses: list or array of poses to be used as control points
    - num_points: number of points to interpolate along the spline
    - k: degree of the spline. Cubic splines (k=3) are commonly used.

    Returns:
    - interpolated_times: interpolated time points
    - interpolated_poses: interpolated poses
    """

    # Convert to numpy arrays
    times = np.array(times, dtype=float)
    poses = np.array(poses, dtype=float)

    # Check if the input dimensions are consistent
    if len(times) != poses.shape[0]:
        raise ValueError("The number of time points must match the number of poses.")

    # Define the number of control points
    n_control_points = len(times)

    # Define the knot vector
    t = np.linspace(0, 1, n_control_points - k + 1)
    t = np.concatenate(([0] * k, t, [1] * k))

    # Normalize times to the range [0, 1]
    times_normalized = (times - times.min()) / (times.max() - times.min())

    # Create the B-spline object
    splines = [BSpline(t, poses[:, dim], k) for dim in range(poses.shape[1])]

    # Create new time points for interpolation
    interpolated_times = np.linspace(0, 1, num_points)

    # Evaluate the B-spline at the new time points
    interpolated_poses = np.array([spline(interpolated_times) for spline in splines]).T

    return times_normalized, interpolated_poses


def b_spline_interpolation(times, poses, num_points=100):
    """
    Perform B-spline interpolation on a set of poses with corresponding times.

    Parameters:
    - times: list or array of time points
    - poses: list or array of poses corresponding to the time points
    - num_points: number of points to interpolate

    Returns:
    - interpolated_times: interpolated time points
    - interpolated_poses: interpolated poses
    """

    # Convert to numpy arrays
    times = np.array(times, dtype=float)
    poses = np.array(poses, dtype=float)

    # Check if the input dimensions are consistent
    if len(times) != poses.shape[0]:
        raise ValueError("The number of time points must match the number of poses.")

    # Transpose poses to match the expected input shape for splprep
    poses = poses.T

    # Create the B-spline representation of the data
    tck, _ = splprep(poses, u=times, k=3, s = 1)

    # Create new time points for interpolation
    interpolated_times = np.linspace(times.min(), times.max(), num_points)

    # Evaluate the B-spline at the new time points
    interpolated_poses = np.array(splev(interpolated_times, tck)).T

    return interpolated_times, interpolated_poses



def create_linear_pose_interpolator(poses, times, min_bounds=None, max_bounds=None):
    """
    Takes a list of robot poses and corresponding times, and returns a function in terms of t
    that smoothly interpolates between the points using linear interpolation.
    
    :param poses: List of robot poses. Each pose is a list or tuple of coordinates.
    :param times: List of times corresponding to each pose.
    :param min_bounds: List or array of minimum bounds for each dimension.
    :param max_bounds: List or array of maximum bounds for each dimension.
    :return: A function that takes a parameter t and returns the interpolated pose.
    """
    times = np.array(times)
    poses = np.array(poses)
    
    # Create linear interpolators for each dimension of the pose
    pose_interpolators = [interp1d(times, poses[:, i], kind='linear') for i in range(poses.shape[1])]
    
    def interpolated_pose(t_val):
        """
        Returns the interpolated pose at time t_val.
        
        :param t_val: A value of t within the range of times
        :return: Interpolated pose as a numpy array
        """
        t_val = np.clip(t_val, times[0], times[-1])
        # Interpolate the position using the linear interpolators
        interpolated_position = np.array([interpolator(t_val) for interpolator in pose_interpolators])
        
        # Clamp the interpolated position to the specified bounds
        if min_bounds is not None:
            interpolated_position = np.maximum(interpolated_position, min_bounds)
        if max_bounds is not None:
            interpolated_position = np.minimum(interpolated_position, max_bounds)
        
        return interpolated_position
    
    return interpolated_pose



def create_pose_interpolator(poses, times):
    """
    Takes a list of robot poses, corresponding quaternions, and times, and returns a function in terms of t
    that smoothly interpolates between the points using a spline for positions and SLERP for quaternions.
    
    :param poses: List of robot poses. Each pose is a list or tuple of coordinates.
    :param quaternions: List of quaternions corresponding to the poses.
    :param times: List of times corresponding to each pose.
    :return: A function that takes a parameter t and returns the interpolated pose and quaternion.
    """
    times = np.array(times)
    poses = np.array(poses)
    
    # Create spline interpolators for each dimension of the pose
    pose_interpolators = [CubicSpline(times, poses[:, i]) for i in range(poses.shape[1])]
    
    def interpolated_pose(t_val):
        """
        Returns the interpolated pose and quaternion at time t_val.
        
        :param t_val: A value of t within the range of times
        :return: Interpolated pose as a numpy array
        """
        t_val = np.clip(t_val, times[0], times[-1])
        # Interpolate the position using the splines
        interpolated_position = np.array([interpolator(t_val) for interpolator in pose_interpolators])
        
        return interpolated_position
    
    return interpolated_pose

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    # Usage example
    times = [0, .1, 2, 3.6, 4]
    poses = [
        [0, 0, 0],
        [1, 1, 0],
        [2, 0, 1],
        [3, 1, 1],
        [4, 0, 0]
    ]

    # Perform B-spline interpolation
    interpolated_times, interpolated_poses = b_spline_interpolation(times, poses, num_points=50)

    print("Interpolated Times:", interpolated_times)
    print("Interpolated Poses:", interpolated_poses)
