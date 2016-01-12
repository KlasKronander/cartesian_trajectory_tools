import numpy as np

import rospy
from geometry_msgs.msg import Pose, Quaternion
import tf
from polynomial_interpolation import ViaPointInterpolator


def input_to_1dnparray(inp):
    if(type(inp) == float):
        return np.array([inp])
    elif(type(inp) == np.float64):
        return np.array([inp])
    elif(type(inp) == np.ndarray):
        return np.squeeze(inp)
    else:
        rospy.logerr("unable to interpret input %s", type(inp))


def slerp(so, to, t):
    start_orient = [so.w, so.x, so.y, so.z]
    target_orient = [to.w, to.x, to.y, to.z]
    orient_arr = tf.transformations.quaternion_slerp(
        start_orient, target_orient, t)
    return Quaternion(w=orient_arr[0], x=orient_arr[1],
                      y=orient_arr[2], z=orient_arr[3])


class PointToPointTask(object):
    def __init__(self, start_pose, target_pose, duration=None):
        self.poses_ = [start_pose, target_pose]
        # default duration of 5 seconds for a traj
        self.duration_ = 3.0 if duration is None else duration
        self.prepare_trajectory_()

    def prepare_trajectory_(self):
        self.traj_pos = []
        t = np.array([0.0, self.duration_])

        x_points = np.array([self.poses_[0].position.x,
                             self.poses_[-1].position.x])
        self.traj_pos.append(ViaPointInterpolator(t, x_points))

        y_points = np.array([self.poses_[0].position.y,
                             self.poses_[-1].position.y])
        self.traj_pos.append(ViaPointInterpolator(t, y_points))

        z_points = np.array([self.poses_[0].position.z,
                             self.poses_[-1].position.z])
        # can add some second derivative constraints later here
        self.traj_pos.append(ViaPointInterpolator(t, z_points))
        # nothing to prepare for slerp




    def trajectory(self, t_query):
        if t_query < 0.0:
            t_query = 0.0
        if t_query > self.duration_:
            t_query = self.duration_
        msg = Pose()
        t_query = input_to_1dnparray(t_query)
        msg.position.x = self.traj_pos[0](t_query)
        msg.position.y = self.traj_pos[1](t_query)
        msg.position.z = self.traj_pos[2](t_query)
        # could add polynomial for smoother orientation interpolation
        msg.orientation = slerp(self.poses_[0].orientation,
                                self.poses_[-1].orientation,
                                t_query/self.duration_)
        return msg
