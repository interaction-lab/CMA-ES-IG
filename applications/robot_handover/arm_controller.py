import numpy as np
import time
import abr_jaco2
from abr_control.controllers import Floating
from pose_interpolator import create_pose_interpolator
from get_char import KBHit

class ArmController:
    def __init__(self) -> None:
        self.robot_config = abr_jaco2.Config(use_cython=True)
        self.ctrlr = Floating(self.robot_config, dynamic=True, task_space=True)
        # run controller once to generate functions / take care of overhead
        # outside of the main loop, because force mode auto-exits after 200ms
        zeros = np.zeros(self.robot_config.N_JOINTS)
        self.ctrlr.generate(zeros, zeros)

        # create our interface for the jaco2
        self.interface = abr_jaco2.Interface(self.robot_config)
        self.interface.connect()
        self.interface.init_position_mode()
        self.key_listener = KBHit()

        self.setpoint = np.array([4.8, 2.9, 1.0, 4.2, 1.4, 1.3], dtype='float32')
        self.p_lims = np.array([5.5, 5.5, 4.5, 2.5, 2.5, 1.5])

        self.interface.send_target_angles(self.setpoint)

    def __del__(self):
        self.interface.init_position_mode()
        time.sleep(0.1)
        self.interface.send_target_angles(np.array([4.8, 2.9, 1.0, 4.2, 1.4, 1.3], dtype='float32'))
        time.sleep(0.1)
        self.interface.disconnect()
        time.sleep(0.1)
        print('Connection Closed.')

    def control(self):
        if self.key_listener.kbhit():
            c = self.key_listener.getch()
            if ord(c) == ord('i'): # close hand
                self.interface.open_hand(False)
            if ord(c) == ord('o'): # open hand
                self.interface.open_hand(True)

        feedback = self.interface.get_feedback()
        u = self.ctrlr.generate(q=feedback['q'], dq=feedback['dq']) #feedback['dq'])

        p = 10 * (self.setpoint - feedback['q'])
        p = np.clip(p, -self.p_lims, self.p_lims)

        d = 2 * (0 - feedback['dq'])

        self.interface.send_forces(np.array(u + p + d, dtype='float32'))

    def update_setpoint(self, setpoint):
        # print(setpoint)
        self.setpoint = np.asarray(setpoint, dtype='float32') 
        

        
if __name__ == '__main__':
    controller = ArmController()

    p1 = np.load('points1.npy', allow_pickle=True)
    p2 = np.load('points2.npy', allow_pickle=True)



    pose1 = [4.8, 2.9, 1.0, 4.2, 1.4, 1.3]
    pose2 = p1[np.random.choice(len(p1))] + .01* np.random.randn(6)
    pose3 = p2[np.random.choice(len(p2))]

    traj = create_pose_interpolator([pose1, pose2, pose3])
    traj_length = 15

    start = time.time()
    end = start + traj_length + 2
    controller.interface.init_force_mode()
    while time.time() < end:
        controller.control()
        t = (time.time() - start) / traj_length
        controller.update_setpoint(traj(t))
    
    del controller
    time.sleep(1)