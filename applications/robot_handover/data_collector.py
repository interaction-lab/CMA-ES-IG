"""Uses force control to compensate for gravity.  The arm will
hold its position while maintaining compliance.  """

import numpy as np
import traceback
import time

import abr_jaco2
from abr_control.controllers import Floating

from get_char import KBHit

# initialize our robot config
robot_config = abr_jaco2.Config(
    use_cython=True)
ctrlr = Floating(robot_config, dynamic=True, task_space=True)
# run controller once to generate functions / take care of overhead
# outside of the main loop, because force mode auto-exits after 200ms
zeros = np.zeros(robot_config.N_JOINTS)
ctrlr.generate(zeros, zeros)

# create our interface for the jaco2
interface = abr_jaco2.Interface(robot_config)
# interface.open_hand(False)

q_track = []
u_track = []
q_T_track = []

# connect to the jaco
interface.connect()
interface.init_position_mode()

feedback = interface.get_feedback()
print(feedback['q'])

target = np.array([4.8, 2.9, 1.0, 4.2, 1.4, 1.3])

kb = KBHit()
sampled_points = []


# Move to home position
# interface.send_target_angles(robot_config.START_ANGLES)
try:
    interface.init_force_mode()

    while True:
        feedback = interface.get_feedback()
        q_T = interface.get_torque_load()

        # if cnt % 100 == 0:
        #     print(feedback['q'])

        u = ctrlr.generate(q=feedback['q'], dq=feedback['dq']) #feedback['dq'])

        p_lims = np.array([3.5, 3.5, 2.5, 1.5, 1.5, 0.5])
        p = 10 * (target - feedback['q'])
        p = np.clip(p, -p_lims, p_lims)

        d = 2 * (0 - feedback['dq'])

        interface.send_forces(np.array(u + p + d, dtype='float32'))
        # interface.send_forces(np.array(u, dtype='float32'))

        if kb.kbhit():
            c = kb.getch()
            if ord(c) == ord('c'): # ESC

                print(feedback['q'])
                sampled_points.append(feedback['q'])
            if ord(c) == ord('q'): # ESC
                break
            print(c)



        # track data
        q_track.append(np.copy(feedback['q']))
        u_track.append(np.copy(u))
        q_T_track.append(np.copy(q_T))

except Exception as e:
    print(traceback.format_exc())

finally:
    interface.init_position_mode()
    # interface.send_target_angles(robot_config.START_ANGLES)
    interface.disconnect()
    np.save('points2.npy', np.array(sampled_points, dtype=object), allow_pickle=True)
    # plot joint angles throughout trial
    q_track = np.array(q_track)
    import matplotlib
    matplotlib.use("TKAgg")
    import matplotlib.pyplot as plt
    col = ['r', 'b', 'g', 'y', 'k', 'm']
    col2 = ['r--', 'b--', 'g--', 'y--', 'k--', 'm--']
    fig = plt.figure()
    a1 = fig.add_subplot(211)
    a1.set_title('Joint Status')
    a1.set_ylabel('Torque [Nm]')
    a2 = fig.add_subplot(212)
    a2.set_ylabel('Position [rad]')
    u_track = np.array(u_track).T
    q_T_track = np.array(q_T_track).T
    for ii in range(0,6):
        a1.plot(u_track[ii], col2[ii], label='u%i'%ii)
        a2.plot(q_T_track[ii], col[ii], label='q%i'%ii)
    a1.legend()
    a2.legend()
    plt.show()
