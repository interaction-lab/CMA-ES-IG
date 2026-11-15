import numpy as np
import time
import abr_jaco2
from abr_control.controllers import Floating
# from pose_interpolator import create_pose_interpolator
import os

# Windows
if os.name == 'nt':
    import msvcrt

# Posix (Linux, OS X)
else:
    import sys
    import termios
    import atexit
    from select import select


PORT = 65431

class ArmControllerProxy:
    def __init__(self, host='localhost', port=PORT) -> None:
        self.host = host
        self.port = port
        self.connection = None
        self.connect()

    def connect(self):
        ''' Establishes a connection to the server. '''
        if self.connection:
            self.connection.close()
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection.connect((self.host, self.port))

    def play(self, message):
        ''' Sends a message to the server and receives a response. '''
        try:
            self.connection.sendall(message.encode())
            data = self.connection.recv(1024)
            # print('Received', repr(data))
        except (socket.error, ConnectionResetError) as e:
            print('Connection error:', e)
            self.connect()  # Reconnect if there was an error

    def close(self):
        ''' Closes the connection to the server. '''
        if self.connection:
            self.connection.close()
            self.connection = None





class KBHit:
    def __init__(self):
        '''Creates a KBHit object that you can call to do various keyboard things.
        '''

        if os.name == 'nt':
            pass
        
        else:
    
            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)
    
            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)
    
            # Support normal-terminal reset at exit
            atexit.register(self.set_normal_term)
    
    
    def set_normal_term(self):
        ''' Resets to normal terminal.  On Windows this is a no-op.
        '''
        
        if os.name == 'nt':
            pass
        
        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)


    def getch(self):
        ''' Returns a keyboard character after kbhit() has been called.
            Should not be called in the same program as getarrow().
        '''
        
        s = ''
        
        if os.name == 'nt':
            return msvcrt.getch().decode('utf-8')
        
        else:
            return sys.stdin.read(1)
                        

    def getarrow(self):
        ''' Returns an arrow-key code after kbhit() has been called. Codes are
        0 : up
        1 : right
        2 : down
        3 : left
        Should not be called in the same program as getch().
        '''
        
        if os.name == 'nt':
            msvcrt.getch() # skip 0xE0
            c = msvcrt.getch()
            vals = [72, 77, 80, 75]
            
        else:
            c = sys.stdin.read(3)[2]
            vals = [65, 67, 66, 68]
        
        return vals.index(ord(c.decode('utf-8')))
        

    def kbhit(self):
        ''' Returns True if keyboard character was hit, False otherwise.
        '''
        if os.name == 'nt':
            return msvcrt.kbhit()
        
        else:
            dr,dw,de = select([sys.stdin], [], [], 0)
            return dr != []



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
        self.p_lims = np.array([4.5, 10.5, 5.5, 2.5, 2.5, 3.5])

        self.interface.send_target_angles(self.setpoint)

        self.trajectories = np.load('./static/handovers.npy')

    def __del__(self):
        self.interface.init_position_mode()
        time.sleep(0.1)
        self.interface.send_target_angles(np.array([4.8, 2.9, 1.0, 4.2, 1.4, 1.3], dtype='float32'))
        time.sleep(0.1)
        self.interface.disconnect()
        time.sleep(0.1)
        print('Connection Closed.')



    def control(self):
    
        feedback = self.interface.get_feedback()
        u = self.ctrlr.generate(q=feedback['q'], dq=feedback['dq']) #feedback['dq'])

        p = 12 * (self.setpoint - feedback['q'])
        p = np.clip(p, -self.p_lims, self.p_lims)

        d = 2 * (0 - feedback['dq'])

        self.interface.send_forces(np.array(u + p + d, dtype='float32'))

    def update_setpoint(self, setpoint):
        # print(setpoint)
        self.setpoint = np.asarray(setpoint, dtype='float32')

    def get_trajectory_setpoint(self, traj_index, t):
        '''
        Get the setpoint from a trajectory based on the trajectory index and time.
        
        Parameters:
        - traj_index: Index of the trajectory to use.
        - t: Current time (or normalized time) used to determine the position in the trajectory.
        
        Returns:
        - The setpoint for the arm at the given time `t` in the specified trajectory.
        '''
        # Ensure the trajectory index is valid
        if traj_index >= len(self.trajectories):
            raise ValueError("Trajectory index out of range.")
        
        # Get the trajectory data
        trajectory = self.trajectories[traj_index]
        
        # Number of points in the trajectory
        num_points = trajectory.shape[0]
        
        # Determine the index of the current trajectory point
        t_normalized = max(0, min(1, t))  # Ensure t is between 0 and 1
        point_index = int(t_normalized * (num_points - 1))
        
        # Linear interpolation
        if point_index < num_points - 1:
            alpha = t_normalized * (num_points - 1) - point_index
            p0 = trajectory[point_index]
            p1 = trajectory[point_index + 1]
            setpoint = p0 + alpha * (p1 - p0)
        else:
            setpoint = trajectory[-1]
        
        return setpoint
        
import socket


if __name__ == '__main__':
    controller = ArmController()
    pose1 = [4.8, 2.9, 1.0, 4.2, 1.4, 1.3]
    traj_length = 15
    host = 'localhost'
    port = PORT

    start = 0
    traj_i = 0
    end = start + traj_length + 2

    controller.interface.init_force_mode()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        s.settimeout(0.01)  # Set a timeout for the socket to avoid blocking

        print('Server started and listening for connections...')
        conn = None

        while True:
            # Check for new connections
            if conn is None:
                try:
                    conn, addr = s.accept()
                    conn.settimeout(0.005)  # Set a timeout for the connection to avoid blocking
                    # print('Connected by', addr)
                except socket.timeout:
                    pass

            # Control logic
            start_loop = time.time()

            if controller.key_listener.kbhit():
                c = controller.key_listener.getch()
                if ord(c) == ord('i'):  # close hand
                    controller.interface.open_hand(False)
                if ord(c) == ord('o'):  # open hand
                    controller.interface.open_hand(True)
                if ord(c) == ord('q'):  # quit
                    break

                if ord(c) == ord('s'):  # state
                    print(controller.interface.get_feedback()['q'])

            controller.control()
            
            TRAJ_LEN = 5
            if time.time() - start < TRAJ_LEN + 2:
                t = (time.time() - start) / TRAJ_LEN
                set_point = controller.get_trajectory_setpoint(traj_i, t)
                controller.update_setpoint(set_point)
            else:
                t = 1 - ((time.time() - start - TRAJ_LEN - 2) / TRAJ_LEN)
                set_point = controller.get_trajectory_setpoint(traj_i, t)
                controller.update_setpoint(set_point)

            # Handle client messages
            
            if conn:
                try:
                    data = conn.recv(1024)
                    if data:
                        print('Received message:', data.decode())

                        if time.time() - start > 2 * TRAJ_LEN + 2:
                            start = time.time()
                            traj_i = int(data.decode())

                        conn.sendall(b'Acknowledged')
                    
                    else:
                        # print('Client disconnected.')
                        conn = None
                except socket.timeout:
                    pass
                except ConnectionResetError:
                    print('Connection was reset by the client.')
                    conn = None
                except Exception as e:
                    print('An error occurred:', e)
                    conn = None

            end_loop = time.time()
            elapsed_time = end_loop - start_loop
            sleep_time = 0.01 - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    del controller
    time.sleep(1)
