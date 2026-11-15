import numpy as np
import time
import sys
sys.path.append("../applications/blossom_movement/blossom-public")
from blossompy import Blossom

class BlossomController:
    def __init__(self) -> None:
        self.data = np.load('./static/blossom_gestures.npy')
        self.blossom = Blossom()
        self.blossom.connect()
        self.reset()
        
        
    def reset(self):
            self.blossom.motor_goto("tower_3", 50)
            self.blossom.motor_goto("tower_2", 50)
            self.blossom.motor_goto("tower_1", 20)
            self.blossom.motor_goto("base", 0)

    def play(self, index):
        behavior = self.data[int(index)]
        for pose in behavior:
            self.blossom.motor_goto("tower_3", pose[3])
            self.blossom.motor_goto("tower_2", pose[2])
            self.blossom.motor_goto("tower_1", pose[1])
            self.blossom.motor_goto("base", pose[0])
            time.sleep(0.1)
            