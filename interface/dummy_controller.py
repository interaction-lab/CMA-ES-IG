import numpy as np
import time
import sys

class DummyController:
    def __init__(self) -> None:
        self.data = np.load('interface/static/dummy_gestures.npy') #replace with path to pre-recorded gestures file
        self.reset()
        
        
    def reset(self):
        # place your robot reset code here
        print('Resetting dummy controller')

    def play(self, index):
        # place your robot play code here

        #this example plays pre-recorded behaviors
        behavior = self.data[int(index)]
        print('playing behavior at index ' + index)