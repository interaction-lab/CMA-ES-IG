import numpy as np
import time

class BlossomController:
    def __init__(self) -> None:
        self.data = np.load('blossom_gestures')


    def play(self, index):
        #TODO: fillout
        print('playing index!!! ' + index)
        time.sleep(1)
