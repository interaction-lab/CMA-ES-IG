import argparse, sys
sys.path.append("./blossom-public")
print(sys.path)
from blossompy import Blossom
from time import sleep
import numpy as np


def reset():
    bl.motor_goto("tower_3", 50)
    bl.motor_goto("tower_2", 50)
    bl.motor_goto("tower_1", 20)
    bl.motor_goto("base", 0)

bl = Blossom()
bl.connect() # safe init and connects to blossom and puts blossom in reset position
reset()

gestures = np.load('blossom_gestures.npy')
recon = np.load('blossom_unnormalized_reconstructions.npy')

def play_pose(pose):
    print(pose)
    bl.motor_goto("tower_1", int(pose[0]))
    bl.motor_goto("tower_2", int(pose[1]))
    bl.motor_goto("tower_3", int(pose[2]))

def play_gesture(array):

    for row in array:
        # print(row)
        bl.motor_goto("base", row[0])
        bl.motor_goto("tower_1", row[1])
        bl.motor_goto("tower_2", row[2])
        bl.motor_goto("tower_3", row[3])
        sleep(0.1)
        
import time

while True:
    index = input('index of the gesture to play: ')
    print(recon[int(index)].shape, gestures[int(index)].shape)
    play_gesture(gestures[int(index)])
    reset()
    time.sleep(1)
    print('reconstructed')
    play_gesture(recon[int(index)])
    reset()
    time.sleep(1)
    
