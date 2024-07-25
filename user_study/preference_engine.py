import time
import csv
import os
import numpy as np
from arm_controller import ArmController
from blossom_controller import BlossomController

class PreferenceLearner:
    '''
    The class that handles the generation of queries and learns the users
    preferences. 
    '''
    def __init__(self, pid, task, method):
        self.preference = None
        self.times = 0
        self.pid = pid
        self.task = task
        self.method = method

        if self.task == 'blossom':
            self.controller = BlossomController()
            self.embeddings = np.load('./static/blossom_embeddings.npy')
            
        if self.task == 'handover':
            self.controller = ArmController()
            self.embeddings = np.load('./static/handover_embeddings.npy')



    def handle_message(self, message):
        
        if message['type'] == 'play':
            self.log_play_message(message['data'])
            print(message['data'].split('/')[-1])
            self.controller.play(message['data'].split('/')[-1])


            print(f"User {self.pid} is playing {message['data']}")

        if message['type'] == 'ranking':
            if len(message['data']) == 3:
                self.log_rank_message(message['data'])
            return self.learn_preference(message['data'])

    def log_play_message(self, data):
        filename = f"./data/play{self.pid}.csv"
        file_exists = os.path.isfile(filename)

        with open(filename, mode='a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if not file_exists:
                csvwriter.writerow(['timestamp', 'pid', 'task', 'method', 'sequence', 'played_trajectory'])  # Write header if file does not exist
            csvwriter.writerow([time.time(), 
                                self.pid,
                                self.task,
                                self.method,
                                self.times,
                                data])
            
    def log_rank_message(self, data):
        print(data)
        filename = f"./data/ranking{self.pid}.csv"
        file_exists = os.path.isfile(filename)

        with open(filename, mode='a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if not file_exists:
                csvwriter.writerow(['timestamp', 'pid', 'task', 'method', 'sequence', 'trajectory1index', 'trajectory1rank','trajectory2index', 'trajectory2rank','trajectory3index', 'trajectory3rank'])  # Write header if file does not exist
            csvwriter.writerow([time.time(), 
                                self.pid,
                                self.task,
                                self.method,
                                self.times,
                                self.query[f'index{data[0]}'],
                                0,
                                self.query[f'index{data[1]}'],
                                1,
                                self.query[f'index{data[2]}'],
                                2
                                ])




    def learn_preference(self, data):
        # Simulate a preference learning task

        print(f'user ranked: {data}')

        if len(data) != 3:
            structured_data = {
                'index0': f'{self.times}.jpg',
                'index1': f'{self.times+1}.jpg',
                'index2': f'{self.times+2}.jpg',
            }
            self.query = structured_data   
            return structured_data

        time.sleep(2)
        self.times += 1
        self.preference = data

        structured_data = {
            'index0': f'{self.times}.jpg',
            'index1': f'{self.times+1}.jpg',
            'index2': f'{self.times+2}.jpg',  
        }
        self.query = structured_data 

        return structured_data
    

def worker(input_queue, output_queue):
    #task can be 'handover' or 'gesture'
    pl = PreferenceLearner(0, 'blossom', 'infogain')

    while True:
        message = input_queue.get()
        if message == 'STOP':
            break
        response = pl.handle_message(message)
        if response is not None:
            output_queue.put(response)