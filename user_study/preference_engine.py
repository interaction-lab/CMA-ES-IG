import time
import csv
import os
import numpy as np
from arm_controller import ArmControllerProxy
# from blossom_controller import BlossomController


from irlpreference.input_models import LuceShepardChoice, WeakPreferenceChoice
from irlpreference.query_generation import InfoGainQueryGenerator, RandomQueryGenerator, VolumeRemovalQueryGenerator
from irlpreference.reward_parameterizations import MonteCarloLinearReward
from cmaes_generators import CMAESGenerator, CMAESIGGenerator




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
            self.controller = ArmControllerProxy()
            self.embeddings = np.load('./static/handover_embeddings.npy')


        self.user_choice_model = LuceShepardChoice(rationality = 10)
        self.user_estimate = MonteCarloLinearReward(self.embeddings.shape[1], number_samples=10_000)

        if self.method == 'infogain':
            self.generator = InfoGainQueryGenerator([(-1,1)] * self.embeddings.shape[1])
        
        if self.method == 'CMA-ES':
            self.generator = CMAESGenerator(self.embeddings.shape[1], [(-1,1)] * self.embeddings.shape[1], 3)

        if self.method == 'CMA-ES-IG':
            self.generator = CMAESIGGenerator(self.embeddings.shape[1], [(-1,1)] * self.embeddings.shape[1], 3)

    def get_best_solution(self):
        # returns the best solution based on the current user estimate
        expected_preference = self.user_estimate.get_expectation()

        best = np.argmin(self.embeddings @ expected_preference)

        return str(best)
    
    def handle_message(self, message):
        # all messages from the web app end up here
        # currently, the message types can be a play command
        # or a ranking command. This function plays the requested
        # trajectory or processes the ranking data and returns the next set of trajectories to rank
        
        #Code for handling play messages
        if message['type'] == 'play':
            print(f"{message['data']}")
            if message['data'] == 'best':
                print(f"User {self.pid} is playing the best trajectory")
                best = self.get_best_solution()
                self.controller.play(best)
                self.log_play_message(f'{best};best')
                return
            
            self.log_play_message(message['data'])
            self.controller.play(message['data'].split('/')[-1])
            print(f"User {self.pid} is playing {message['data']}")

        #code for handling ranking messages
        if message['type'] == 'ranking':
            if len(message['data']) == 3:
                self.log_rank_message(message['data'])
            return self.learn_preference(message['data'])
        
        if message['type'] == 'set_favorite':
            self.log_favorite_message(message['data'])
            return


    def log_favorite_message(self, data):
        filename = f"./data/favorite{self.pid}.csv"
        file_exists = os.path.isfile(filename)

        with open(filename, mode='a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if not file_exists:
                csvwriter.writerow(['timestamp', 'pid', 'task', 'method', 'sequence', 'favorite_id'])  # Write header if file does not exist
            csvwriter.writerow([time.time(), 
                                self.pid,
                                self.task,
                                self.method,
                                self.times,
                                str(data)])
                  
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
                                str(data)])
            
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
                                data[0],
                                0,
                                data[1],
                                1,
                                data[2],
                                2
                                ])

    def get_query_indices(self, query):
        # returns the closest indices in self.embeddings to the query
        indices = []
        for q in query:
            indices.append(np.argmin(np.linalg.norm(self.embeddings - q, axis=1)))

        return indices


    def learn_preference(self, data):
        # Simulate a preference learning task
        print(f'user ranked: {data}')

        # base case: get the first query
        if len(data) != 3:
            query = self.generator.get_query(3, self.user_estimate, self.user_choice_model) #generates choice between two options
            indices = self.get_query_indices(query)
            structured_data = {
                'index0': f'{indices[0]}',
                'index1': f'{indices[1]}',
                'index2': f'{indices[2]}',
            }

            self.query = structured_data   
            return structured_data

        
        if self.method == 'infogain':
            for comparison in [[0,1], [1,2], [0,2]]:
                options = [self.embeddings[int(data[comparison[0]])], self.embeddings[int(data[comparison[1]])]]
                self.user_choice_model.tell_input(1, options) #second option is selected in each comparison
                self.user_estimate.update(self.user_choice_model.get_probability_of_input) 

        if self.method == 'CMA-ES' or self.method == 'CMA-ES-IG':
            ranking = [1,2,3] #gets the indices
            query = [self.embeddings[int(data[0])], self.embeddings[int(data[1])], self.embeddings[int(data[2])]]
            self.generator.tell(query, ranking)

        # user_estimate.update(user_choice_model.get_probability_of_input)   
        
        # time.sleep(2)
        self.times += 1
        self.preference = data

        query = self.generator.get_query(3, self.user_estimate, self.user_choice_model) #generates choice between two options
        indices = self.get_query_indices(query)
        structured_data = {
            'index0': f'{indices[0]}',
            'index1': f'{indices[1]}',
            'index2': f'{indices[2]}',
        }

        self.query = structured_data   
        return structured_data
    

def worker(input_queue, output_queue):
    #task can be 'handover' or 'gesture'

    # one of 
    # infogain
    # CMA-ES
    # CMA-ES-IG
    pl = PreferenceLearner(16, 'handover', 'CMA-ES-IG')
    

    while True:
        message = input_queue.get()
        if message == 'STOP':
            pl.controller.stop()
            break
        response = pl.handle_message(message)
        if response is not None:
            output_queue.put(response)