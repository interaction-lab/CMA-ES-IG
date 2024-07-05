import time

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

    def handle_message(self, message):
        if message['type'] == 'play':
            print(f"User {self.pid} is playing {message['data']}")
        if message['type'] == 'ranking':
            return self.learn_preference(message['data'])

    def learn_preference(self, data):
        # Simulate a preference learning task

        print(f'user ranked: {data}')

        if len(data) != 5:
            structured_data = {
                'index0': f'{self.times}.jpg',
                'index1': f'{self.times+1}.jpg',
                'index2': f'{self.times+2}.jpg',
                'index3': f'{self.times+3}.jpg',
                'index4': f'{self.times+4}.jpg',  
            }   
            return structured_data

        time.sleep(2)
        self.times += 1
        self.preference = data

        structured_data = {
            'index0': f'{self.times}.jpg',
            'index1': f'{self.times+1}.jpg',
            'index2': f'{self.times+2}.jpg',
            'index3': f'{self.times+3}.jpg',
            'index4': f'{self.times+4}.jpg',  
        }

        return structured_data
    

def worker(input_queue, output_queue):
    pl = PreferenceLearner(1, 'handover', 'infogain')

    while True:
        message = input_queue.get()
        if message == 'STOP':
            break
        response = pl.handle_message(message)
        if response is not None:
            output_queue.put(response)