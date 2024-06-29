import time


class PreferenceLearner:
    def __init__(self):
        self.preference = None
        self.times = 0

    def learn_preference(self, data):
        # Simulate a preference learning task
        time.sleep(2)
        self.times += 1
        self.preference = data
        return f"Learned preference: {data}, times: {self.times}"
    

def worker(input_queue, output_queue):
    pl = PreferenceLearner()
    while True:
        message = input_queue.get()
        if message == 'STOP':
            break
        response = pl.learn_preference(message)
        output_queue.put(response)