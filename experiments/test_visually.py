import numpy as np
import matplotlib.pyplot as plt

def plot_query(query, true_preference, estimate):
    def rastrigin(x, y):
        return true_preference[0]*x + true_preference[1]*y + .6*(2 -  np.cos(4*x) - 1* np.cos(4*y)) 

    # Generate x and y values
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    y = np.linspace(-2 * np.pi, 2 * np.pi, 100)

    # Create a meshgrid
    X, Y = np.meshgrid(x, y)

    # Compute z values
    Z = rastrigin(X, Y)

    # Create the heatmap
    plt.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis', aspect='auto')
    plt.plot(query[:,0], query[:,1], 'rx')
    plt.arrow(0, 0, estimate[0], estimate[1], head_width=0.05, head_length=0.05, fc='red', ec='red')
    plt.arrow(0, 0, true_preference[0], true_preference[1], head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    # Show the plot
    plt.show()

def alignment_metric(true_w, guessed_w):
    return np.dot(guessed_w, true_w) / (np.linalg.norm(guessed_w) * np.linalg.norm(true_w))

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from irlpreference.input_models import LuceShepardChoice, WeakPreferenceChoice
from irlpreference.query_generation import InfoGainQueryGenerator, RandomQueryGenerator, VolumeRemovalQueryGenerator
from irlpreference.reward_parameterizations import MonteCarloLinearReward

from user_study.cmaes_generators import CMAESGenerator, CMAESIGGenerator

dim_embedding = 2
number_of_trials = 30
max_number_of_queries = 30
items_per_query = 4

#User Input and Estimation of reward functions
user_choice_model = LuceShepardChoice(rationality = 10)
user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=10_000)

#Generators
random_generator = RandomQueryGenerator( [(-1,1)] * dim_embedding)
vr_generator = VolumeRemovalQueryGenerator( [(-1,1)] * dim_embedding)
ig_generator = InfoGainQueryGenerator([(-1,1)] * dim_embedding)

cma_es = CMAESGenerator(dim_embedding,[(-1,1)] * dim_embedding, items_per_query)
cma_es_ig = CMAESIGGenerator(dim_embedding,[(-1,1)] * dim_embedding, items_per_query, use_boundary_mediods=True)

generators = [cma_es_ig, ig_generator, cma_es, random_generator]
names = ['CMA-ES-IG', 'IG', 'CMA-ES', 'Random']

# fig, ax = plt.subplots(2)

for generator, name in zip(generators, names):
    cumulative_values = []
    cumulative_per_query_alignment = []

    for _ in tqdm(range(number_of_trials)):

        user_estimate.reset()
        if name == 'CMA-ES' or name == 'CMA-ES-IG':
            generator.reset()
        true_preference = np.random.uniform(low=-1, high=1, size=dim_embedding)
        def rastrigin(x, y):
            return true_preference[0]*x + true_preference[1]*y + (2 -  np.cos(2*x) - 1* np.cos(2*y)) 
        alignment = [0]
        per_query_alignment = []


        for _ in range(max_number_of_queries):
            query = generator.get_query(items_per_query, user_estimate, user_choice_model) #generates choice between two options
            # choice = np.argmax(user_choice_model.get_choice_probabilities(query, np.array([true_preference]))) #selects choice from model
            
            plot_query(query, true_preference, user_estimate.get_expectation())

            # probabilities = user_choice_model.get_choice_probabilities(query, np.array([true_preference])).flatten()
            probabilities = [rastrigin(q[0], q[1]) for q in query]
            probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
            choice = np.random.choice(np.arange(items_per_query), p=probabilities)
            user_choice_model.tell_input(choice, query)

            if name == 'CMA-ES' or name == 'CMA-ES-IG':
                ranking = probabilities.argsort().argsort() #gets the indices
                generator.tell(list(query), ranking)

            user_estimate.update(user_choice_model.get_probability_of_input)
            alignment.append(alignment_metric(user_estimate.get_expectation(), true_preference))
            per_query_alignment.append(np.max([alignment_metric(q, true_preference) for q in query]))

        cumulative_values += [alignment]
        cumulative_per_query_alignment += [per_query_alignment]


    m = np.mean(np.array(cumulative_values), axis=0) 
    std = np.std(np.array(cumulative_values), axis=0) 
    ax[0].fill_between(range(max_number_of_queries+1), m-std, m+std, alpha=0.3)
    ax[0].plot(m, label=name)

    m = np.mean(np.array(cumulative_per_query_alignment), axis=0) 
    std = np.std(np.array(cumulative_per_query_alignment), axis=0) 
    ax[1].fill_between(range(max_number_of_queries), m-std, m+std, alpha=0.3)
    ax[1].plot(m, label=name)

plt.title('Alignment Scores by Methodology')
plt.xlabel('Number of Queries')
plt.ylabel('Alignment')
plt.legend()
plt.show()   