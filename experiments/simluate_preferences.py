import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from irlpreference.input_models import LuceShepardChoice, WeakPreferenceChoice
from irlpreference.query_generation import InfoGainQueryGenerator, RandomQueryGenerator, VolumeRemovalQueryGenerator
from irlpreference.reward_parameterizations import MonteCarloLinearReward
from user_study.cmaes_generators import CMAESGenerator, CMAESIGGenerator

fig, ax = plt.subplots(2)

def alignment_metric(true_w, guessed_w):
    return np.dot(guessed_w, true_w) / (np.linalg.norm(guessed_w) * np.linalg.norm(true_w))

#Experimental Constants
dim_embedding = 32
true_preference = np.random.uniform(low=-1, high=1, size=dim_embedding)
number_of_trials = 30
max_number_of_queries = 30
items_per_query = 4

#User Input and Estimation of reward functions
user_choice_model = LuceShepardChoice(rationality = 5)
# user_choice_model = WeakPreferenceChoice(delta=1.0)

user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=10_000)

#Generators
random_generator = RandomQueryGenerator( [(-1,1)] * dim_embedding)
vr_generator = VolumeRemovalQueryGenerator( [(-1,1)] * dim_embedding)
ig_generator = InfoGainQueryGenerator([(-1,1)] * dim_embedding)

cma_es = CMAESGenerator(dim_embedding,[(-1,1)] * dim_embedding, items_per_query)
cma_es_ig = CMAESIGGenerator(dim_embedding,[(-1,1)] * dim_embedding, items_per_query)

generators = [cma_es_ig, cma_es, random_generator, ig_generator]
names = ['CMA-ES-IG', 'CMA-ES','Random']#, 'IG']


for generator, name in zip(generators, names):
    cumulative_values = []
    cumulative_per_query_alignment = []

    for _ in tqdm(range(number_of_trials)):

        user_estimate.reset()
        if name == 'CMA-ES' or name == 'CMA-ES-IG':
            generator.reset()
        true_preference = np.random.uniform(low=-1, high=1, size=dim_embedding)
        alignment = [0]
        per_query_alignment = []


        for _ in range(max_number_of_queries):
            query = generator.get_query(items_per_query, user_estimate, user_choice_model) #generates choice between two options
            # choice = np.argmax(user_choice_model.get_choice_probabilities(query, np.array([true_preference]))) #selects choice from model
            probabilities = user_choice_model.get_choice_probabilities(query, np.array([true_preference])).flatten()
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
    std = np.std(np.array(cumulative_values), axis=0) / np.sqrt(number_of_trials)
    ax[0].fill_between(range(max_number_of_queries+1), m-std, m+std, alpha=0.3)
    ax[0].plot(m, label=name)

    m = np.mean(np.array(cumulative_per_query_alignment), axis=0) 
    std = np.std(np.array(cumulative_per_query_alignment), axis=0) / np.sqrt(number_of_trials)
    ax[1].fill_between(range(max_number_of_queries), m-std, m+std, alpha=0.3)
    ax[1].plot(m, label=name)

plt.title('Alignment Scores by Methodology')
plt.xlabel('Number of Queries')
plt.ylabel('Alignment')
plt.legend()
plt.show()
