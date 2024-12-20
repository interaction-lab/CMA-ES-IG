import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.stats import entropy

from irlpreference.input_models import LuceShepardChoice, WeakPreferenceChoice
from irlpreference.query_generation import InfoGainQueryGenerator, RandomQueryGenerator, VolumeRemovalQueryGenerator
from irlpreference.reward_parameterizations import MonteCarloLinearReward

from cmaesig_query_generator import CMAESIGGenerator
from cmaes_query_generator import CMAESGenerator

fig, ax = plt.subplots(2)

def alignment_metric(true_w, guessed_w):
    return np.dot(guessed_w, true_w) / (np.linalg.norm(guessed_w) * np.linalg.norm(true_w))

def random_point_in_unit_ball(dim_embedding):
    # Generate a random direction on the unit sphere
    direction = np.random.normal(0, 1, dim_embedding)
    direction /= np.linalg.norm(direction)  # Normalize to make it a unit vector

    # Scale by a random radius for uniform distribution within the ball
    radius = np.random.uniform(0, 1) ** (1 / dim_embedding)
    point_in_ball = direction * radius
    
    return point_in_ball

#no label was dim 16, 5items
#Experimental Constants
dim_embedding = 8
items_per_query = 4

true_preference = np.random.uniform(low=-1, high=1, size=dim_embedding)
number_of_trials = 30
max_number_of_queries = 30


#User Input and Estimation of reward functions
user_choice_model = LuceShepardChoice(rationality = 10)
user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=5_000)

#Generators
random_generator = RandomQueryGenerator( [(-1,1)] * dim_embedding)
ig_generator = InfoGainQueryGenerator([(-1,1)] * dim_embedding)

cma_es = CMAESGenerator(dim_embedding,[(-1,1)] * dim_embedding, items_per_query, sigma=0.5)
cma_es_ig = CMAESIGGenerator(dim_embedding,[(-1,1)] * dim_embedding, items_per_query, sigma=0.5)

generators = [cma_es_ig, cma_es]
names = ['CMA-ES-IG', 'CMA-ES','Random', 'IG']

for generator, name in zip(generators, names):
    cumulative_values = []
    cumulative_regret = []
    cumulative_per_query_alignment = []

    for _ in tqdm(range(number_of_trials)):

        user_estimate.reset()
        if name == 'CMA-ES' or name == 'CMA-ES-IG':
            generator.reset()

        all_trajectories = np.random.randn(5000, dim_embedding)
        true_preference = random_point_in_unit_ball(dim_embedding)
        alignment = [0]
        regret = []
        per_query_alignment = []
        


        for _ in range(max_number_of_queries):
            query = generator.get_query(items_per_query, user_estimate, user_choice_model) #generates choice between two options
            probabilities = user_choice_model.get_choice_probabilities(query, np.array([true_preference])).flatten()
            choice = np.argmax(probabilities)
            user_choice_model.tell_input(choice, query)

            if name == 'CMA-ES' or name == 'CMA-ES-IG':
                ranking = probabilities.argsort().argsort() #gets the indices
                generator.tell(list(query), ranking)

            user_estimate.update(user_choice_model.get_probability_of_input)   

            estimated_omega = user_estimate.get_expectation()
            best_est_trajectory = all_trajectories[np.argmax(all_trajectories@estimated_omega)]
            regret.append(np.max(all_trajectories@true_preference) - best_est_trajectory@true_preference)

            alignment.append(alignment_metric(estimated_omega, true_preference))
            per_query_alignment.append(np.average([alignment_metric(q, true_preference) for q in query]))



        cumulative_values += [alignment]
        cumulative_regret += [regret]
        cumulative_per_query_alignment += [per_query_alignment]
    

    np.save(f'./results/{name}_alignment_{items_per_query}items_dim{dim_embedding}.npy', cumulative_values)
    np.save(f'./results/{name}_regret_{items_per_query}items_dim{dim_embedding}.npy', cumulative_regret)
    np.save(f'./results/{name}_per_query_alignment_{items_per_query}items_dim{dim_embedding}.npy', cumulative_per_query_alignment)

    m = np.mean(np.array(cumulative_regret), axis=0) 
    std = np.std(np.array(cumulative_regret), axis=0) / np.sqrt(number_of_trials)
    ax[0].fill_between(range(max_number_of_queries), m-std, m+std, alpha=0.3)
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
