from irlpreference.input_models import LuceShepardChoice, WeakPreferenceChoice
from irlpreference.reward_parameterizations import MonteCarloLinearReward

from scipy.spatial.distance import cdist
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
from utils import get_features
sys.path.append("../../experiments")
from cmaes_generators import CMAESGenerator, CMAESIGGenerator
from irlpreference.query_generation import InfoGainQueryGenerator, RandomQueryGenerator, VolumeRemovalQueryGenerator



def alignment_metric(true_w, guessed_w):
    return np.dot(guessed_w, true_w) / (np.linalg.norm(guessed_w) * np.linalg.norm(true_w))

#no label was dim 16, 5items
#Experimental Constants
for dim_embedding in [2, 4, 6 , 8]:

    items_per_query = 4

    true_preference = np.random.uniform(low=-1, high=1, size=dim_embedding)
    number_of_trials = 60
    max_number_of_queries = 20


    #User Input and Estimation of reward functions
    user_choice_model = LuceShepardChoice(rationality = 10)
    user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=10_000)

    #Generators
    random_generator = RandomQueryGenerator( [(-1,1)] * dim_embedding)
    ig_generator = InfoGainQueryGenerator([(-1,1)] * dim_embedding)

    cma_es = CMAESGenerator(dim_embedding,[(-1,1)] * dim_embedding, items_per_query)
    cma_es_ig = CMAESIGGenerator(dim_embedding,[(-1,1)] * dim_embedding, items_per_query)

    generators = [cma_es_ig, cma_es, random_generator]
    names = ['CMA-ES-IG', 'CMA-ES','Random', 'IG']

    for generator, name in zip(generators, names):
        data = []

        for trial_num in tqdm(range(number_of_trials)):

            user_estimate.reset()
            if name == 'CMA-ES' or name == 'CMA-ES-IG':
                generator.reset()

            all_trajectories = get_features(dim_embedding)
            true_preference = np.random.uniform(low=-1, high=1, size=dim_embedding)

            for q in range(max_number_of_queries):
                query = generator.get_query(items_per_query, user_estimate, user_choice_model) #generates choice between two options

                closest_items = cdist(all_trajectories, query).argmin(axis=0)
                query = all_trajectories[closest_items]

                # choice = np.argmax(user_choice_model.get_choice_probabilities(query, np.array([true_preference]))) #selects choice from model
                probabilities = user_choice_model.get_choice_probabilities(query, np.array([true_preference])).flatten()
                choice = np.random.choice(np.arange(items_per_query), p=probabilities)
                user_choice_model.tell_input(choice, query)

                if name == 'CMA-ES' or name == 'CMA-ES-IG':
                    ranking = probabilities.argsort().argsort() #gets the indices
                    generator.tell(list(query), ranking)

                user_estimate.update(user_choice_model.get_probability_of_input)   

                estimated_omega = user_estimate.get_expectation()
                quality = [alignment_metric(q, true_preference) for q in query]
                data.append({
                    'query_num': q,
                    'trial': trial_num,
                    'method': name,
                    'alignment': alignment_metric(estimated_omega, true_preference),
                    'regret': np.max(all_trajectories@true_preference) - all_trajectories[np.argmax(all_trajectories@estimated_omega)]@true_preference,
                    'quality_avg': np.average(quality),
                    'quality_max': np.max(quality),
                    'quality_min': np.min(quality),
                    'quality_median': np.median(quality)
                })

        df = pd.DataFrame(data)
        df.to_csv(f'./data/results/{name}_{dim_embedding}dim_{items_per_query}items.csv', index=False)