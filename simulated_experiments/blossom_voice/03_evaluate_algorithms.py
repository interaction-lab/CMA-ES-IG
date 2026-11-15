from irlpreference.input_models import LuceShepardChoice, WeakPreferenceChoice
from irlpreference.reward_parameterizations import MonteCarloLinearReward

from scipy.spatial.distance import cdist
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils import load_data, calculate_reward_alignment, calculate_regret, calculate_query_qualities
import sys
sys.path.append("../../experiments")

from cmaes_generators import CMAESGenerator, CMAESIGGenerator
from irlpreference.query_generation import InfoGainQueryGenerator, RandomQueryGenerator, VolumeRemovalQueryGenerator


NUM_TRIALS = 30
MAX_QUERIES = 30
ITEMS_PER_QUERY = 4
DIM_EMBEDDING = 32


# create simulation models
for DIM_EMBEDDING in [4, 8, 16, 32]:

    choice_model = LuceShepardChoice(rationality=1)
    user_estimate = MonteCarloLinearReward(DIM_EMBEDDING, number_samples=10_000)

    random_generator = RandomQueryGenerator( [(-1,1)] * DIM_EMBEDDING)
    ig_generator = InfoGainQueryGenerator([(-1,1)] * DIM_EMBEDDING)
    cma_es = CMAESGenerator(DIM_EMBEDDING,[(-1,1)] * DIM_EMBEDDING, ITEMS_PER_QUERY)
    cma_es_ig = CMAESIGGenerator(DIM_EMBEDDING,[(-1,1)] * DIM_EMBEDDING, ITEMS_PER_QUERY)

    for name, generator in zip(['Random', 'CMA-ES', 'CMA-ES-IG'], [random_generator, cma_es, cma_es_ig]):
       
        data = []
        print(f"\nStarting evaluation for {name} generator...\n") 

        for trial in tqdm(range(NUM_TRIALS)):
            user_estimate.reset()
            true_preference = np.random.uniform(low=-1, high=1, size=DIM_EMBEDDING)
            dataset_measures = load_data(DIM_EMBEDDING, 'pca')

            for query_num in range(MAX_QUERIES):

                query = generator.get_query(ITEMS_PER_QUERY, user_estimate, choice_model)

                # get closest items in dataset to each query item
                query = np.array(query)
                dists = cdist(dataset_measures.T, query)
                indices = np.argmin(dists, axis=0)
                closest_items = dataset_measures.T[indices]

                # simulate user choice
                probabilities = choice_model.get_choice_probabilities(closest_items, np.array([true_preference])).flatten()
                choice = np.random.choice(np.arange(ITEMS_PER_QUERY), p=probabilities)
                choice_model.tell_input(choice, query)
                user_estimate.update(choice_model.get_probability_of_input)

                # calculate metrics
                estimated_omega = user_estimate.get_expectation()

                alignment = calculate_reward_alignment(estimated_omega, true_preference)
                regret = calculate_regret(estimated_omega, true_preference, dataset_measures.T)
                query_quality = calculate_query_qualities(true_preference, closest_items)

                # store metrics
                data.append({
                    'trial': trial,
                    'query_num': query_num,
                    'alignment': alignment,
                    'regret': regret,
                    'query_quality_max': query_quality[0],
                    'query_quality_avg': query_quality[1],
                    'query_quality_median': query_quality[2],
                    'query_quality_min': query_quality[3],
                })

        df = pd.DataFrame(data)
        df.to_csv(f"./data/results/{name}_{DIM_EMBEDDING}dim_{ITEMS_PER_QUERY}items.csv", index=False)
