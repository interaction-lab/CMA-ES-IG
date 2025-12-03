import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import cdist
from tqdm import tqdm

from irlpreference.input_models import LuceShepardChoice, WeakPreferenceChoice
from irlpreference.reward_parameterizations import MonteCarloLinearReward

sys.path.append("../experiments")
from cmaes_generators import CMAESGenerator, CMAESIGGenerator
from irlpreference.query_generation import InfoGainQueryGenerator, RandomQueryGenerator, VolumeRemovalQueryGenerator


##########################################################################################
#
#                       Parameters for running simulated experiments
#
##########################################################################################

EXPERIMENTAL_DOMAIN = 'pylips_appearance'  # 'lunar_lander', 'pylips_appearance', 'blossom_voice', 'driving'
NUM_TRIALS = 100
MAX_QUERIES = 30
ITEMS_PER_QUERY = 5
DIM_EMBEDDING = 8  # 2, 4, 6, 8

sys.path.append(f"./{EXPERIMENTAL_DOMAIN}")
from utils import get_features # each domain has its own get_features function
############################################################################################

def alignment_metric(true_w, guessed_w):
    return np.dot(guessed_w, true_w) / (np.linalg.norm(guessed_w) * np.linalg.norm(true_w))



if __name__ == "__main__":

    #User Input and Estimation of reward functions
    user_choice_model = LuceShepardChoice(rationality = 5)
    user_estimate = MonteCarloLinearReward(DIM_EMBEDDING, number_samples=10_000)

    #Generators
    random_generator = RandomQueryGenerator( [(-1,1)] * DIM_EMBEDDING)
    ig_generator = InfoGainQueryGenerator([(-1,1)] * DIM_EMBEDDING)
    cma_es = CMAESGenerator(DIM_EMBEDDING,[(-1,1)] * DIM_EMBEDDING, ITEMS_PER_QUERY, sigma=1.3)
    cma_es_ig = CMAESIGGenerator(DIM_EMBEDDING,[(-1,1)] * DIM_EMBEDDING, ITEMS_PER_QUERY, sigma=1.6)

    generators = [cma_es_ig, cma_es, random_generator, ig_generator]
    names = ['CMA-ES-IG', 'CMA-ES','Random', 'IG']

    for generator, name in zip(generators, names):
        data = []
        print(f"\nStarting evaluation for {name} generator...\n") 

        for trial_num in tqdm(range(NUM_TRIALS)):

            user_estimate.reset()
            if name == 'CMA-ES' or name == 'CMA-ES-IG':
                generator.reset()

            all_trajectories, _ = get_features(DIM_EMBEDDING, emb_dir=f"./{EXPERIMENTAL_DOMAIN}/data/embeddings/")
            all_trajectories = minmax_scale(all_trajectories, feature_range=(-1, 1))
            # np.clip(all_trajectories, -1, 1, out=all_trajectories) # clipping to bounds
            
            true_preference = np.random.uniform(low=-1, high=1, size=DIM_EMBEDDING)

            for q in range(MAX_QUERIES):
                query = generator.get_query(ITEMS_PER_QUERY, user_estimate, user_choice_model) #generates choice between two options
                quality = [alignment_metric(q, true_preference) for q in query]

                # get closest items in dataset to each query item
                query = np.array(query)
                dists = cdist(all_trajectories, query)
                indices = np.argmin(dists, axis=0)
                closest_items = all_trajectories[indices]

                # simulate user choice by iteratively selecting top items to get a ranking
                sorted_query = []
                for i in range(ITEMS_PER_QUERY-1):
                    probabilities = user_choice_model.get_choice_probabilities(query, np.array([true_preference])).flatten()
                    choice = np.random.choice(np.arange(len(query)), p=probabilities)

                    user_choice_model.tell_input(choice, query)
                    user_estimate.update(user_choice_model.get_probability_of_input) 

                    sorted_query.append(query[choice])
                    #remove chosen item and repeat
                    query = np.delete(query, choice, axis=0)
                
                sorted_query.append(query[0])

                if name == 'CMA-ES' or name == 'CMA-ES-IG':
                    ranking = probabilities.argsort().argsort() #gets the indices
                    generator.tell(sorted_query, list(range(ITEMS_PER_QUERY))[::-1]) #tells generator the ranking is complete


                # calculate metrics
                estimated_omega = user_estimate.get_expectation()
            
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
        df.to_csv(f'./{EXPERIMENTAL_DOMAIN}/data/results/{name}_dim{DIM_EMBEDDING}_items{ITEMS_PER_QUERY}.csv', index=False)