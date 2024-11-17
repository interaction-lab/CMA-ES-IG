import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm
import pandas as pd

from cmaes_query_generator import CMAESGenerator
from cmaesig_query_generator import CMAESIGGenerator

from irlpreference.input_models import LuceShepardChoice
from irlpreference.reward_parameterizations import MonteCarloLinearReward

# Experimental Constants
items_per_query = 4
number_of_trials = 60
max_number_of_queries = 30

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

#conduct the experiment
results_dataframe = []

for dim_embedding in [8,16,32]:
    #User Input and Estimation of reward functions
    user_choice_model = LuceShepardChoice(rationality = 10)
    user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=5_000)

    #Alignment metric (cosine similarity)
    


    for generator_obj, name in [(CMAESGenerator, 'CMAES'), (CMAESIGGenerator,'CMAESIG')]:
        for sigma in np.linspace(0.01, 1.5, 10):

            generator = generator_obj(dim_embedding, [(-1,1)] * dim_embedding, items_per_query, sigma=sigma)

            for _ in tqdm(range(number_of_trials)):

                user_estimate.reset()
                generator.reset()

                all_trajectories = np.random.randn(5000, dim_embedding)
                true_preference = random_point_in_unit_ball(dim_embedding)
                alignment = []
                regret = []
                per_query_alignment = []

                for _ in range(max_number_of_queries):
                    query = generator.get_query(items_per_query, user_estimate, user_choice_model)
                    probabilities = user_choice_model.get_choice_probabilities(query, np.array([true_preference])).flatten()
                    choice = np.argmax(probabilities)
                    user_choice_model.tell_input(choice, query)

                    ranking = probabilities.argsort().argsort() #gets the indices
                    generator.tell(list(query), ranking)

                    user_estimate.update(user_choice_model.get_probability_of_input)   

                    estimated_omega = user_estimate.get_expectation()
                    best_est_trajectory = all_trajectories[np.argmax(all_trajectories@estimated_omega)]
                    regret.append(np.max(all_trajectories@true_preference) - best_est_trajectory@true_preference)

                    alignment.append(alignment_metric(estimated_omega, true_preference))
                    per_query_alignment.append(np.average([alignment_metric(q, true_preference) for q in query]))


                results_dataframe.append({
                    'generator': name,
                    'sigma': sigma,
                    'dim_embedding': dim_embedding,
                    'alignment': np.trapz(alignment,dx=1/len(alignment)),
                    'regret': np.trapz(regret,dx=1/len(regret)),
                    'per_query_alignment': np.trapz(per_query_alignment,dx=1/len(per_query_alignment))
                })

pd.DataFrame(results_dataframe).to_csv('cmaes_param_sensitivity.csv', index=False)



