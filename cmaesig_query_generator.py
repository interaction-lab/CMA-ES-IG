import numpy as np
from cmaes import CMA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import cdist

class CMAESIGGenerator:
    def __init__(self, dim, limits, population_size=10, use_boundary_mediods=False, sigma=1.0):
        self.optimizer = CMA(mean=np.zeros(dim), sigma=sigma, population_size=population_size)
        self.dimension = dim
        self.limits = limits
        self.population_size = population_size
        self.sigma = sigma
        self.use_boundary_mediods = use_boundary_mediods

    def _info_gain(self, reward_parameterization, input_model, query):
        '''
        Calculates the information gain of a query under a reward parameterization and human input model.
        This equation is based on eq. 5 from https://arxiv.org/pdf/1910.04365
        Args:
            reward_parameterization (RewardParameterization): The reward parameterization provided by the 
                irl preference package (for example, MonteCarloLinearReward).
            input_model (InputModel): The input model to use, for example, LuceShepardChoice.
            query (np.ndarray): The query to calculate the information gain of---The query should be of
                shape (n, d) where n is the number of queries and d is the number of features for each 
                item in the query.
        '''
        return reward_parameterization.get_best_entropy(query, input_model) - \
                reward_parameterization.get_human_entropy(query, input_model)
    
    def get_query(self, number_items, reward_parameterization=None, input_model=None, ):
        '''
        returns a list of items to present to the user, where each item is a feature vector.

        Args:
            number_items (int): the number of items to present to the user.
            reward_parameterization (RewardParameterization): The reward parameterization provided by the 
                irl preference package (for example, MonteCarloLinearReward).
            input_model (InputModel): The input model that represents how users may rank items, for example, 
                LuceShepardChoice. 
        '''

        candidates = []
        for _ in range(200):
            x = self.optimizer.ask()
            x = np.clip(x, [lim[0] for lim in self.limits], [lim[1] for lim in self.limits])
            candidates.append(x)


        #uses boundary medoids selection from https://proceedings.mlr.press/v87/biyik18a/biyik18a.pdf
        if self.use_boundary_mediods:
            kmeans = KMedoids(n_clusters=number_items, init='k-medoids++').fit(candidates)
            query = kmeans.cluster_centers_
        else:
            kmeans = KMeans(n_clusters=number_items, n_init="auto").fit(candidates)
            query = kmeans.cluster_centers_
            
        return query

    
    def tell(self, solutions, rankings):
        '''
        allows the CMA-ES optimizer to learn from the rankings of the solutions.

        Args:
            solutions (list): a list of features that correspond to the items
                presented to the user.
            rankings (list): the index that the user ranked that particular item
                at. For example, if the user ranked the first item in `solutions`
                as the worst, the second item as the best, and the third
                item as the second best, then `rankings` would be [2, 0, 1].
        '''
        answer = []
    
        for i, solution in enumerate(solutions):
            answer.append((solution, -rankings[i]))

        self.optimizer.tell(answer)
    
    def reset(self):
        '''
        Resets the optimizer to its initial state.
        '''
        self.optimizer = CMA(mean=np.zeros(self.dimension), sigma=self.sigma, population_size=self.population_size)
