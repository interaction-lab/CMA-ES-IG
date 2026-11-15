import cmaes
import scipy.optimize as opt
import numpy as np
from cmaes import CMA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2


class CMAESGenerator:
    def __init__(self, dim, limits, population_size=10, sigma=1.3):
        self.optimizer = CMA(mean=np.zeros(dim), sigma=sigma, population_size=population_size)
        self.dimension = dim
        self.population_size = population_size
        self.limits = limits
        self.sigma = sigma

    def get_query(self, number_queries, reward_parameterization=None, input_model=None):
        '''
        '''
        queries = []
        for _ in range(number_queries):
            x = self.optimizer.ask()
            x = np.clip(x, -1, 1)# todo: change this to self.limits
            queries.append(x)

        return np.array(queries)
    
    def tell(self, solutions, rankings):
        '''
        '''
        answer = []
        for i, solution in enumerate(solutions):
            answer.append((solution, -rankings[i]))

        self.optimizer.tell(answer)

    def reset(self):
        self.optimizer = CMA(mean=np.zeros(self.dimension), sigma=self.sigma, population_size=self.population_size)




class CMAESIGGenerator:
    def __init__(self, dim, limits, population_size=10, use_boundary_mediods=False, sigma=1.3):
        self.optimizer = CMA(mean=np.zeros(dim), sigma=sigma, population_size=population_size)
        self.dimension = dim
        self.limits = limits
        self.population_size = population_size
        self.use_boundary_mediods = use_boundary_mediods
        self.sigma = sigma

    def _info_gain(self, reward_parameterization, input_model, query):
        '''

        '''
        return reward_parameterization.get_best_entropy(query, input_model) - \
                reward_parameterization.get_human_entropy(query, input_model)
    
    def get_query(self, number_queries, reward_parameterization=None, input_model=None, ):
        '''
        
        '''

        candidates = []
        for _ in range(100):
            x = self.optimizer.ask()
            x = np.clip(x, -1, 1)
            candidates.append(x)

        kmeans = KMeans(n_clusters=number_queries, n_init="auto").fit(candidates)
        query = kmeans.cluster_centers_

        return query

    
    def tell(self, solutions, rankings):
        '''
        '''
        answer = []

        for i, solution in enumerate(solutions):
            answer.append((solution, -rankings[i]))

        self.optimizer.tell(answer)
    
    def reset(self):
        self.optimizer = CMA(mean=np.zeros(self.dimension), sigma=self.sigma, population_size=self.population_size)



class DiscreteCMAESIGGenerator:
    def __init__(self, dim, limits, trajectory_samples, population_size=10, sigma=1.3):
        self.optimizer = CMA(mean=np.zeros(dim), sigma=sigma, population_size=population_size)
        self.dimension = dim
        self.limits = limits
        self.population_size = population_size
        self.sigma = sigma
        self.trajectory_samples = trajectory_samples
    
    def _info_gain(self, reward_parameterization, input_model, query):
        '''

        '''
        return reward_parameterization.get_best_entropy(query, input_model) - \
                reward_parameterization.get_human_entropy(query, input_model)

    def get_query(self, number_queries, reward_parameterization=None, input_model=None):
        '''
        '''
        candidates = []

        for _ in range(100):
            x = self.optimizer.ask()
            # get nearest trajectory sample
            # print(x.shape, self.trajectory_samples.shape)
            x = self.trajectory_samples[np.argmin(np.linalg.norm(self.trajectory_samples - x, axis=1))]
            x = np.clip(x, -1, 1)  # Discretize to -1 or 1
            candidates.append(x)

        kmeans = KMeans(n_clusters=number_queries, n_init="auto").fit(candidates)
        query = kmeans.cluster_centers_

        return np.array(query)
    
    def tell(self, solutions, rankings):
        '''
        '''
        answer = []
        for i, solution in enumerate(solutions):
            answer.append((solution, -rankings[i]))

        self.optimizer.tell(answer)

    def reset(self):
        self.optimizer = CMA(mean=np.zeros(self.dimension), sigma=self.sigma, population_size=self.population_size)



if __name__ == "__main__":
    optimizer = CMA(mean=np.zeros(2), sigma=1.3)

    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            print(x)
            value = quadratic(x[0], x[1])
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)