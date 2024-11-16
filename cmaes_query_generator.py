import numpy as np
from cmaes import CMA

class CMAESGenerator:
    def __init__(self, dim, limits, population_size=10, sigma=1.0):
        self.optimizer = CMA(mean=np.zeros(dim), sigma=sigma, population_size=population_size)
        self.sigma = sigma
        self.dimension = dim
        self.population_size = population_size
        self.limits = limits

    def get_query(self, number_items, reward_parameterization=None, input_model=None):
        '''
        
        '''
        queries = []
        for _ in range(number_items):
            x = self.optimizer.ask()
            x = np.clip(x, [lim[0] for lim in self.limits], [lim[1] for lim in self.limits])
            queries.append(x)

        return np.array(queries)
    
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