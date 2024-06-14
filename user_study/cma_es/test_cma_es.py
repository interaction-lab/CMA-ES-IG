from cmaes import CMA
import numpy as np

#defining a function
def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

#asking users for ranking 5 options 10 times
if __name__ == "__main__":
    optimizer = CMA(mean=np.zeros(2), sigma=1.3, population_size=5)

    for generation in range(10):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = quadratic(x[0], x[1])
            solutions.append((x, value))
            if generation % 1 == 0:
                print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        
        ranked_solutions = [(solution[0], i) for i, solution in enumerate(sorted(solutions, key=lambda x: x[1]))]
        optimizer.tell(ranked_solutions)