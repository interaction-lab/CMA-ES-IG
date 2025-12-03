import sys
import time
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import torch
from dask.distributed import Client

from utils import vector_to_model, count_parameters, LanderAction, LanderMeasureSpace

from ribs.schedulers import Scheduler
from ribs.archives import GridArchive, CVTArchive
from ribs.emitters import EvolutionStrategyEmitter


def simulate(model, seed=None, video_env=None):
    """Simulates the lunar lander model.

    Args:
        model (np.ndarray): The array of weights for the linear policy.
        seed (int): The seed for the environment.
        video_env (gym.Env): If passed in, this will be used instead of creating a new
            env. This is used primarily for recording video during evaluation.
    Returns:
        total_reward (float): The reward accrued by the lander throughout its
            trajectory.
        trajectory (list[np.ndarray]): The list of observations at each timestep.
    """
    if video_env is None:
        env = gym.make("LunarLander-v3")
    else:
        env = video_env

    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    # model = model.reshape((action_dim, obs_dim))
    total_reward = 0.0
    steps = 0
    states = np.zeros((600, obs_dim), dtype=np.float32)  # Pre-allocate array for 600 timesteps
    obs, _ = env.reset(seed=seed)
    done = False

    sum_x_displacement = 0.0
    sum_angular_velocity = 0.0
    earliest_impact = 600

    while not done and steps < 600:
        # Store the current observation
        states[steps] = obs
        steps += 1

        sum_x_displacement += abs(obs[0])
        sum_angular_velocity += abs(obs[5])

        if (obs[6] == 1 or obs[7] == 1) and steps < earliest_impact:  # If either leg has made contact
            earliest_impact = steps


        # action = np.argmax(model @ obs)  # Linear policy.
        output = model(torch.tensor(obs, requires_grad=False)).detach().numpy()
        action = np.argmax(output)  # NN policy.

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    # Only close the env if it was not a video env.
    if video_env is None:
        env.close()

    measures = np.array([
                        (sum_x_displacement - 300) / (steps/2), # Mean absolute x displacement
                        (sum_angular_velocity - 300) / (steps/2), # Mean absolute angular velocity
                        np.mean(states[:steps, 0]) / .75, # Mean x position
                        (np.max(states[:steps, 0]) - 0) / .75, # Max x position
                        (np.min(states[:steps, 0]) + 0) / .75, # Min x position
                        (np.mean(states[:steps, 1]) - 1) / 1.5, # Mean y position
                        (earliest_impact - 300) / (steps/2), # Normalized time of first leg contact
                        states[earliest_impact - 1, 0] / 1., # x position at first leg contact
                        (states[earliest_impact - 1, 3] + 1.3) / 2., # y velocity at first leg contact
                        ])

    return total_reward, torch.from_numpy(states.flatten()), measures


def train(scheduler, archive, measure_model, total_itrs=300, verbose=False, env_seed=42):
    start_time = time.time()

    for itr in trange(1, total_itrs + 1, file=sys.stdout, desc="Iterations"):
        # Request models from the scheduler.
        solutions = scheduler.ask()

        results = [simulate(vector_to_model(model), env_seed) for model in solutions]

        objectives, measures = [], []
        for reward, trajectory, measure in results:
            objectives.append(reward)

            # measure = measure_model(trajectory).detach().numpy()

            measures.append(measure)

        # Send the results back to the scheduler.
        scheduler.tell(objectives, measures)

        # Logging.
        if (itr % 25 == 0 or itr == total_itrs) and verbose:
            # fmt: off
            tqdm.write(f"> {itr} itrs completed after {time.time() - start_time:.2f}s")
            tqdm.write(f"  - Size: {archive.stats.num_elites}")    # Number of elites in the archive. len(archive) also provides this info.
            tqdm.write(f"  - Coverage: {archive.stats.coverage}")  # Proportion of archive cells which have an elite.
            tqdm.write(f"  - QD Score: {archive.stats.qd_score}")  # QD score, i.e. sum of objective values of all elites in the archive.
                                                                # Accounts for qd_score_offset as described in the GridArchive section.
            tqdm.write(f"  - Max Obj: {archive.stats.obj_max}")    # Maximum objective value in the archive.
            tqdm.write(f"  - Mean Obj: {archive.stats.obj_mean}")  # Mean objective value of elites in the archive.
            # fmt: on

if __name__ == "__main__":
    # Create an environment so that we can obtain information about it.
    reference_env = gym.make("LunarLander-v3")
    action_dim = reference_env.action_space.n
    obs_dim = reference_env.observation_space.shape[0]
    model_size = count_parameters(LanderAction())

    # Test out the simulation with a random policy.
    reward, traj, measures = simulate(vector_to_model(np.random.randn(model_size)))
    print("Reward from random policy:", reward)
    print("Trajectory shape:", traj.shape)  # Should be (T, 8).
    print("Measures shape:", measures.shape)  # Should be (1, 9).
    n_measures = len(measures)

    # # Test out the measure model.
    measure_model = LanderMeasureSpace(input_size=len(traj), dim_embedding=2)
    # measures = measure_model(traj)
    # print("Measures shape:", measures.shape)  # Should be (1, 2).

    # for run in range(5):
    #     for embedding_size in [2, 4, 6, 8]:

    #         print(f"\n\nStarting run {run} with embedding size {embedding_size}...\n\n")
    #         # set up the pyribs objects
    #         initial_model = np.zeros(model_size)
    #         measure_model = LanderMeasureSpace(input_size=len(traj), dim_embedding=embedding_size)

    #         archive = CVTArchive(
    #             solution_dim=model_size,  # Dimensionality of solutions in the archive.
    #             cells=50_000,  # 50000 cells.
    #             ranges=[(-1.0, 1.0)] * embedding_size,  # (-1, 1) for x-pos and (-3, 0) for y-vel.
    #             learning_rate=0.3,
    #             threshold_min= -300,
    #             qd_score_offset=-600,  # See the note below.
    #         )

    #         emitters = [
    #             EvolutionStrategyEmitter(
    #                 archive=archive,
    #                 x0=initial_model,
    #                 sigma0=1.0,  # Initial step size.
    #                 ranker="imp",
    #                 selection_rule="mu",
    #                 restart_rule="basic",
    #                 # ranker="2imp",
    #                 # If we do not specify a batch size, the emitter will automatically use a
    #                 # batch size equal to the default population size of CMA-ES.
    #                 batch_size=25,
    #             )
    #             for _ in range(5)  # Create 5 separate emitters.
    #         ]

    #         scheduler = Scheduler(archive, emitters)

    #         train(scheduler, archive, measure_model, total_itrs=500, verbose=True, env_seed=42)
    #         scheduler.archive.data(return_type="pandas").to_csv(f"./data/embeddings/lunar_lander_dim{embedding_size}_run{run}.csv")
            
            
    print(f"\n\nStarting run with handcrafted features...\n\n")
    # set up the pyribs objects
    initial_model = np.zeros(model_size)
    # measure_model = LanderMeasureSpace(input_size=len(traj), dim_embedding=embedding_size)

    archive = CVTArchive(
        solution_dim=model_size,  # Dimensionality of solutions in the archive.
        cells=100_000,  # 100000 cells.
        ranges=[(-0.8, 0.8)] * n_measures,  # all values normalized to (-1, 1)
        learning_rate=0.1,
        threshold_min= -300,
        qd_score_offset=-600,  # See the note below.
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive=archive,
            x0=initial_model,
            sigma0=1.2,  # Initial step size.
            ranker="imp",
            selection_rule="mu",
            restart_rule="basic",
            # ranker="2imp",
            # If we do not specify a batch size, the emitter will automatically use a
            # batch size equal to the default population size of CMA-ES.
            batch_size=25,
        )
        for _ in range(25)  # Create 5 separate emitters.
    ]
   

    scheduler = Scheduler(archive, emitters)

    train(scheduler, archive, measure_model, total_itrs=1500, verbose=True, env_seed=42)
    scheduler.archive.data(return_type="pandas").to_csv(f"./data/embeddings/lunar_lander_handcrafted.csv")
            
            
