This is the repository for the code we used to perform the study described in:

> Improving User Experience in Preference-Based Optimization of Reward Functions for Assistive Robots

This repository contains the final implementation of CMA-ES-IG---a query-generation algorithm that
efficiently searches robot behavioral representation spaces. This algorithm is specifically useful
for researchers adapting robot behaviors that interact with non-expert users, as the queries
generated with CMA-ES-IG align with the user's preference after each iteration, and the queries 
are easy to answer from the user's perspective.

We provide the code for CMA-ES-IG in the file `cmaesig_query_generation.py`. We also provide the web interface we used
in our experiments to be a resource for other researchers. You will have to supply the code to play a
particular behavior ID on your physical robot though :). We have provided dummy interfaces for you!


# Installation 

1. Clone this repository
2. Install submodules with `git submodule update --init`
3. Install dependencies:
```
pip install -e preference-learning-from-selection
pip install -r requirements.txt
```

# Simulated Results

This repository provides the data we used to evaluate CMA-ES-IG in simulation. To visualize the data, run:

```
python plot_comparisons.py
```

We also include data for regret as a metric. To see this in the graph, edit lines 15-17 and replace "alignment" with "regret".

To re-run the experiments, you can run the code:
```
python simulate_preferences.py
```
you will have to run this at least three times, with dimensions of size 8, 16, and 32. WARNING: the information gain query generation may 
take some time (several hours) to finish.

There are a few key points to keep in mind if you plan to re-use this code for your own purposes:
1. We assume that the features associated with the trajectories are approximately normalized to generally be within the unit ball. You can achieve this in practice by learning trajectory representations with a wieght penalty on the size of the representation, using a KL-divergence loss (a la VAEs), or by scaling your representations post-hoc. 
2. The information gain objective we optimized scales pretty badly with higher feature dimensions and number of items in the query. It may take intractibly long for larger dimensions. While our implementation of CMA-ES-IG scales a bit better, you may find limited usefulness for feature spaces larger than about 100 dimensions or so. If you have feature space that large, you may want to explore methods to reduce its size.

#User Interface

To quickly view the user interface we designed, run this command from the top level directory:
```
python interface/start_interface
```
Then, go to a web browser and type in `localhost:8001/study` to view the interface.

In order to use this interface for your own use-cases, you will need to do a few things:
1. First, you will need a way to generate robot trajectories and their associated features. In this study, we pre-computed these trajectories and features and saved them as numpy arrays under `interface/static/dummy_gestures.npy` for the raw trajectories that we played, and `interface/static/dummy_embeddings.npy` for the associated features for each trajectory. Pre-computing these trajectories can lead to a better user experience, but generating these trajectories and features quickly can also be an interesting area for future research!
2. Next you will have to write the lower-level robot code that takes the trajectories passed in, and plays them on the actual robot. The example script to do this is located at `interface/dummy_controller.py`. Currently, the trajectory execution is just a print statement, so it will need to be updated for your specific use case.
3. If you want to change the algorithm you use to generate queries, you can update the string at the end of `preference_engine.py` from 'CMA-ES-IG' to 'CMA-ES' or 'infogain'. Adding other functions will take a little more work, but you can follow the examples in lines 38-48 of `preference_engine.py`

