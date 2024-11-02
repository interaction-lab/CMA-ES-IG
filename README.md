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

The algorithmic implementation is provided in 