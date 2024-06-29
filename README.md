# preference-based-optimization
ISRR 2024 submission, evaluating CMA-ES vs. Bayesian IRL methods for preference-based optimization of robot policies.

We will explore 4 different applications for preference learning that can be used to adapt different robot modalities:
- Lunar Lander, to specify how a robot can land in a goal in different ways
- Robot Handovers, to specify how a robot can provide assistance to humans in different ways
- Blossom Voice, to specify how robots can provide social assistance
- Blossom Gestures, to specify how robots can provide social nonverbal cues

Each of these will have different techniques to generate the behaviors, but we are interested
in the preference learning process. The overarching structure for each of these applications are:

1. Generate or collect a diverse set of behaviors
2. Generate a representation function for these behaviors
3. Learn a user's preference over this representation

To run this study, make sure you have the files for the specific application. Place those files in the static/data folder.

Then run ` python3 start_interface.py`