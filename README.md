# CMA-ES-IG for Efficient Preference-Based Exploration of Learned Representation Spaces

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code used in the research paper:

> **Improving User Experience in Preference-Based Optimization of Reward Functions for Assistive Robots**

This work introduces **CMA-ES-IG**, a novel query-generation algorithm designed for efficient exploration of learned representation spaces. CMA-ES-IG is particularly valuable for researchers adapting robot behaviors through interactions with non-expert users. The algorithm generates queries that align with the user's preferences over time, simultaneously selecting queries are intuitive and easy for users to answer.

## Contents

We provide the code for CMA-ES-IG in the file `cmaesig_query_generation.py`. We also provide the web interface we used
in our experiments to be a resource for other researchers. You will have to supply the code to play a
particular behavior ID on your physical robot though :). We have provided dummy interfaces for you!


* [`cmaesig_query_generation.py`](cmaesig_query_generation.py): Implementation of the CMA-ES-IG query generation algorithm.
* [`cmaes_query_generation.py`](cmaesig_query_generation.py): Implementation of the CMA-ES query generation baseline algorithm.
* [`preference-learning-from-selection/`](preference-learning-from-selection/): Submodule for preference learning library (contains InformationGain query generator implementation).
* [`plot_comparisons.py`](plot_comparisons.py): Script to visualize simulated results.
* [`simulate_preferences.py`](simulate_preferences.py): Script to run preference simulation experiments.
* [`requirements.txt`](requirements.txt): List of Python dependencies.

* [`results/`](results/): Contains the cached experimental results from our simulations, including alignment and regret metrics for different algorithms and dimensionality settings.
* [`interface/`](interface/): Contains the web interface used in our experiments.
    * [`start_interface`](interface/start_interface): Script to launch the web interface.
    * [`static/`](interface/static/): Static files for the web interface (e.g., dummy data).
        * [`dummy_gestures.npy`](interface/static/dummy_gestures.npy): Dummy robot trajectories.
        * [`dummy_embeddings.npy`](interface/static/dummy_embeddings.npy): Dummy feature embeddings for trajectories.
    * [`dummy_controller.py`](interface/dummy_controller.py): Example script for controlling a robot (currently a dummy implementation).
    * [`preference_engine.py`](interface/preference_engine.py): Backend logic for the preference learning interface.

## Installation

Follow these steps to set up the repository:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/interaction-lab/CMA-ES-IG.git
    cd CMA-ES-IG
    ```

2.  **Initialize submodules:**
    ```bash
    git submodule update --init
    ```

3.  **Set up a Conda environment (recommended):**
    ```bash
    conda create -n cmaesig python=3.10
    conda activate cmaesig
    ```

4.  **Install dependencies:**
    ```bash
    pip install -e preference-learning-from-selection
    pip install -r requirements.txt
    ```
## Usage

### 1. Evaluating Simulated Performance

This repository includes the simulation data and scripts used in our publication.

* **Visualize Pre-computed Results:**
    To plot the comparison graphs (e.g., alignment vs. iteration) from the paper:
    ```bash
    python generate_paper_data/plot_comparisons.py
    ```
    *Note: To plot 'regret' instead of 'alignment', use the `--use-regret` flag.*

    You can also view the parameter sensitivity plot, and print out the table data with:

    ```bash
    python generate_paper_data/plot_sensitivity_data.py
    python generate_paper_data/show_table1_data.py
    ```

* **Re-run Simulations:**
    To reproduce the simulation experiments:
    ```bash
    python simulate_preferences.py --dim <dimension>
    ```
    Replace `<dimension>` with the desired feature space dimensionality (e.g., 8, 16, 32, as used in the paper).

    **⚠️ Performance Advisory:** The information gain calculation, especially within the `infogain` and consequently `CMA-ES-IG` methods, exhibits significant computational cost that scales with the feature dimensionality and the number of items per query. Simulations, particularly for higher dimensions (e.g., 32), may require several hours to complete.

### 2. Running the User Interface

A demonstration web interface is provided to provide a starting point for other people.

* **Launch the Interface:**
    From the top-level directory of the repository, run:
    ```bash
    python interface/start_interface.py
    ```

* **Access the Interface:**
    Open a web browser and navigate to `http://localhost:8001/study`.

### 3. Integrating with a Physical Robot or Custom System

To adapt this interface for your specific robotic platform or application:

1.  **Trajectory and Feature Generation:**
    * The system requires robot trajectories (raw command sequences) and corresponding feature vectors (embeddings).
    * The demo uses pre-computed data stored in `interface/static/dummy_gestures.npy` (trajectories) and `interface/static/dummy_embeddings.npy` (features).
    * **Your Implementation:** You must provide mechanisms to:
        * Generate or load your robot's trajectories.
        * Generate or load the corresponding feature vectors for these trajectories. Ensure these features adhere to the normalization assumption (see Technical Considerations).
        * Replace the dummy `.npy` files or modify the data loading logic within the interface code accordingly. Pre-computation often yields a smoother user experience.

2.  **Robot Control Backend:**
    * The interface communicates with a backend process to execute robot behaviors. The placeholder is `interface/dummy_controller.py`.
    * **Your Implementation:** Modify or replace `dummy_controller.py` with code that receives a trajectory ID (or the trajectory itself) and commands your physical robot hardware to execute the corresponding behavior. The current implementation merely prints the action; this needs to be replaced with your robot's specific API calls.

3.  **Algorithm Selection:**
    * The query generation algorithm can be configured in `preference_engine.py`.
    * To switch between implemented algorithms, modify the string assignment near the end of the file (e.g., from `'CMA-ES-IG'` to `'CMA-ES'` or `'infogain'`).
    * To add custom query generation strategies, follow the structure exemplified in lines 38-48 of `preference_engine.py`, implementing the required interface for your new algorithm.

## Technical Considerations

1.  **Feature Space Normalization:**
    * The underlying preference learning models and the CMA-ES-IG algorithm generally assume that feature vectors are normalized, ideally residing approximately within a unit ball.
    * **Practical Steps:** Ensure your feature extraction process incorporates normalization. Techniques include:
        * Applying L2 weight penalties during representation learning.
        * Utilizing KL-divergence regularization (common in VAEs).
        * Performing post-hoc scaling (e.g., L2 normalization) of generated feature vectors.

2.  **Scalability:**
    * The computational complexity of the information gain objective is sensitive to both the dimensionality of the feature space and the number of items presented per query.
    * While CMA-ES-IG offers improved scaling compared to pure information gain optimization, its practical application may become computationally intensive for feature spaces significantly larger than approximately 100 dimensions.
    * For very high-dimensional spaces, consider employing dimensionality reduction techniques (e.g., PCA, Autoencoders) prior to applying preference-based optimization.

## Contributing

Contributions are welcome! Please follow standard practices: Fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to existing coding style is appreciated.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or the CMA-ES-IG algorithm in your research, please cite our publication:

```bibtex
@inproceedings{dennler2024improving,
  author    = {Dennler, Nathaniel and Shi, Zhonghao and Nikolaidis, Stefanos and Matari{\'c}, Maja},
  title     = {Improving User Experience in Preference-Based Optimization of Reward Functions for Assistive Robots},
  booktitle = {International Symposium on Robotics Research (ISRR)},
  publisher = {IFRR},
  year      = {2024},
  url       = {https://doi.org/10.48550/arXiv.2411.11182}
}
```
