# CMA-ES-IG: Searching Learned Feature Spaces through Interaction

**[Project page & interactive demo](https://interaction-lab.github.io/CMA-ES-IG/)** · **[Paper (arXiv:2603.09011)](https://arxiv.org/abs/2603.09011)**

CMA-ES-IG is a query-generation algorithm for learning a user's preferences over robot
behaviors from rankings. It generates queries that are both informative and improve over
time, so they stay easy for non-experts to answer while converging on behaviors the user likes.

This repository contains the code and data for the journal version of the work
(*Improving through Interaction: Searching Behavioral Representation Spaces with CMA-ES-IG*).
The original code release for the ISRR 2024 paper is kept unchanged in [`isrr2024/`](isrr2024/).

## Repository layout

| Folder | Contents |
|---|---|
| [`website/`](website/) | Project page with an in-browser demo (rank Jaco2 handovers and watch CMA-ES-IG adapt). Deployed to GitHub Pages by [`.github/workflows/pages.yml`](.github/workflows/pages.yml). |
| [`experiments/`](experiments/) | Hypersphere (parameter-estimation) simulations: query generators, simulation drivers, cached results, and plotting scripts. |
| [`simulated_experiments/`](simulated_experiments/) | Simulations in four robot domains: `lunar_lander`, `driving`, `blossom_voice`, and `pylips_appearance`. Each folder has numbered scripts that build the dataset, run the algorithms, and plot results. |
| [`applications/`](applications/) | Behavior datasets and learned embeddings for the physical robots (Blossom gestures, Jaco2 handovers). |
| [`user_study/`](user_study/) | Web interface and robot controllers used in the in-person user study, plus the per-participant study data in `user_study/data/`. |
| [`rebuttal/`](rebuttal/) | Additional analyses for the revision: the Habibian et al. baseline, the resampling belief, the IG ablation, user-study statistics, figure/table generation, and an MNIST preference explorer. |
| [`figures/`](figures/) | Scripts for paper figures. |
| [`isrr2024/`](isrr2024/) | ISRR 2024 code release: standalone CMA-ES-IG implementation, simulations, and a demo interface. |

## Setup

```bash
git clone --recurse-submodules https://github.com/interaction-lab/CMA-ES-IG.git
cd CMA-ES-IG
pip install -e experiments/preference-learning-from-selection
```

The [`preference-learning-from-selection`](https://github.com/ndennler/preference-learning-from-selection)
submodule provides the Bayesian preference model and the information-gain query generator.

### Data not included

The Blossom voice audio (VoxCeleb source clips and XTTS-generated voices, ~2 GB) is not in this
repository. The speaker embeddings and results derived from it are included. To regenerate the
audio, follow [`simulated_experiments/blossom_voice/readme.txt`](simulated_experiments/blossom_voice/readme.txt).

## Citation

```bibtex
@misc{dennler2026cmaesig,
  title         = {Improving through Interaction: Searching Behavioral Representation Spaces with {CMA-ES-IG}},
  author        = {Dennler, Nathaniel and Shi, Zhonghao and Tao, Yiran and Bobu, Andreea and Nikolaidis, Stefanos and Matari{\'c}, Maja},
  year          = {2026},
  eprint        = {2603.09011},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2603.09011}
}

@inproceedings{dennler2024improving,
  author    = {Dennler, Nathaniel and Shi, Zhonghao and Nikolaidis, Stefanos and Matari{\'c}, Maja},
  title     = {Improving User Experience in Preference-Based Optimization of Reward Functions for Assistive Robots},
  booktitle = {International Symposium on Robotics Research (ISRR)},
  publisher = {IFRR},
  year      = {2024},
  url       = {https://doi.org/10.48550/arXiv.2411.11182}
}
```
