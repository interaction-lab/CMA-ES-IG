# Regenerated results with the Habibian et al. baseline

AUC convention: trapezoid(y) / (number of intervals), i.e. the curve average;
regret is normalized by each user's reward range. Welch t-tests. 100 simulated users, 30 queries, 4 items (hypersphere)
and 5 items (robot domains).

## Pairwise t-tests, CMA-ES-IG vs each baseline

### Hypersphere (Table 1 / Figure 2)

| metric | d | baseline | CMA-ES-IG | baseline mean | t | p | Cohen d |
|---|---|---|---|---|---|---|---|
| alignment | 4 | Infogain | 0.959 | 0.968 | -5.17 | 6.39e-07 | -0.73 |
| alignment | 4 | Informative+Revealing | 0.959 | 0.961 | -0.68 | 4.97e-01 | -0.10 |
| alignment | 4 | CMA-ES | 0.959 | 0.958 | 0.70 | 4.88e-01 | +0.10 |
| alignment | 8 | Infogain | 0.926 | 0.938 | -4.85 | 2.49e-06 | -0.69 |
| alignment | 8 | Informative+Revealing | 0.926 | 0.932 | -2.68 | 7.88e-03 | -0.38 |
| alignment | 8 | CMA-ES | 0.926 | 0.929 | -1.52 | 1.31e-01 | -0.21 |
| alignment | 16 | Infogain | 0.862 | 0.880 | -5.15 | 6.21e-07 | -0.73 |
| alignment | 16 | Informative+Revealing | 0.862 | 0.879 | -4.96 | 1.53e-06 | -0.70 |
| alignment | 16 | CMA-ES | 0.862 | 0.869 | -2.02 | 4.51e-02 | -0.29 |
| alignment | 32 | Infogain | 0.754 | 0.754 | 0.08 | 9.39e-01 | +0.01 |
| alignment | 32 | Informative+Revealing | 0.754 | 0.776 | -4.73 | 4.44e-06 | -0.67 |
| alignment | 32 | CMA-ES | 0.754 | 0.762 | -1.50 | 1.34e-01 | -0.21 |
| regret | 4 | Infogain | 0.011 | 0.006 | 3.68 | 3.35e-04 | +0.52 |
| regret | 4 | Informative+Revealing | 0.011 | 0.008 | 2.07 | 4.00e-02 | +0.29 |
| regret | 4 | CMA-ES | 0.011 | 0.012 | -0.09 | 9.26e-01 | -0.01 |
| regret | 8 | Infogain | 0.027 | 0.020 | 2.47 | 1.48e-02 | +0.35 |
| regret | 8 | Informative+Revealing | 0.027 | 0.020 | 2.32 | 2.16e-02 | +0.33 |
| regret | 8 | CMA-ES | 0.027 | 0.021 | 1.93 | 5.59e-02 | +0.27 |
| regret | 16 | Infogain | 0.056 | 0.048 | 2.28 | 2.37e-02 | +0.32 |
| regret | 16 | Informative+Revealing | 0.056 | 0.051 | 1.34 | 1.83e-01 | +0.19 |
| regret | 16 | CMA-ES | 0.056 | 0.051 | 1.43 | 1.53e-01 | +0.20 |
| regret | 32 | Infogain | 0.116 | 0.112 | 0.77 | 4.45e-01 | +0.11 |
| regret | 32 | Informative+Revealing | 0.116 | 0.106 | 1.76 | 8.05e-02 | +0.25 |
| regret | 32 | CMA-ES | 0.116 | 0.109 | 1.41 | 1.62e-01 | +0.20 |
| per_query_alignment | 4 | Infogain | 0.834 | -0.287 | 153.46 | 2.21e-192 | +21.70 |
| per_query_alignment | 4 | Informative+Revealing | 0.834 | 0.510 | 45.53 | 1.04e-97 | +6.44 |
| per_query_alignment | 4 | CMA-ES | 0.834 | 0.774 | 5.43 | 1.99e-07 | +0.77 |
| per_query_alignment | 8 | Infogain | 0.772 | -0.140 | 177.32 | 2.64e-183 | +25.08 |
| per_query_alignment | 8 | Informative+Revealing | 0.772 | 0.357 | 84.45 | 1.90e-121 | +11.94 |
| per_query_alignment | 8 | CMA-ES | 0.772 | 0.687 | 10.97 | 1.04e-21 | +1.55 |
| per_query_alignment | 16 | Infogain | 0.650 | -0.089 | 128.14 | 4.62e-141 | +18.12 |
| per_query_alignment | 16 | Informative+Revealing | 0.650 | 0.246 | 71.98 | 3.45e-101 | +10.18 |
| per_query_alignment | 16 | CMA-ES | 0.650 | 0.530 | 14.67 | 3.07e-33 | +2.07 |
| per_query_alignment | 32 | Infogain | 0.474 | -0.065 | 95.58 | 2.84e-119 | +13.52 |
| per_query_alignment | 32 | Informative+Revealing | 0.474 | 0.159 | 57.98 | 7.86e-85 | +8.20 |
| per_query_alignment | 32 | CMA-ES | 0.474 | 0.350 | 17.26 | 2.94e-41 | +2.44 |

### Robot domains (Table 3 / Figure 4)

| domain | metric | baseline | CMA-ES-IG | baseline mean | t | p | Cohen d |
|---|---|---|---|---|---|---|---|
| Lunar Lander | alignment | Infogain | 0.974 | 0.990 | -6.17 | 6.97e-09 | -0.87 |
| Lunar Lander | alignment | Informative+Revealing | 0.974 | 0.982 | -2.84 | 5.17e-03 | -0.40 |
| Lunar Lander | alignment | CMA-ES | 0.974 | 0.977 | -0.97 | 3.35e-01 | -0.14 |
| Lunar Lander | regret | Infogain | 0.014 | 0.005 | 2.86 | 4.98e-03 | +0.40 |
| Lunar Lander | regret | Informative+Revealing | 0.014 | 0.010 | 1.10 | 2.75e-01 | +0.15 |
| Lunar Lander | regret | CMA-ES | 0.014 | 0.011 | 0.97 | 3.32e-01 | +0.14 |
| Lunar Lander | quality_avg | Infogain | 0.825 | -0.182 | 119.84 | 2.83e-183 | +16.95 |
| Lunar Lander | quality_avg | Informative+Revealing | 0.825 | 0.462 | 51.08 | 1.08e-93 | +7.22 |
| Lunar Lander | quality_avg | CMA-ES | 0.825 | 0.785 | 3.21 | 1.63e-03 | +0.45 |
| Driving | alignment | Infogain | 0.978 | 0.990 | -5.96 | 2.47e-08 | -0.84 |
| Driving | alignment | Informative+Revealing | 0.978 | 0.981 | -1.33 | 1.86e-01 | -0.19 |
| Driving | alignment | CMA-ES | 0.978 | 0.979 | -0.23 | 8.17e-01 | -0.03 |
| Driving | regret | Infogain | 0.011 | 0.005 | 2.87 | 4.90e-03 | +0.41 |
| Driving | regret | Informative+Revealing | 0.011 | 0.010 | 0.80 | 4.25e-01 | +0.11 |
| Driving | regret | CMA-ES | 0.011 | 0.010 | 0.81 | 4.17e-01 | +0.12 |
| Driving | quality_avg | Infogain | 0.821 | -0.176 | 123.19 | 3.05e-188 | +17.42 |
| Driving | quality_avg | Informative+Revealing | 0.821 | 0.463 | 50.26 | 1.08e-104 | +7.11 |
| Driving | quality_avg | CMA-ES | 0.821 | 0.794 | 2.89 | 4.31e-03 | +0.41 |
| Robot Face Design | alignment | Infogain | 0.980 | 0.992 | -7.80 | 1.88e-12 | -1.10 |
| Robot Face Design | alignment | Informative+Revealing | 0.980 | 0.981 | -0.25 | 8.06e-01 | -0.03 |
| Robot Face Design | alignment | CMA-ES | 0.980 | 0.972 | 3.37 | 9.25e-04 | +0.48 |
| Robot Face Design | regret | Infogain | 0.009 | 0.004 | 2.19 | 3.01e-02 | +0.31 |
| Robot Face Design | regret | Informative+Revealing | 0.009 | 0.009 | -0.32 | 7.49e-01 | -0.05 |
| Robot Face Design | regret | CMA-ES | 0.009 | 0.016 | -2.02 | 4.49e-02 | -0.29 |
| Robot Face Design | quality_avg | Infogain | 0.823 | -0.183 | 104.32 | 1.22e-174 | +14.75 |
| Robot Face Design | quality_avg | Informative+Revealing | 0.823 | 0.468 | 48.52 | 1.69e-89 | +6.86 |
| Robot Face Design | quality_avg | CMA-ES | 0.823 | 0.785 | 3.15 | 1.95e-03 | +0.44 |
| Robot Voice Design | alignment | Infogain | 0.974 | 0.991 | -4.95 | 2.60e-06 | -0.70 |
| Robot Voice Design | alignment | Informative+Revealing | 0.974 | 0.978 | -1.06 | 2.90e-01 | -0.15 |
| Robot Voice Design | alignment | CMA-ES | 0.974 | 0.979 | -1.30 | 1.96e-01 | -0.18 |
| Robot Voice Design | regret | Infogain | 0.011 | 0.005 | 3.25 | 1.49e-03 | +0.46 |
| Robot Voice Design | regret | Informative+Revealing | 0.011 | 0.011 | -0.07 | 9.46e-01 | -0.01 |
| Robot Voice Design | regret | CMA-ES | 0.011 | 0.011 | 0.14 | 8.90e-01 | +0.02 |
| Robot Voice Design | quality_avg | Infogain | 0.824 | -0.182 | 115.66 | 3.51e-180 | +16.36 |
| Robot Voice Design | quality_avg | Informative+Revealing | 0.824 | 0.463 | 46.80 | 1.08e-95 | +6.62 |
| Robot Voice Design | quality_avg | CMA-ES | 0.824 | 0.801 | 2.60 | 9.96e-03 | +0.37 |

## Runtime (Table 2), measured on this machine

| algorithm | d=4 | d=8 | d=16 | d=32 |
|---|---|---|---|---|
| CMA-ES-IG | 0.8 ms | 0.8 ms | 0.8 ms | 11.7 ms |
| CMA-ES | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| Infogain | 70.5 ms | 382.0 ms | 2771.0 ms | 7147.0 ms |
| Informative+Revealing | 35.2 ms | 38.7 ms | 44.8 ms | 54.4 ms |
