# Regenerated results with the Habibian et al. baseline

AUC convention: trapezoid(y) / (number of intervals), i.e. the curve average;
regret is normalized by each user's reward range. Welch t-tests. 100 simulated users, 30 queries, 4 items (hypersphere)
and 5 items (robot domains).

## Pairwise t-tests, CMA-ES-IG vs each baseline

### Hypersphere (Table 1 / Figure 2)

| metric | d | baseline | CMA-ES-IG | baseline mean | t | p | Cohen d |
|---|---|---|---|---|---|---|---|
| alignment | 4 | Infogain | 0.968 | 0.974 | -2.41 | 1.73e-02 | -0.34 |
| alignment | 4 | Informative+Revealing | 0.968 | 0.971 | -1.00 | 3.19e-01 | -0.14 |
| alignment | 4 | CMA-ES | 0.968 | 0.968 | 0.20 | 8.40e-01 | +0.03 |
| alignment | 8 | Infogain | 0.949 | 0.954 | -3.37 | 9.03e-04 | -0.48 |
| alignment | 8 | Informative+Revealing | 0.949 | 0.951 | -1.40 | 1.62e-01 | -0.20 |
| alignment | 8 | CMA-ES | 0.949 | 0.949 | -0.42 | 6.77e-01 | -0.06 |
| alignment | 16 | Infogain | 0.904 | 0.915 | -4.15 | 4.92e-05 | -0.59 |
| alignment | 16 | Informative+Revealing | 0.904 | 0.914 | -4.05 | 7.63e-05 | -0.57 |
| alignment | 16 | CMA-ES | 0.904 | 0.909 | -1.86 | 6.37e-02 | -0.26 |
| alignment | 32 | Infogain | 0.817 | 0.822 | -1.35 | 1.79e-01 | -0.19 |
| alignment | 32 | Informative+Revealing | 0.817 | 0.833 | -4.78 | 3.51e-06 | -0.68 |
| alignment | 32 | CMA-ES | 0.817 | 0.830 | -3.55 | 4.79e-04 | -0.50 |
| regret | 4 | Infogain | 0.007 | 0.004 | 2.34 | 2.09e-02 | +0.33 |
| regret | 4 | Informative+Revealing | 0.007 | 0.006 | 0.53 | 5.94e-01 | +0.08 |
| regret | 4 | CMA-ES | 0.007 | 0.007 | -0.26 | 7.93e-01 | -0.04 |
| regret | 8 | Infogain | 0.013 | 0.012 | 0.95 | 3.45e-01 | +0.13 |
| regret | 8 | Informative+Revealing | 0.013 | 0.014 | -0.56 | 5.74e-01 | -0.08 |
| regret | 8 | CMA-ES | 0.013 | 0.013 | -0.36 | 7.22e-01 | -0.05 |
| regret | 16 | Infogain | 0.036 | 0.031 | 1.90 | 5.88e-02 | +0.27 |
| regret | 16 | Informative+Revealing | 0.036 | 0.029 | 2.84 | 5.08e-03 | +0.40 |
| regret | 16 | CMA-ES | 0.036 | 0.032 | 1.37 | 1.73e-01 | +0.19 |
| regret | 32 | Infogain | 0.084 | 0.077 | 1.65 | 1.01e-01 | +0.23 |
| regret | 32 | Informative+Revealing | 0.084 | 0.071 | 3.20 | 1.59e-03 | +0.45 |
| regret | 32 | CMA-ES | 0.084 | 0.074 | 2.41 | 1.69e-02 | +0.34 |
| per_query_alignment | 4 | Infogain | 0.833 | -0.334 | 142.03 | 2.02e-199 | +20.09 |
| per_query_alignment | 4 | Informative+Revealing | 0.833 | 0.463 | 51.94 | 5.23e-103 | +7.34 |
| per_query_alignment | 4 | CMA-ES | 0.833 | 0.805 | 3.20 | 1.59e-03 | +0.45 |
| per_query_alignment | 8 | Infogain | 0.789 | -0.162 | 191.28 | 2.19e-222 | +27.05 |
| per_query_alignment | 8 | Informative+Revealing | 0.789 | 0.325 | 109.63 | 2.33e-142 | +15.50 |
| per_query_alignment | 8 | CMA-ES | 0.789 | 0.715 | 11.53 | 2.22e-23 | +1.63 |
| per_query_alignment | 16 | Infogain | 0.685 | -0.100 | 166.90 | 4.83e-194 | +23.60 |
| per_query_alignment | 16 | Informative+Revealing | 0.685 | 0.224 | 111.79 | 1.00e-125 | +15.81 |
| per_query_alignment | 16 | CMA-ES | 0.685 | 0.581 | 13.97 | 1.08e-29 | +1.98 |
| per_query_alignment | 32 | Infogain | 0.513 | -0.080 | 120.17 | 3.07e-146 | +16.99 |
| per_query_alignment | 32 | Informative+Revealing | 0.513 | 0.148 | 79.93 | 5.41e-100 | +11.30 |
| per_query_alignment | 32 | CMA-ES | 0.513 | 0.396 | 18.18 | 5.00e-44 | +2.57 |

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
