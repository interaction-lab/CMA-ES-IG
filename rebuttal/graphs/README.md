# Regenerated results with the Habibian et al. baseline

AUC convention: `trapezoid(y) / 30`, reproducing the published Infogain row of
Table 1 to three decimals. 100 simulated users, 30 queries, 4 items (hypersphere)
and 5 items (robot domains).

## Pairwise t-tests, CMA-ES-IG vs each baseline

### Hypersphere (Table 1 / Figure 2)

| metric | d | baseline | CMA-ES-IG | baseline mean | t | p | Cohen d |
|---|---|---|---|---|---|---|---|
| alignment | 4 | Infogain | 0.958 | 0.951 | 2.68 | 8.06e-03 | +0.38 |
| alignment | 4 | Informative+Revealing | 0.958 | 0.962 | -2.00 | 4.71e-02 | -0.28 |
| alignment | 4 | CMA-ES | 0.958 | 0.955 | 0.92 | 3.59e-01 | +0.13 |
| alignment | 8 | Infogain | 0.901 | 0.860 | 8.89 | 3.86e-16 | +1.26 |
| alignment | 8 | Informative+Revealing | 0.901 | 0.898 | 0.68 | 4.94e-01 | +0.10 |
| alignment | 8 | CMA-ES | 0.901 | 0.889 | 2.90 | 4.16e-03 | +0.41 |
| alignment | 16 | Infogain | 0.728 | 0.644 | 9.56 | 4.94e-18 | +1.35 |
| alignment | 16 | Informative+Revealing | 0.728 | 0.704 | 3.65 | 3.38e-04 | +0.52 |
| alignment | 16 | CMA-ES | 0.728 | 0.683 | 5.34 | 2.54e-07 | +0.76 |
| alignment | 32 | Infogain | 0.497 | 0.383 | 8.57 | 5.92e-15 | +1.28 |
| alignment | 32 | Informative+Revealing | 0.497 | 0.457 | 4.43 | 1.57e-05 | +0.63 |
| alignment | 32 | CMA-ES | 0.497 | 0.453 | 4.54 | 9.76e-06 | +0.64 |
| regret | 4 | Infogain | 0.093 | 0.102 | -0.59 | 5.56e-01 | -0.08 |
| regret | 4 | Informative+Revealing | 0.093 | 0.075 | 1.41 | 1.59e-01 | +0.20 |
| regret | 4 | CMA-ES | 0.093 | 0.102 | -0.54 | 5.93e-01 | -0.08 |
| regret | 8 | Infogain | 0.482 | 0.673 | -3.29 | 1.18e-03 | -0.47 |
| regret | 8 | Informative+Revealing | 0.482 | 0.458 | 0.44 | 6.62e-01 | +0.06 |
| regret | 8 | CMA-ES | 0.482 | 0.543 | -1.00 | 3.17e-01 | -0.14 |
| regret | 16 | Infogain | 1.996 | 2.908 | -5.70 | 4.39e-08 | -0.81 |
| regret | 16 | Informative+Revealing | 1.996 | 2.313 | -2.29 | 2.30e-02 | -0.32 |
| regret | 16 | CMA-ES | 1.996 | 2.506 | -3.31 | 1.12e-03 | -0.47 |
| regret | 32 | Infogain | 5.628 | 7.178 | -4.90 | 2.22e-06 | -0.72 |
| regret | 32 | Informative+Revealing | 5.628 | 6.182 | -2.33 | 2.05e-02 | -0.33 |
| regret | 32 | CMA-ES | 5.628 | 5.902 | -1.19 | 2.37e-01 | -0.17 |
| per_query_alignment | 4 | Infogain | 0.801 | -0.006 | 122.47 | 1.36e-188 | +17.32 |
| per_query_alignment | 4 | Informative+Revealing | 0.801 | 0.479 | 47.56 | 2.71e-110 | +6.73 |
| per_query_alignment | 4 | CMA-ES | 0.801 | 0.750 | 4.95 | 1.56e-06 | +0.70 |
| per_query_alignment | 8 | Infogain | 0.746 | -0.000 | 152.39 | 3.44e-207 | +21.55 |
| per_query_alignment | 8 | Informative+Revealing | 0.746 | 0.321 | 79.50 | 3.41e-152 | +11.24 |
| per_query_alignment | 8 | CMA-ES | 0.746 | 0.657 | 9.64 | 2.76e-18 | +1.36 |
| per_query_alignment | 16 | Infogain | 0.631 | 0.003 | 130.21 | 8.48e-194 | +18.41 |
| per_query_alignment | 16 | Informative+Revealing | 0.631 | 0.173 | 85.60 | 2.23e-158 | +12.11 |
| per_query_alignment | 16 | CMA-ES | 0.631 | 0.513 | 14.18 | 5.83e-32 | +2.01 |
| per_query_alignment | 32 | Infogain | 0.457 | 0.000 | 76.88 | 1.52e-134 | +12.68 |
| per_query_alignment | 32 | Informative+Revealing | 0.457 | 0.076 | 69.75 | 2.50e-141 | +9.86 |
| per_query_alignment | 32 | CMA-ES | 0.457 | 0.339 | 16.23 | 3.21e-38 | +2.29 |

### Robot domains (Table 3 / Figure 4)

| domain | metric | baseline | CMA-ES-IG | baseline mean | t | p | Cohen d |
|---|---|---|---|---|---|---|---|
| Lunar Lander | alignment | Infogain | 0.939 | 0.959 | -7.19 | 1.27e-11 | -1.02 |
| Lunar Lander | alignment | Informative+Revealing | 0.939 | 0.950 | -3.55 | 4.82e-04 | -0.50 |
| Lunar Lander | alignment | CMA-ES | 0.939 | 0.942 | -0.93 | 3.56e-01 | -0.13 |
| Lunar Lander | regret | Infogain | 0.015 | 0.004 | 3.93 | 1.19e-04 | +0.56 |
| Lunar Lander | regret | Informative+Revealing | 0.015 | 0.012 | 0.92 | 3.58e-01 | +0.13 |
| Lunar Lander | regret | CMA-ES | 0.015 | 0.019 | -1.12 | 2.63e-01 | -0.16 |
| Lunar Lander | quality_avg | Infogain | 0.784 | 0.007 | 96.60 | 1.59e-168 | +13.66 |
| Lunar Lander | quality_avg | Informative+Revealing | 0.784 | 0.442 | 43.89 | 5.56e-104 | +6.21 |
| Lunar Lander | quality_avg | CMA-ES | 0.784 | 0.761 | 2.28 | 2.38e-02 | +0.32 |
| Driving | alignment | Infogain | 0.944 | 0.959 | -9.00 | 1.84e-16 | -1.27 |
| Driving | alignment | Informative+Revealing | 0.944 | 0.949 | -2.21 | 2.83e-02 | -0.31 |
| Driving | alignment | CMA-ES | 0.944 | 0.940 | 1.53 | 1.28e-01 | +0.22 |
| Driving | regret | Infogain | 0.020 | 0.007 | 3.80 | 1.92e-04 | +0.54 |
| Driving | regret | Informative+Revealing | 0.020 | 0.013 | 1.94 | 5.41e-02 | +0.27 |
| Driving | regret | CMA-ES | 0.020 | 0.026 | -1.08 | 2.79e-01 | -0.15 |
| Driving | quality_avg | Infogain | 0.809 | 0.017 | 125.39 | 1.34e-190 | +17.73 |
| Driving | quality_avg | Informative+Revealing | 0.809 | 0.447 | 61.40 | 7.50e-131 | +8.68 |
| Driving | quality_avg | CMA-ES | 0.809 | 0.763 | 4.98 | 1.38e-06 | +0.70 |
| Robot Face Design | alignment | Infogain | 0.946 | 0.960 | -9.61 | 3.34e-18 | -1.36 |
| Robot Face Design | alignment | Informative+Revealing | 0.946 | 0.949 | -1.41 | 1.60e-01 | -0.20 |
| Robot Face Design | alignment | CMA-ES | 0.946 | 0.943 | 1.40 | 1.62e-01 | +0.20 |
| Robot Face Design | regret | Infogain | 0.015 | 0.005 | 1.94 | 5.32e-02 | +0.28 |
| Robot Face Design | regret | Informative+Revealing | 0.015 | 0.014 | 0.09 | 9.32e-01 | +0.01 |
| Robot Face Design | regret | CMA-ES | 0.015 | 0.016 | -0.24 | 8.07e-01 | -0.03 |
| Robot Face Design | quality_avg | Infogain | 0.794 | 0.010 | 109.69 | 2.97e-179 | +15.51 |
| Robot Face Design | quality_avg | Informative+Revealing | 0.794 | 0.447 | 51.31 | 2.55e-116 | +7.26 |
| Robot Face Design | quality_avg | CMA-ES | 0.794 | 0.757 | 3.83 | 1.74e-04 | +0.54 |
| Robot Voice Design | alignment | Infogain | 0.944 | 0.955 | -2.16 | 3.21e-02 | -0.31 |
| Robot Voice Design | alignment | Informative+Revealing | 0.944 | 0.944 | -0.11 | 9.09e-01 | -0.02 |
| Robot Voice Design | alignment | CMA-ES | 0.944 | 0.943 | 0.42 | 6.74e-01 | +0.06 |
| Robot Voice Design | regret | Infogain | 0.014 | 0.006 | 3.03 | 2.75e-03 | +0.43 |
| Robot Voice Design | regret | Informative+Revealing | 0.014 | 0.014 | 0.10 | 9.22e-01 | +0.01 |
| Robot Voice Design | regret | CMA-ES | 0.014 | 0.016 | -0.43 | 6.70e-01 | -0.06 |
| Robot Voice Design | quality_avg | Infogain | 0.796 | 0.008 | 98.80 | 2.02e-170 | +13.97 |
| Robot Voice Design | quality_avg | Informative+Revealing | 0.796 | 0.443 | 45.95 | 1.43e-107 | +6.50 |
| Robot Voice Design | quality_avg | CMA-ES | 0.796 | 0.756 | 3.84 | 1.63e-04 | +0.54 |

## Runtime (Table 2), measured on this machine

| algorithm | d=4 | d=8 | d=16 | d=32 |
|---|---|---|---|---|
| CMA-ES-IG | 0.8 ms | 0.8 ms | 0.8 ms | 11.7 ms |
| CMA-ES | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| Infogain | 70.5 ms | 381.6 ms | 2770.9 ms | 7147.1 ms |
| Informative+Revealing | 35.2 ms | 38.7 ms | 44.8 ms | 54.4 ms |

## CMA-ES-IG information-gain approximation ablation

All variants share identical CMA-ES machinery; only the rule for choosing
which sampled candidates to show differs. Exhaustive search over the deployed
100-sample pool is intractable (3,921,225 subsets per query), so `exact` uses a
20-sample pool and `k-means, pool 20` is included as the control that separates
the selection rule from the pool size.

| variant | metric | d=4 | d=8 | d=16 | d=32 |
|---|---|---|---|---|---|
| k-means, pool 100 (deployed) | quality | 0.807 | 0.739 | 0.625 | 0.461 |
| k-means, pool 100 (deployed) | regret | 0.069 | 0.446 | 2.030 | 5.428 |
| k-means, pool 100 (deployed) | alignment | 0.960 | 0.902 | 0.715 | 0.511 |
| k-means, pool 20 | quality | 0.779 | 0.702 | 0.567 | 0.387 |
| k-means, pool 20 | regret | 0.076 | 0.457 | 2.137 | 6.270 |
| k-means, pool 20 | alignment | 0.960 | 0.890 | 0.709 | 0.464 |
| k-medoids, pool 20 | quality | 0.777 | 0.713 | 0.581 | 0.384 |
| k-medoids, pool 20 | regret | 0.072 | 0.495 | 2.585 | 6.259 |
| k-medoids, pool 20 | alignment | 0.959 | 0.891 | 0.688 | 0.440 |
| exact (exhaustive), pool 20 | quality | 0.269 | 0.307 | 0.401 | 0.322 |
| exact (exhaustive), pool 20 | regret | 0.048 | 0.414 | 3.083 | 8.606 |
| exact (exhaustive), pool 20 | alignment | 0.969 | 0.902 | 0.621 | 0.293 |
| ellipsoid 90%, fin. diff. | quality | 0.273 | 0.304 | 0.375 | 0.275 |
| ellipsoid 90%, fin. diff. | regret | 0.042 | 0.544 | 2.938 | 7.235 |
| ellipsoid 90%, fin. diff. | alignment | 0.967 | 0.887 | 0.617 | 0.363 |
| ellipsoid 90%, analytic grad. | quality | 0.269 | 0.334 | 0.364 | 0.274 |
| ellipsoid 90%, analytic grad. | regret | 0.047 | 0.531 | 3.008 | 7.076 |
| ellipsoid 90%, analytic grad. | alignment | 0.968 | 0.884 | 0.628 | 0.365 |
| ellipsoid 90%, grad. + subsample | quality | 0.411 | 0.403 | 0.412 | 0.291 |
| ellipsoid 90%, grad. + subsample | regret | 0.051 | 0.503 | 2.611 | 6.211 |
| ellipsoid 90%, grad. + subsample | alignment | 0.967 | 0.890 | 0.657 | 0.414 |

### Paired comparison against the deployed rule (k-means, pool 100)

| variant | metric | d | delta | t | p |
|---|---|---|---|---|---|
| k-means, pool 20 | quality | 4 | -0.028 | -3.17 | 1.79e-03 |
| k-medoids, pool 20 | quality | 4 | -0.030 | -3.57 | 4.52e-04 |
| exact (exhaustive), pool 20 | quality | 4 | -0.538 | -24.18 | 5.31e-61 |
| ellipsoid 90%, fin. diff. | quality | 4 | -0.534 | -38.21 | 2.54e-93 |
| ellipsoid 90%, analytic grad. | quality | 4 | -0.538 | -36.66 | 3.49e-90 |
| ellipsoid 90%, grad. + subsample | quality | 4 | -0.396 | -26.80 | 8.73e-68 |
| k-means, pool 20 | quality | 8 | -0.037 | -4.71 | 4.60e-06 |
| k-medoids, pool 20 | quality | 8 | -0.026 | -3.18 | 1.72e-03 |
| exact (exhaustive), pool 20 | quality | 8 | -0.432 | -31.91 | 5.57e-80 |
| ellipsoid 90%, fin. diff. | quality | 8 | -0.436 | -47.46 | 4.10e-110 |
| ellipsoid 90%, analytic grad. | quality | 8 | -0.405 | -40.94 | 1.39e-98 |
| ellipsoid 90%, grad. + subsample | quality | 8 | -0.336 | -30.43 | 1.38e-76 |
| k-means, pool 20 | quality | 16 | -0.059 | -7.35 | 5.21e-12 |
| k-medoids, pool 20 | quality | 16 | -0.045 | -6.00 | 9.17e-09 |
| exact (exhaustive), pool 20 | quality | 16 | -0.225 | -23.89 | 3.04e-60 |
| ellipsoid 90%, fin. diff. | quality | 16 | -0.251 | -37.97 | 7.74e-93 |
| ellipsoid 90%, analytic grad. | quality | 16 | -0.261 | -38.79 | 1.87e-94 |
| ellipsoid 90%, grad. + subsample | quality | 16 | -0.213 | -26.66 | 1.93e-67 |
| k-means, pool 20 | quality | 32 | -0.074 | -11.51 | 8.13e-24 |
| k-medoids, pool 20 | quality | 32 | -0.077 | -12.07 | 1.66e-25 |
| exact (exhaustive), pool 20 | quality | 32 | -0.139 | -22.80 | 2.82e-57 |
| ellipsoid 90%, fin. diff. | quality | 32 | -0.187 | -35.54 | 7.02e-88 |
| ellipsoid 90%, analytic grad. | quality | 32 | -0.187 | -36.92 | 1.01e-90 |
| ellipsoid 90%, grad. + subsample | quality | 32 | -0.170 | -28.20 | 2.83e-71 |
| k-means, pool 20 | regret | 4 | +0.007 | 0.53 | 6.00e-01 |
| k-medoids, pool 20 | regret | 4 | +0.003 | 0.26 | 7.95e-01 |
| exact (exhaustive), pool 20 | regret | 4 | -0.022 | -2.24 | 2.65e-02 |
| ellipsoid 90%, fin. diff. | regret | 4 | -0.028 | -3.04 | 2.71e-03 |
| ellipsoid 90%, analytic grad. | regret | 4 | -0.022 | -2.35 | 1.97e-02 |
| ellipsoid 90%, grad. + subsample | regret | 4 | -0.018 | -1.89 | 5.99e-02 |
| k-means, pool 20 | regret | 8 | +0.011 | 0.20 | 8.38e-01 |
| k-medoids, pool 20 | regret | 8 | +0.049 | 0.90 | 3.71e-01 |
| exact (exhaustive), pool 20 | regret | 8 | -0.032 | -0.67 | 5.06e-01 |
| ellipsoid 90%, fin. diff. | regret | 8 | +0.098 | 1.49 | 1.38e-01 |
| ellipsoid 90%, analytic grad. | regret | 8 | +0.085 | 1.36 | 1.74e-01 |
| ellipsoid 90%, grad. + subsample | regret | 8 | +0.057 | 1.06 | 2.92e-01 |
| k-means, pool 20 | regret | 16 | +0.107 | 0.70 | 4.84e-01 |
| k-medoids, pool 20 | regret | 16 | +0.555 | 3.65 | 3.39e-04 |
| exact (exhaustive), pool 20 | regret | 16 | +1.053 | 5.17 | 5.74e-07 |
| ellipsoid 90%, fin. diff. | regret | 16 | +0.908 | 4.79 | 3.21e-06 |
| ellipsoid 90%, analytic grad. | regret | 16 | +0.978 | 4.65 | 6.10e-06 |
| ellipsoid 90%, grad. + subsample | regret | 16 | +0.581 | 3.52 | 5.45e-04 |
| k-means, pool 20 | regret | 32 | +0.842 | 3.14 | 1.97e-03 |
| k-medoids, pool 20 | regret | 32 | +0.832 | 3.25 | 1.37e-03 |
| exact (exhaustive), pool 20 | regret | 32 | +3.179 | 9.42 | 1.18e-17 |
| ellipsoid 90%, fin. diff. | regret | 32 | +1.807 | 5.58 | 7.68e-08 |
| ellipsoid 90%, analytic grad. | regret | 32 | +1.649 | 4.97 | 1.44e-06 |
| ellipsoid 90%, grad. + subsample | regret | 32 | +0.784 | 2.90 | 4.13e-03 |

### Query-generation cost of each approximation (ms, idle machine)

| variant | d=4 | d=8 | d=16 | d=32 |
|---|---|---|---|---|
| k-means, pool 100 (deployed) | 0.7 | 0.7 | 1.7 | 2.0 |
| k-means, pool 20 | 0.5 | 0.4 | 0.9 | 12.0 |
| k-medoids, pool 20 | 0.6 | 0.5 | 0.5 | 0.6 |
| exact (exhaustive), pool 20 | 720.2 | 723.7 | 726.1 | 720.9 |
| ellipsoid 90%, fin. diff. | 313.0 | 889.7 | 2076.4 | 2690.1 |
