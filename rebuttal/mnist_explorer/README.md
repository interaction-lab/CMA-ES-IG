# MNIST preference explorer

Search a PCA latent space of MNIST by ranking digits, using the query generators from the
paper (Infogain, CMA-ES, CMA-ES-IG with k-means, and CMA-ES-IG with exact information gain)
and either the plain or the resampling belief.

    uv run prepare.py      # once: downloads MNIST (~10 MB) and fits the latent space
    uv run app.py          # then open http://127.0.0.1:5000

Think of a target digit and click the images from most to least like it. Each tab is its own
session, so open two tabs to compare algorithms or beliefs side by side. The right panel shows
the real digits the robot currently ranks highest, the information the current query carries
about your preference (first choice, in bits), and the belief's effective sample size.
