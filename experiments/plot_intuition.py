import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from cmaes import CMA
from sklearn.cluster import KMeans

def objective(x: np.ndarray) -> float:
    """A simple 2D objective function with a minimum at (2, 3)."""
    return (x[0] - 2)**2 + (x[1] - 3)**2

def run_cma_and_collect_samples(iterations: int):
    """Run CMA-ES and return all samples with their iteration number plus mean/covariance data."""
    optimizer = CMA(mean=np.array([-4., -2.]), sigma=0.9, population_size=10)
    all_samples = []
    mean_cov_data = []

    for generation in range(iterations):
        solutions = []
        sample_count = 0

        candidates = []
        for _ in range(100):
            x = optimizer.ask()
            candidates.append(x)

        kmeans = KMeans(n_clusters=optimizer.population_size, n_init="auto").fit(candidates)
        query = kmeans.cluster_centers_

        for x in query:
            x = optimizer.ask()
            value = objective(x)
            solutions.append((x, value))
            # Only add first 3 samples to the plot
            if sample_count < 3 and generation % 3 == 0:
                all_samples.append((x, generation))
                sample_count += 1
        optimizer.tell(solutions)
        
        # Store mean and covariance for every third iteration only
        if generation % 3 == 0:
            mean = optimizer.mean.copy()
            # Get covariance matrix (sigma^2 * C)
            cov = (optimizer._sigma ** 2) * optimizer._C
            mean_cov_data.append((mean, cov, generation))
    
    return all_samples, mean_cov_data

def covariance_to_ellipse(mean, cov, n_std=2.0):
    """Convert covariance matrix to ellipse parameters."""
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
    
    # Calculate ellipse parameters
    width, height = 2 * n_std * np.sqrt(eigenvals)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    return width, height, angle

def plot_cma_samples(iterations: int, output_file: str):
    """Generate and save the visualization of CMA-ES samples over time."""
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the objective function contours
    x_range = np.linspace(-6, 8, 400)
    y_range = np.linspace(-4, 6, 400)
    X, Y = np.meshgrid(x_range, y_range)
    Z = objective(np.array([X, Y]))
    
    ax.contourf(X, Y, Z, levels=20, cmap='gray_r', alpha=0.5)
    ax.contour(X, Y, Z, levels=20, colors='white', alpha=0.3)

    # Run CMA-ES and get samples
    samples, mean_cov_data = run_cma_and_collect_samples(iterations)
    points = np.array([s[0] for s in samples])
    generations = np.array([s[1] for s in samples])

    # Use a colormap to show time progression
    gradient_colors = ["#2E4A40","#37584c","#668b7f", "#ff91af","#FFDDE7"]
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list("custom", gradient_colors)
    scatter = ax.scatter(points[:, 0], points[:, 1], c=generations, cmap=custom_cmap, label='CMA-ES Samples')
    
    # Plot mean and covariance ellipses
    # Create color array based on number of iterations we're plotting
    n_ellipses = len(mean_cov_data)
    if n_ellipses <= len(gradient_colors):
        colors = gradient_colors[:n_ellipses]
    else:
        # If we have more ellipses than colors, interpolate
        colors = [custom_cmap(i / (n_ellipses - 1)) for i in range(n_ellipses)]
    
    for i, (mean, cov, gen) in enumerate(mean_cov_data):
        width, height, angle = covariance_to_ellipse(mean, cov)
        ellipse = Ellipse(mean, width, height, angle=angle, 
                         fill=False, edgecolor=colors[i], 
                         linestyle='--', linewidth=2, alpha=0.8)
        ax.add_patch(ellipse)
        # Mark the mean
        ax.plot(mean[0], mean[1], 'x', color=colors[i], markersize=8, markeredgewidth=2)

    # Draw line connecting successive means
    if len(mean_cov_data) > 1:
        mean_positions = np.array([data[0] for data in mean_cov_data])
        ax.plot(mean_positions[:, 0], mean_positions[:, 1], 
               color='#ffcc00', linewidth=2, alpha=0.8, zorder=5, linestyle=':')

    ax.set_xlim(x_range.min(), x_range.max())
    ax.set_ylim(y_range.min(), y_range.max())
    ax.set_xlabel('dimension 1')
    ax.set_ylabel('dimension 2')
    ax.set_aspect('equal', 'box')
    
    # Create a colorbar for iterations with same height as plot
    cbar = fig.colorbar(scatter, ax=ax, shrink=1.0)
    cbar.set_label('Iteration')

    plt.savefig(output_file)
    print(f"Saved visualization to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize CMA-ES samples.')
    parser.add_argument('--iterations', type=int, default=15, help='Number of CMA-ES iterations to run.')
    parser.add_argument('--output', type=str, default='cma_samples.png', help='Output file name.')
    args = parser.parse_args()
    
    plot_cma_samples(args.iterations, args.output)

