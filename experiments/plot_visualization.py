import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

def gradient_of_function(x, y, flatness=0.1):
    """Calculate the gradient of the bent paper function."""
    # Function: Z = flatness * (0.5*x + 0.3*y + 0.03*x^2 - 0.015*y^2 + 0.02*x*y - 0.005*x^3 + 0.003*y^3)
    dZ_dx = flatness * (0.5 + 0.06*x + 0.02*y - 0.015*x**2) - 0.4*x
    dZ_dy = flatness * (0.3 - 0.03*y + 0.02*x + 0.009*y**2) - 0.2*y
    return dZ_dx, dZ_dy

def create_simple_manifold(flatness=1.0, gradient_colors=['#2E86AB', '#A23B72', '#F18F01']):
    """
    Create a bent paper-like manifold with gradient ascent visualization.
    
    Parameters:
    flatness (float): Controls the height of the paper bend (default: 1.0)
    gradient_colors (list): List of hex colors for the gradient (default: blue to pink to orange)
    """
    
    # Create parameter space
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    # Bent paper function: more curvy but still viewable from one angle
    Z = flatness * (0.5*X + 0.3*Y + 0.03*X**2 - 0.015*Y**2 + 0.02*X*Y - 0.005*X**3 + 0.003*Y**3) - .2*X**2 - .1*Y**2
    
    # Create custom colormap from the provided colors
    custom_cmap = LinearSegmentedColormap.from_list('custom', gradient_colors)
    
    # Create the plot with larger, clearer elements
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface with custom colors (ensure colormap is applied)
    surface = ax.plot_surface(X, Y, Z, cmap=custom_cmap, alpha=0.8,
                             linewidth=0, antialiased=True, vmin=Z.min(), vmax=Z.max())
    
    # Starting point x_0 (adjusted for better path)
    x0, y0 = -1.3, -1.2
    z0 = flatness * (0.5*x0 + 0.3*y0 + 0.03*x0**2 - 0.015*y0**2 + 0.02*x0*y0 - 0.005*x0**3 + 0.003*y0**3) - .2*x0**2 - .1*y0**2
    
    # Mark the starting point (elevated above surface)
    z0_elevated = z0 + 0.05
    ax.scatter([x0], [y0], [z0_elevated], color='#ffcc00', s=150, 
              label='Gradient ascent path', zorder=10)
    
    # Gradient ascent path with larger steps
    learning_rate = 0.8
    num_steps = 3
    path_x, path_y, path_z = [x0], [y0], [z0]
    
    current_x, current_y = x0, y0
    
    for step in range(num_steps):
        # Calculate gradient
        grad_x, grad_y = gradient_of_function(current_x, current_y, flatness)
        
        # Take gradient step (larger steps)
        current_x += learning_rate * grad_x
        current_y += learning_rate * grad_y
        current_z = flatness * (0.5*current_x + 0.3*current_y + 0.03*current_x**2 - 0.015*current_y**2 + 0.02*current_x*current_y - 0.005*current_x**3 + 0.003*current_y**3) - .2*current_x**2 - .1*current_y**2

        path_x.append(current_x)
        path_y.append(current_y)
        path_z.append(current_z)
        
        # Stop if we go out of bounds
        if abs(current_x) > 2 or abs(current_y) > 2:
            break
    
    # Plot the gradient ascent path (elevated above surface)
    path_z_elevated = [z + 0.05 for z in path_z]  # Lift path above surface
    ax.plot(path_x, path_y, path_z_elevated, 'o-', color='#ffcc00', linewidth=4, markersize=10, 
            markeredgecolor='#000000', markeredgewidth=1, zorder=10)
    
    # Add text labels for each point (elevated above surface)
    for i, (px, py, pz) in enumerate(zip(path_x, path_y, path_z)):
        ax.text(px, py, pz + 0.15, f'x{i}', fontsize=14, fontweight='bold', 
                ha='center', va='bottom', color='#000000', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#ffffff'),
                zorder=15)
    
    # Add arrows showing gradient direction between consecutive points (elevated)
    for i in range(len(path_x)-1):
        dx = path_x[i+1] - path_x[i]
        dy = path_y[i+1] - path_y[i]
        dz = path_z[i+1] - path_z[i]
        
        # Scale arrows to be more visible and elevate them
        scale = 0.7
        z_start = path_z[i] + 0.05  # Elevate arrow start
        ax.quiver(path_x[i], path_y[i], z_start, 
                 dx*scale, dy*scale, dz*scale, 
                 color='#ffcc00', arrow_length_ratio=0.25, linewidth=4, zorder=12)
    
    # Mark the final point with a star (also elevated)
    final_z_elevated = path_z[-1] + 0.05
    ax.scatter([path_x[-1]], [path_y[-1]], [final_z_elevated], 
              color='#ffcc00', s=200, marker='*', edgecolors='#000000', 
              linewidth=2, zorder=11)
    
    # Remove all labels, axes, and visual elements for clean poster image
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_title('')
    
    # Set a good viewing angle
    ax.view_init(elev=5, azim=-100)
    
    # Remove all axis elements for completely clean look
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # Remove axis tick marks and labels
    ax.xaxis._axinfo["tick"]["inward_factor"] = 0
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0
    
    plt.tight_layout()
    
    # Save as high-resolution PNG
    plt.savefig('gradient_ascent_manifold.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', transparent=False)
    
    plt.show()

if __name__ == "__main__":
    # Create bent paper with gradient ascent visualization
    gradient_colors=["#2E4A40","#37584c","#668b7f", "#ff91af","#FFDDE7"]
    create_simple_manifold(gradient_colors=gradient_colors)
    
    # Alternative: flatter paper
    # create_simple_manifold(flatness=0.5)
    
    # Alternative: with different colors
    # create_simple_manifold(gradient_colors=['#000080', '#FFD700', '#FFFFFF'])
    
    # Create parameter space for second plot
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    # Bent paper function
    flatness = 1.0
    Z = flatness * (0.5*X + 0.3*Y + 0.03*X**2 - 0.015*Y**2 + 0.02*X*Y - 0.005*X**3 + 0.003*Y**3) - .2*X**2 - .1*Y**2
    
    # Create custom colormap from the new gradient colors
    custom_cmap = LinearSegmentedColormap.from_list('custom', gradient_colors)
    
    # Create the plot with larger, clearer elements
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface with custom colors
    surface = ax.plot_surface(X, Y, Z, cmap=custom_cmap, alpha=0.9,
                             linewidth=0, antialiased=True)
    
    # Simplified labels and title
    ax.set_title('2D Manifold in 3D Space', fontsize=16, fontweight='bold')
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Function Value', fontsize=14)
    
    # Add colorbar with clear labeling
    cbar = fig.colorbar(surface, ax=ax, shrink=0.6, aspect=15)
    cbar.set_label('Higher Values → Lighter Colors', fontsize=12)
    
    # Set a good viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Remove background grid for cleaner look
    ax.grid(False)
    
    # Make the plot look cleaner
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.tight_layout()
    plt.show()

