
"""
Visualization for camera trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_trajectory_2d(positions_est, positions_gt=None, title="Camera Trajectory (Top View)",
                      save_path=None, show=True):
    """
    Plot 2D trajectory (top view: x-z plane).
    
    Args:
        positions_est: Nx3 array of estimated camera positions
        positions_gt: Nx3 array of ground truth positions (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Plot estimated trajectory
    plt.plot(positions_est[:, 0], positions_est[:, 2], 'b-', linewidth=2, label='Estimated')
    plt.plot(positions_est[0, 0], positions_est[0, 2], 'go', markersize=10, label='Start')
    plt.plot(positions_est[-1, 0], positions_est[-1, 2], 'ro', markersize=10, label='End')
    
    # Plot ground truth if provided
    if positions_gt is not None:
        plt.plot(positions_gt[:, 0], positions_gt[:, 2], 'r--', linewidth=2, label='Ground Truth', alpha=0.7)
    
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved 2D trajectory plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_trajectory_3d(positions_est, positions_gt=None, title="Camera Trajectory (3D)",
                      save_path=None, show=True):
    """
    Plot 3D trajectory.
    
    Args:
        positions_est: Nx3 array of estimated camera positions
        positions_gt: Nx3 array of ground truth positions (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot estimated trajectory
    ax.plot(positions_est[:, 0], positions_est[:, 1], positions_est[:, 2], 
            'b-', linewidth=2, label='Estimated')
    ax.scatter(positions_est[0, 0], positions_est[0, 1], positions_est[0, 2],
              c='g', marker='o', s=100, label='Start')
    ax.scatter(positions_est[-1, 0], positions_est[-1, 1], positions_est[-1, 2],
              c='r', marker='o', s=100, label='End')
    
    # Plot ground truth if provided
    if positions_gt is not None:
        ax.plot(positions_gt[:, 0], positions_gt[:, 1], positions_gt[:, 2],
               'r--', linewidth=2, label='Ground Truth', alpha=0.7)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved 3D trajectory plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_trajectory_error(positions_est, positions_gt, title="Trajectory Error Over Time",
                         save_path=None, show=True):
    """
    Plot trajectory error over time.
    
    Args:
        positions_est: Nx3 array of estimated camera positions
        positions_gt: Nx3 array of ground truth positions
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    # Compute errors
    errors = np.linalg.norm(positions_est - positions_gt, axis=1)
    frames = np.arange(len(errors))
    
    plt.figure(figsize=(12, 6))
    plt.plot(frames, errors, 'b-', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Position Error (meters)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    plt.axhline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.3f}m')
    plt.axhline(max_error, color='orange', linestyle='--', label=f'Max: {max_error:.3f}m')
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved trajectory error plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_trajectory_comparison(positions_est, positions_gt, title="Trajectory Comparison",
                               save_path=None, show=True):
    """
    Plot comprehensive trajectory comparison with multiple views.
    
    Args:
        positions_est: Nx3 array of estimated camera positions
        positions_gt: Nx3 array of ground truth positions
        title: Overall title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=16)
    
    # Top view (x-z)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(positions_est[:, 0], positions_est[:, 2], 'b-', linewidth=2, label='Estimated')
    ax1.plot(positions_gt[:, 0], positions_gt[:, 2], 'r--', linewidth=2, label='Ground Truth')
    ax1.plot(positions_est[0, 0], positions_est[0, 2], 'go', markersize=8)
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Z (meters)')
    ax1.set_title('Top View (X-Z)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Side view (x-y)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(positions_est[:, 0], positions_est[:, 1], 'b-', linewidth=2, label='Estimated')
    ax2.plot(positions_gt[:, 0], positions_gt[:, 1], 'r--', linewidth=2, label='Ground Truth')
    ax2.plot(positions_est[0, 0], positions_est[0, 1], 'go', markersize=8)
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_title('Side View (X-Y)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Error over time
    ax3 = plt.subplot(2, 2, 3)
    errors = np.linalg.norm(positions_est - positions_gt, axis=1)
    frames = np.arange(len(errors))
    ax3.plot(frames, errors, 'b-', linewidth=2)
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Position Error (meters)')
    ax3.set_title('Trajectory Error')
    ax3.grid(True, alpha=0.3)
    
    # 3D view
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot(positions_est[:, 0], positions_est[:, 1], positions_est[:, 2],
            'b-', linewidth=2, label='Estimated')
    ax4.plot(positions_gt[:, 0], positions_gt[:, 1], positions_gt[:, 2],
            'r--', linewidth=2, label='Ground Truth')
    ax4.scatter(positions_est[0, 0], positions_est[0, 1], positions_est[0, 2],
               c='g', marker='o', s=50)
    ax4.set_xlabel('X (meters)')
    ax4.set_ylabel('Y (meters)')
    ax4.set_zlabel('Z (meters)')
    ax4.set_title('3D View')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved trajectory comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()