import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# Determine project root relative to this script
project_root = os.path.dirname(os.path.abspath(__file__))

# Set the base directory where all result folders are stored
base_dir = os.path.join(project_root, 'results')
burn_in = 0.05  # Drop the first 25% of values for burn-in
output_dir = os.path.join(project_root, 'uq_plots')  # Directory for output PNG files

# Directory for Arizona experimental data and master full data file
exp_base_dir = os.path.join(project_root, 'AZ_data')
master_exp_file = os.path.join(exp_base_dir, 'Arizona_flu_full.exp')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

names = ['19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32']

# Load the full Arizona experimental time series once
try:
    full_obs = np.genfromtxt(master_exp_file)
    print(f"Loaded full Arizona experimental data, shape: {full_obs.shape}")
except Exception as e:
    print(f"Error loading full Arizona experimental data from {master_exp_file}: {e}")
    full_obs = None


for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)

    # Only process if the subdirectory is a directory and matches a state name
    if os.path.isdir(subdir_path) and subdir in names:
        print(f"Processing directory: {subdir}")

        state_name = subdir  # Extract the state name from the folder

        # Dynamically create the paths for traj_file and exp_file
        traj_file = os.path.join(base_dir, state_name, f'Results/A_MCMC/Runs/traj_noise_{state_name}_fluH_weekly_chain_0.txt')
        exp_file = os.path.join(exp_base_dir, f'{state_name}.exp')


        # Print the paths to verify if they are correct
        print(f"Checking files for {state_name}...")
        print(f"Trajectory file path: {traj_file}")
        print(f"Experiment file path: {exp_file}")
        #print(f"Params file path: {params_file}")
        #print(f"Scores file path: {scores_file}")
        #print(f"Combined Params file path: {combined_params_file}")

        # Check if the necessary files exist
        if os.path.exists(traj_file) and os.path.exists(exp_file):
            print(f"Files found for {state_name}, loading data...")

            # Load trajectory data
            try:
                d = np.genfromtxt(traj_file)
                print(f"Loaded trajectory data for {state_name}, shape: {d.shape}")
            except Exception as e:
                print(f"Error loading trajectory data for {state_name}: {e}")
                continue

            # Load observed data (training window for this forecast)
            try:
                obs = np.genfromtxt(exp_file)
                print(f"Loaded observed data for {state_name}, shape: {obs.shape}")
            except Exception as e:
                print(f"Error loading observed data for {state_name}: {e}")
                continue

            # Check if trajectory data has content
            if d.size == 0:
                print(f"Warning: Trajectory data for {state_name} is empty. Skipping plot generation.")
                continue

            # Determine the last time point used for training (based on this exp file)
            try:
                training_last_time = int(np.max(obs[:, 0]))
            except Exception:
                training_last_time = int(state_name)

            # Compute quantiles for uncertainty visualization
            qtlMark = 1.00 * np.array([0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450,
                                       0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950,
                                       0.975, 0.990])
            qtlLog = np.zeros((len(qtlMark), d.shape[1]))
            for i in range(d.shape[1]):
                qtlLog[:, i] = np.quantile(d[:, i], qtlMark)

            # Define time span for the x-axis
            tSpan = np.linspace(0, d.shape[1] - 1, d.shape[1])

            # Create plot for uncertainty quantification
            plt.figure()
            colors = cm.plasma(np.linspace(0, 1, 12))
            for i in range(11):
                plt.fill_between(tSpan, qtlLog[i, :], qtlLog[22 - i, :], facecolor=colors[11 - i], zorder=i)
            # Training data used for this forecast
            plt.scatter(obs[:, 0], obs[:, 1], 10, marker='+', color='k', zorder=500, label='Training data')

            # Overlay the next 4 observed data points from the full time series
            if full_obs is not None:
                # Select the first four points after the training window
                future_points = full_obs[full_obs[:, 0] > training_last_time][:4]
                if future_points.size > 0:
                    plt.scatter(
                        future_points[:, 0],
                        future_points[:, 1],
                        30,
                        marker='o',
                        color='red',
                        edgecolor='k',
                        zorder=550,
                        label='Next 4 weeks'
                    )
                    plt.legend()

            plt.title(f"Uncertainty Quantification for {state_name}")

            # Save plot as PNG
            png_filename = os.path.join(output_dir, f"{state_name}_UQ.png")
            plt.savefig(png_filename)
            plt.close()
            print(f"Plot saved as {png_filename}")