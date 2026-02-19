import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import glob
import sys

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
exp_file = os.path.join(script_dir, 'AZ_data', '29.exp')
results_dir = os.path.join(script_dir, 'results', '29_line_2', 'Results')

try:
    # Load the experimental data - this will be plotted as points
    exp_data = np.genfromtxt(exp_file, skip_header=1)
    exp_time_weeks = exp_data[:, 0]
    exp_H_weekly = exp_data[:, 1]  # Experimental H_weekly values

    # Convert experimental weeks to dates starting from June 25, 2025 (Wednesday) - same as AZ_data_plot.py
    start_date = datetime(2025, 6, 25)
    exp_dates = [start_date + timedelta(weeks=int(week)) for week in exp_time_weeks]

    # Find the most recent generation folder (e.g., 32_gen50ind4)
    gen_folders = [d for d in os.listdir(results_dir) 
                   if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('29_gen')]
    if not gen_folders:
        raise ValueError(f"No generation folders found in {results_dir}")

    # Sort by modification time to get the most recent
    gen_folders.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    gen_folder = gen_folders[0]
    gen_folder_path = os.path.join(results_dir, gen_folder)

    print(f"Using generation folder: {gen_folder}")

    # Find the most recent timestamp folder inside the generation folder
    timestamp_folders = [d for d in os.listdir(gen_folder_path) 
                         if os.path.isdir(os.path.join(gen_folder_path, d))]
    if not timestamp_folders:
        raise ValueError(f"No timestamp folders found in {gen_folder_path}")

    # Sort by modification time to get the most recent
    timestamp_folders.sort(key=lambda x: os.path.getmtime(os.path.join(gen_folder_path, x)), reverse=True)
    timestamp_folder = timestamp_folders[0]
    timestamp_folder_path = os.path.join(gen_folder_path, timestamp_folder)

    print(f"Using timestamp folder: {timestamp_folder}")

    # Find the .gdat file (should be named like {gen_folder}_32.gdat)
    gdat_pattern = os.path.join(timestamp_folder_path, f"{gen_folder}_32.gdat")
    gdat_files = glob.glob(gdat_pattern)

    if not gdat_files:
        # Try alternative pattern - just look for any .gdat file
        gdat_files = glob.glob(os.path.join(timestamp_folder_path, "*.gdat"))
        if not gdat_files:
            raise ValueError(f"No .gdat file found in {timestamp_folder_path}")
        gdat_file = gdat_files[0]
    else:
        gdat_file = gdat_files[0]

    print(f"Using gdat file: {os.path.basename(gdat_file)}")

    # Load the gdat file - it has multiple columns, H_weekly is the last column
    gdat_data = np.genfromtxt(gdat_file, skip_header=1)
    model_time = gdat_data[:, 0]  # Time column from gdat
    model_H_weekly = gdat_data[:, -1]  # Last column is H_weekly (model fit)

    # Convert model time to dates (model has 4 more points than experimental data)
    model_dates = [start_date + timedelta(weeks=float(week)) for week in model_time]

    print(f"Experimental data: {len(exp_H_weekly)} points")
    print(f"Model fit data: {len(model_H_weekly)} points")

    # Set up matplotlib style for presentation - same as AZ_data_plot.py
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.linestyle'] = '--'

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the model fit as a line only (no markers)
    ax.plot(model_dates, model_H_weekly, 
            linewidth=4, 
            linestyle='-',
            color='#2E86AB',
            label='Model Fit (MLE)',
            zorder=2)

    # Plot the experimental data as points only (no line)
    ax.plot(exp_dates, exp_H_weekly, 
            linestyle='None',  # No line
            marker='o', 
            markersize=10,
            markerfacecolor='#E63946',
            markeredgecolor='#A01D26',
            markeredgewidth=2.5,
            label='Experimental Data',
            zorder=3)

    # Customize the plot
    ax.set_xlabel('Date', fontweight='bold', fontsize=18)
    ax.set_ylabel('Hospitalizations', fontweight='bold', fontsize=18)
    ax.set_title('Arizona Flu Hospitalization Data - MLE Fit', fontweight='bold', fontsize=22, pad=20)

    # Format x-axis dates - select every 4 weeks for readability (same as AZ_data_plot.py)
    # Use experimental dates for xticks to match AZ_data_plot.py
    num_ticks = 8
    tick_indices = np.linspace(0, len(exp_dates)-1, num_ticks, dtype=int)
    tick_dates = [exp_dates[i] for i in tick_indices]
    ax.set_xticks(tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))
    # Rotate date labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Improve tick marks
    ax.tick_params(which='major', length=8, width=1.5)
    ax.tick_params(which='minor', length=4, width=1)

    # Set nice axis limits with some padding
    # Use the full range of both datasets
    all_dates = exp_dates + model_dates
    max_y = max(max(exp_H_weekly), max(model_H_weekly))
    ax.set_xlim(exp_dates[0] - timedelta(days=3), max(all_dates) + timedelta(days=3))
    ax.set_ylim(0, max_y * 1.1)

    # Add a subtle background color
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')

    # Add legend
    ax.legend(loc='best')

    # Tight layout for better spacing
    plt.tight_layout()

    # Save high-resolution figure
    output_file = os.path.join(script_dir, 'az_MLE_plot.png')
    plt.savefig(output_file, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close(fig)  # Close the figure to free memory
    print(f"Plot saved to: {output_file}")
    sys.stdout.flush()  # Ensure output is flushed

    # Display the plot (commented out for script execution, uncomment if running interactively)
    # plt.show()

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
