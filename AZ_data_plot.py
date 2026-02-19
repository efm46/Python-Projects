import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, 'spring_2026_flu_talk', 'AZ_data', 'Arizona_flu_full.exp')

# Load the data
data = np.genfromtxt(data_file, skip_header=1)
time_weeks = data[:, 0]
hospitalizations = data[:, 1]

# Convert weeks to dates starting from June 25, 2025 (Wednesday)
start_date = datetime(2025, 6, 25)
dates = [start_date + timedelta(weeks=int(week)) for week in time_weeks]

# Set up matplotlib style for presentation

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

# Plot the data with a bold line and prominent markers
ax.plot(dates, hospitalizations, 
        linewidth=4, 
        marker='o', 
        markersize=10,
        markerfacecolor='#2E86AB',
        markeredgecolor='#1A5F7A',
        markeredgewidth=2.5,
        color='#2E86AB',
        label='Weekly Hospitalizations',
        zorder=3)

# Customize the plot
ax.set_xlabel('Date', fontweight='bold', fontsize=18)
ax.set_ylabel('Hospitalizations', fontweight='bold', fontsize=18)
ax.set_title('Arizona Flu Hospitalization Data', fontweight='bold', fontsize=22, pad=20)

# Format x-axis dates - select every 4 weeks for readability
num_ticks = 8
tick_indices = np.linspace(0, len(dates)-1, num_ticks, dtype=int)
tick_dates = [dates[i] for i in tick_indices]
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
ax.set_xlim(dates[0] - timedelta(days=3), dates[-1] + timedelta(days=3))
ax.set_ylim(0, max(hospitalizations) * 1.1)

# Add a subtle background color
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')

# Tight layout for better spacing
plt.tight_layout()

# Save high-resolution figure
output_file = os.path.join(script_dir, 'AZ_hospitalization_plot.png')
plt.savefig(output_file, dpi=500, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Plot saved to: {output_file}")

# Display the plot (commented out for script execution, uncomment if running interactively)
# plt.show()
