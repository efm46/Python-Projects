import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pymmwr as pm
from datetime import datetime
from datetime import date
from datetime import timedelta
from scipy.integrate import solve_ivp
from scipy.stats import gamma, nbinom
from scipy.special import loggamma
from copy import copy
from scipy.optimize import minimize, Bounds
from sys import argv
from scipy.stats import nbinom
from matplotlib.pyplot import cm

# State names and abbreviations
state_names = ["District_of_Columbia", "Puerto_Rico","Florida","Alabama", "Alaska","Arkansas", "Arizona", "California", "Colorado", "Connecticut", "Delaware", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North_Carolina", "North_Dakota", "Nebraska", "New_Hampshire", "New_Jersey", "New_Mexico", "Nevada", "New_York", "Ohio", "Oklahoma", "Oregon", "Puerto_Rico", "Pennsylvania", "Rhode_Island", "South_Carolina", "South_Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West_Virginia", "Wyoming"]

state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District_of_Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New_Hampshire": "NH",
    "New_Jersey": "NJ",
    "New_Mexico": "NM",
    "New_York": "NY",
    "North_Carolina": "NC",
    "North_Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Puerto_Rico": "PR",
    "Pennsylvania": "PA",
    "Rhode_Island": "RI",
    "South_Carolina": "SC",
    "South_Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West_Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",}

# Set your local directories
#data_folder = '/Users/l-biosci-posnerlab/Desktop/2025_NAU_FLU/current_job/'
#exp_folder = '/Users/l-biosci-posnerlab/Desktop/2025_NAU_FLU/current_job/exp_files/'
data_folder = '/Users/elymiller/Desktop/Current_Research/Misc-Python-Projects/spring_2026_flu_talk/'
exp_folder = '/Users/elymiller/Desktop/Current_Research/Misc-Python-Projects/spring_2026_flu_talk/AZ_data/'

# Load the COVID data
covid = pd.read_csv(data_folder + '021126_Hdata.csv')

# Prepare the influenza data for processing
#HI = covid[['date','state','previous_day_admission_influenza_confirmed','total_patients_hospitalized_confirmed_influenza_coverage']].sort_values(['state','date'])
HI = covid[['Week Ending Date','Geographic aggregation','Total Influenza Admissions']].sort_values(['Geographic aggregation','Week Ending Date'])

# Define the target season and onset/end dates
targetSeason = 2025
onset = pm.epiweek_to_date(pm.Epiweek(targetSeason, 26))
end = pm.epiweek_to_date(pm.Epiweek(targetSeason+1, 47))

# Initialize arrays to hold data for each state
season2023Data = np.nan * np.zeros((len(state_names), (end-onset).days))
season2023dateArray = [[] for i in range(len(state_names))]

# Initialize array to accumulate US-wide hospitalizations
# We'll track the maximum length needed and sum values at each time point
max_length = 0
us_hospitalizations = None

# Loop through all states
for locationIndex in range(len(state_names)):
    targetState = state_names[locationIndex]
    
    # Filter the data for the specific state
    state_HI_data = HI[HI['Geographic aggregation'] == state_to_abbrev[targetState]]
    
    # Process the date array and filter based on the season
    dateArray = [datetime.strptime(state_HI_data['Week Ending Date'].values[i], '%Y-%m-%d') - timedelta(days=1) for i in range(len(state_HI_data))]
    binary = [onset <= buff.date() < end for buff in dateArray]  # previous day 
    
    tSpan = np.array([(dateArray[i] - dateArray[0]).days for i in range(len(state_HI_data))])
    
    # Fill the seasonal data and date array for each state
    season2023Data[locationIndex, :np.sum(binary)] = state_HI_data['Total Influenza Admissions'][binary]
    season2023dateArray[locationIndex] = np.array(dateArray)[binary]
    
    # Convert dateArray to timeArray for the current state
    def dateArray_to_timeArray(dateArray):
        timeArray = np.zeros(len(dateArray))
        timeArray[:] = np.array([(date - dateArray[0]).days for date in dateArray], dtype='int64')
        return timeArray.astype('int64')
    
    Y23 = season2023Data[locationIndex, :]
    X23 = dateArray_to_timeArray(season2023dateArray[locationIndex])
    Y23 = Y23[~np.isnan(Y23)]  # Remove NaNs
    
    # Prepare the data for saving
    my_array = np.transpose(np.array([list(range(len(Y23))), Y23]))
    df = pd.DataFrame(my_array, columns=['#time', 'H_weekly'])
    df['#time'] = df['#time'].astype('int')
    
    # Save the .exp file for the current state
    df.to_csv(exp_folder + targetState + '_flu.exp', sep='\t', index=False)
    
    # Accumulate hospitalizations for US-wide sum
    current_length = len(Y23)
    if current_length > max_length:
        # Expand the US array if needed
        if us_hospitalizations is None:
            us_hospitalizations = np.zeros(current_length)
        else:
            # Pad with zeros to match new length
            us_hospitalizations = np.pad(us_hospitalizations, (0, current_length - max_length), 'constant')
        max_length = current_length
    
    # Add current location's hospitalizations to US sum
    us_hospitalizations[:current_length] += Y23
    
    # Output confirmation
    print(f"File saved for {targetState}")

# Save the US-wide summed hospitalizations
if us_hospitalizations is not None:
    # Prepare the data for saving
    us_array = np.transpose(np.array([list(range(len(us_hospitalizations))), us_hospitalizations]))
    us_df = pd.DataFrame(us_array, columns=['#time', 'H_weekly'])
    us_df['#time'] = us_df['#time'].astype('int')
    
    # Save the .exp file for US cases
    us_df.to_csv(exp_folder + 'US_cases.exp', sep='\t', index=False)
    print(f"File saved for US_cases")
