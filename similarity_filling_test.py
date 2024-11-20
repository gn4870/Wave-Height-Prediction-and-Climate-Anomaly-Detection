import os
import pandas as pd
import numpy as np
import h5py
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics.pairwise import cosine_similarity

# Define file paths and variables
files = [
    'Hourly_Data/41008.csv', 'Hourly_Data/41009.csv', 'Hourly_Data/41010.csv',
    'Hourly_Data/42022.csv', 'Hourly_Data/42036.csv', 'Hourly_Data/fwyf1.csv',
    'Hourly_Data/smkf1.csv', 'Hourly_Data/venf1.csv'
]
target_station = '41009'  # Only filling missing values in 41009.csv for testing
variables = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP']

# Load all station data into a dictionary
data = {os.path.splitext(os.path.basename(path))[0]: pd.read_csv(path, parse_dates=['Date/Time']) for path in files}

# Ensure 'Date/Time' is in datetime format
for station in data:
    data[station]['Date/Time'] = pd.to_datetime(data[station]['Date/Time'])


# Function to filter data by year range
def filter_data_by_year(data, start_year, end_year):
    filtered_data = {}
    for station, df in data.items():
        # Filter rows where the year is between start_year and end_year
        filtered_data[station] = df[(df['Date/Time'].dt.year >= start_year) & (df['Date/Time'].dt.year <= end_year)]
    return filtered_data


# Use data from 2006 to 2015
filtered_data = filter_data_by_year(data, 2006, 2015)


def simulate_missing_data(data, target_columns, missing_fraction=0.05, max_missing_length=24):
    np.random.seed(42)
    data_copy = data.copy()
    simulated_missing_mask = pd.DataFrame(False, index=data.index, columns=target_columns)

    for target in target_columns:
        # Identify indices where missing values already exist
        pre_existing_missing_indices = set(data_copy[data_copy[target].isna()].index)

        # Identify fully non-missing contiguous blocks for each target column
        valid_ranges = []
        start_idx = None

        for idx in data_copy.index:
            if not np.isnan(data_copy.at[idx, target]):
                if start_idx is None:
                    start_idx = idx
            else:
                if start_idx is not None:
                    end_idx = idx - 1
                    if data_copy.loc[start_idx:end_idx, target].notna().all():
                        valid_ranges.append((start_idx, end_idx))
                    start_idx = None
        if start_idx is not None:
            end_idx = data_copy.index[-1]
            if data_copy.loc[start_idx:end_idx, target].notna().all():
                valid_ranges.append((start_idx, end_idx))

        total_valid_points = sum(end - start + 1 for start, end in valid_ranges)
        missing_points = int(missing_fraction * total_valid_points)
        missing_indices = []
        attempts = 0

        while len(missing_indices) < missing_points and attempts < missing_points * 10:
            start, end = valid_ranges[np.random.choice(len(valid_ranges))]
            range_length = end - start + 1
            actual_length = min(max_missing_length, range_length)
            if actual_length > 1 and (end - actual_length + 1) > start:
                high_bound = end - actual_length + 1
                if start < high_bound:
                    start_idx = np.random.randint(start, high_bound)
                    end_idx = start_idx + actual_length - 1
                    if data_copy.loc[start_idx:end_idx, target].notna().all():
                        simulated_range_indices = set(range(start_idx, end_idx + 1))
                        if simulated_range_indices.isdisjoint(pre_existing_missing_indices):
                            missing_indices.append((start_idx, end_idx))
                            simulated_missing_mask.loc[start_idx:end_idx, target] = True

                            # Print original values before setting to NaN for comparison
                            original_values = data_copy.loc[start_idx:end_idx, target].values
                            print(f"Simulated missing values for '{target}' from {start_idx} to {end_idx}")
                            print(f"  Original Values: {original_values}")
                    else:
                        pass

        # Apply NaN to simulated ranges and verify with print statements
        for start_idx, end_idx in missing_indices:
            if data_copy.loc[start_idx:end_idx, target].notna().all():
                data_copy.loc[start_idx:end_idx, target] = np.nan

    # Debug print for simulated mask coverage and comparison with original data
    print("\nFinal Simulated Missing Mask Coverage (True values):")
    for target in target_columns:
        true_indices = simulated_missing_mask[simulated_missing_mask[target]].index
        print(f"Simulated missing indices for '{target}': {true_indices}")

        # Verify mask consistency by printing masked indices and corresponding original data points
        mask_indices = simulated_missing_mask[simulated_missing_mask[target]].index
        masked_values = data.loc[mask_indices, target].values
        print(f"  Masked Indices: {mask_indices}")
        print(f"  Masked Original Values: {masked_values}")

    return data_copy, simulated_missing_mask


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recalculate_similarity_matrix(data, variables, simulated_missing_mask):
    print("Recalculating similarity matrix...")
    station_names = list(data.keys())
    num_stations = len(station_names)
    max_timesteps = max(len(data[station]) for station in station_names)
    similarity_matrix = np.zeros((max_timesteps, num_stations, num_stations))

    zero_similarity_count = 0
    zero_similarity_simulated_count = 0

    for t in range(max_timesteps):
        timestep_data = []
        for station in station_names:
            if t < len(data[station]):
                timestep_data.append(data[station][variables].iloc[t].values)
            else:
                # Append NaNs if station data is shorter than max_timesteps
                timestep_data.append([np.nan] * len(variables))

        timestep_data = np.array(timestep_data)
        similarity_matrix_t = np.zeros((num_stations, num_stations))

        for i in range(num_stations):
            for j in range(i, num_stations):
                valid_mask = ~np.isnan(timestep_data[i]) & ~np.isnan(timestep_data[j])
                if valid_mask.sum() > 0:
                    similarity = cosine_similarity(timestep_data[i][valid_mask].reshape(1, -1),
                                                   timestep_data[j][valid_mask].reshape(1, -1))[0, 0]
                    similarity_matrix_t[i, j] = similarity
                    similarity_matrix_t[j, i] = similarity
                else:
                    similarity_matrix_t[i, j] = similarity_matrix_t[j, i] = 0.0
                    zero_similarity_count += 1

                    # Check if this zero similarity relates to simulated missing data
                    if any(simulated_missing_mask.loc[t, var] for var in variables):
                        zero_similarity_simulated_count += 1
                        #print(f"Zero similarity between '{station_names[i]}' and '{station_names[j]}' at timestep {t} for simulated missing data.")

        similarity_matrix[t] = similarity_matrix_t

    print(f"Total zero similarity cases: {zero_similarity_count}")
    print(f"Zero similarity cases related to simulated missing data: {zero_similarity_simulated_count}")
    if zero_similarity_count > 0:
        percentage_simulated = (zero_similarity_simulated_count / zero_similarity_count) * 100
    else:
        percentage_simulated = 0
    print(f"Percentage of zero similarity cases related to simulated missing data: {percentage_simulated:.2f}%")

    return similarity_matrix


def similarity_fill(data, similarity_matrix, target_station, variables, simulated_missing_mask):
    station_index = list(data.keys()).index(target_station)
    filled_data = data[target_station].copy()
    filled_data = filled_data.sort_values(by='Date/Time').reset_index(drop=True)

    # Ensure indices in simulated_missing_mask match those in filled_data
    simulated_missing_mask = simulated_missing_mask.reindex(filled_data.index)

    for variable in variables:
        # Get simulated missing indices for this variable
        simulated_missing_indices = simulated_missing_mask[simulated_missing_mask[variable]].index
        print(f"\nProcessing variable '{variable}'")
        print(f"  Simulated missing indices for filling: {simulated_missing_indices}")

        for t in simulated_missing_indices:
            print(f"\nProcessing simulated missing value at index {t} for '{variable}'")

            # Retrieve similarity values and other station data for this index
            row_similarities = similarity_matrix[t, station_index]
            print(f"    Similarity values at index {t}: {row_similarities}")

            weighted_sum = 0
            weight_sum = 0
            valid_values_count = 0

            for other_station_index, similarity in enumerate(row_similarities):
                if other_station_index != station_index:
                    other_station_name = list(data.keys())[other_station_index]
                    if t < len(data[other_station_name]):
                        other_station_value = data[other_station_name][variable].iloc[t]
                        if not np.isnan(other_station_value):
                            print(
                                f"      Using value from '{other_station_name}' at index {t}: {other_station_value} with similarity {similarity}")
                            weighted_sum += other_station_value * similarity
                            weight_sum += similarity
                            valid_values_count += 1

            if weight_sum > 0 and valid_values_count > 0:
                filled_value = weighted_sum / weight_sum
                print(f"  Filling missing value for '{variable}' at index {t}: {filled_value}")
                filled_data.at[t, variable] = filled_value
            else:
                print(
                    f"  Skipping fill for index {t} due to lack of valid data (weight_sum: {weight_sum}, valid values: {valid_values_count})")

    print("\nSimilarity-based filling process completed.\n")
    return filled_data


# Function to evaluate the filling quality
def evaluate_fill_methods(original, filled, target_columns):
    evaluation_results = {}
    for target in target_columns:
        filled_indices = filled[filled[target].notna() & original[target].notna()].index
        original_non_null = original.loc[filled_indices, target]
        filled_non_null = filled.loc[filled_indices, target]

        if not filled_non_null.empty:
            mse = mean_squared_error(original_non_null, filled_non_null)
            mae = mean_absolute_error(original_non_null, filled_non_null)
            mape = mean_absolute_percentage_error(original_non_null, filled_non_null)

            evaluation_results[target] = {
                'MSE': mse,
                'MAE': mae,
                'MAPE': mape
            }
        else:
            evaluation_results[target] = {'MSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}

    return pd.DataFrame(evaluation_results).T


validation_data, simulated_missing_mask = simulate_missing_data(filtered_data[target_station].copy(), variables)

# Recalculate the similarity matrix
similarity_matrix = recalculate_similarity_matrix(filtered_data, variables, simulated_missing_mask)

# Fill using the mask to ensure only simulated missing values are filled
filled_data_similarity = similarity_fill(filtered_data, similarity_matrix, target_station, variables, simulated_missing_mask)

# Evaluate the filling quality
eval_results = evaluate_fill_methods(validation_data, filled_data_similarity, variables)
print(f"Similarity Interpolation results for {target_station} from 2006 to 2015:\n", eval_results)

