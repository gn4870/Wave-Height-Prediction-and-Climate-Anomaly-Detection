import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import h5py
import logging
from datetime import datetime


def setup_logging():
    """Configure logging with timestamp in filename."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'filling_process_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename


def calculate_similarity_matrix(data, variables):
    """Calculate similarity matrix for all stations and timesteps."""
    logging.info("Calculating new similarity matrix...")
    station_names = list(data.keys())
    num_stations = len(station_names)
    max_timesteps = max(len(data[station]) for station in station_names)
    similarity_matrix = np.zeros((max_timesteps, num_stations, num_stations))

    for t in range(max_timesteps):
        if t % 1000 == 0:  # Log progress every 1000 timesteps
            logging.info(f"Processing timestep {t}/{max_timesteps}")

        timestep_data = []
        for station in station_names:
            if t < len(data[station]):
                timestep_data.append(data[station][variables].iloc[t].values)
            else:
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

        similarity_matrix[t] = similarity_matrix_t

    return similarity_matrix


def get_similarity_matrix(data, variables, output_dir):
    """Get similarity matrix from cache or calculate if not exists."""
    matrix_path = os.path.join(output_dir, 'similarity_matrix_2016_2023.h5')

    if os.path.exists(matrix_path):
        logging.info("Loading cached similarity matrix...")
        try:
            with h5py.File(matrix_path, 'r') as f:
                similarity_matrix = f['similarity_matrix'][:]
            logging.info("Cached similarity matrix loaded successfully")
            return similarity_matrix
        except Exception as e:
            logging.warning(f"Error loading cached similarity matrix: {e}")
            logging.info("Calculating new similarity matrix...")
    else:
        logging.info("No cached similarity matrix found")

    # Calculate new similarity matrix
    similarity_matrix = calculate_similarity_matrix(data, variables)

    # Save the new matrix
    logging.info("Saving new similarity matrix...")
    try:
        with h5py.File(matrix_path, 'w') as f:
            f.create_dataset('similarity_matrix', data=similarity_matrix)
        logging.info("Similarity matrix saved successfully")
    except Exception as e:
        logging.error(f"Error saving similarity matrix: {e}")

    return similarity_matrix


def fill_missing_data(data, similarity_matrix, target_station, variables):
    """Fill missing values using similarity-based interpolation with detailed logging."""
    station_names = list(data.keys())
    station_index = station_names.index(target_station)
    filled_data = data[target_station].copy()
    filled_data = filled_data.sort_values(by='Date/Time').reset_index(drop=True)

    total_missing = {var: filled_data[var].isna().sum() for var in variables}
    filled_count = {var: 0 for var in variables}

    logging.info(f"\nProcessing station: {target_station}")
    for var, missing in total_missing.items():
        logging.info(f"Variable {var}: {missing} missing values")

    for variable in variables:
        missing_indices = filled_data[filled_data[variable].isna()].index
        logging.info(f"\nProcessing variable: {variable}")
        logging.info(f"Total missing values: {len(missing_indices)}")

        for t in missing_indices:
            row_similarities = similarity_matrix[t, station_index]
            weighted_sum = 0
            weight_sum = 0
            contributing_stations = []

            for other_station_index, similarity in enumerate(row_similarities):
                if other_station_index != station_index:
                    other_station_name = station_names[other_station_index]
                    if t < len(data[other_station_name]):
                        other_station_value = data[other_station_name][variable].iloc[t]
                        if not np.isnan(other_station_value):
                            weighted_sum += other_station_value * similarity
                            weight_sum += similarity
                            contributing_stations.append({
                                'station': other_station_name,
                                'value': other_station_value,
                                'similarity': similarity
                            })

            if weight_sum > 0:
                filled_value = weighted_sum / weight_sum
                filled_data.at[t, variable] = filled_value
                filled_count[variable] += 1

        # Log filling statistics for this variable
        total_missing_var = total_missing[variable]
        filled_var = filled_count[variable]
        if total_missing_var > 0:
            fill_rate = (filled_var / total_missing_var) * 100
            logging.info(f"\nVariable {variable} filling statistics:")
            logging.info(f"Total missing: {total_missing_var}")
            logging.info(f"Successfully filled: {filled_var}")
            logging.info(f"Fill rate: {fill_rate:.2f}%")

    return filled_data, filled_count, total_missing


def main():
    # Set up logging
    log_filename = setup_logging()

    # Define parameters
    files = [
        'Hourly_Data/41008.csv', 'Hourly_Data/41009.csv', 'Hourly_Data/41010.csv',
        'Hourly_Data/42022.csv', 'Hourly_Data/42036.csv', 'Hourly_Data/fwyf1.csv',
        'Hourly_Data/smkf1.csv', 'Hourly_Data/venf1.csv'
    ]
    variables = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP']
    output_dir = 'Filled_Data_2016_2023'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data and filter for 2016-2023
    logging.info("Loading and filtering data for 2016-2023...")
    data = {}
    for path in files:
        station_name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, parse_dates=['Date/Time'])
        df = df[(df['Date/Time'].dt.year >= 2016) & (df['Date/Time'].dt.year <= 2023)]
        data[station_name] = df
        logging.info(f"Loaded {station_name}: {len(df)} records")

    # Get or calculate similarity matrix
    similarity_matrix = get_similarity_matrix(data, variables, output_dir)

    # Process all stations
    fill_statistics = {}
    for station in data.keys():
        logging.info(f"\nProcessing station: {station}")

        # Fill missing data
        filled_data, filled_count, total_missing = fill_missing_data(data, similarity_matrix, station, variables)

        # Save filled data
        output_path = os.path.join(output_dir, f'{station}_filled_2016_2023.csv')
        filled_data.to_csv(output_path, index=False)
        logging.info(f"Filled data saved to {output_path}")

        # Store statistics
        fill_statistics[station] = {
            'total_missing': total_missing,
            'filled_count': filled_count
        }

    # Log summary statistics
    logging.info("\n=== OVERALL FILLING STATISTICS ===")
    for station in fill_statistics:
        logging.info(f"\nStation: {station}")
        for variable in variables:
            total = fill_statistics[station]['total_missing'][variable]
            filled = fill_statistics[station]['filled_count'][variable]
            if total > 0:
                fill_rate = (filled / total) * 100
                logging.info(f"{variable}: {filled}/{total} filled ({fill_rate:.2f}%)")

    logging.info(f"\nProcess complete. Full log saved to: {log_filename}")


if __name__ == "__main__":
    main()