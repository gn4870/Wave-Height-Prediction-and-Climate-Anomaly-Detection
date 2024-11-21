import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
# from statsmodels.stats.diagnostic import acf
from statsmodels.tsa.stattools import acf
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats


def perform_stl_decomposition(data, column_name, period=12):
    """
    Perform STL decomposition on the time series data.
    """
    # Ensure the data is complete and equally spaced
    data_clean = data[column_name].interpolate(method='linear')

    # Perform STL decomposition
    stl = STL(data_clean, period=period)
    result = stl.fit()

    return result


def check_residual_autocorrelation(residuals, nlags=40):
    """
    Check if residuals are autocorrelated using ACF.
    Returns acf values and boolean indicating significant autocorrelation.
    """
    acf_values = acf(residuals, nlags=nlags)
    # Calculate confidence intervals (95%)
    confidence_interval = 1.96 / np.sqrt(len(residuals))

    # Check if any ACF values (excluding lag 0) are outside confidence intervals
    is_autocorrelated = any(abs(acf_values[1:]) > confidence_interval)

    return acf_values, is_autocorrelated


def fit_arima_model(residuals):
    """
    Model residuals using ARIMA if they are autocorrelated.
    """
    # Determine order using AIC
    best_aic = np.inf
    best_order = None

    # Try different ARIMA orders
    for p in range(4):
        for q in range(4):
            try:
                model = ARIMA(residuals, order=(p, 0, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, 0, q)
            except:
                continue

    if best_order is not None:
        try:
            final_model = ARIMA(residuals, order=best_order)
            fitted_model = final_model.fit()
            return fitted_model
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            return None
    return None


def detect_outliers_stl(data, column_name, threshold=3):
    """
    Detect outliers using STL decomposition and residual analysis.
    """
    try:
        # Perform STL decomposition
        stl_result = perform_stl_decomposition(data, column_name)
        residuals = stl_result.resid

        # Check for autocorrelation in residuals
        acf_values, is_autocorrelated = check_residual_autocorrelation(residuals)

        if is_autocorrelated:
            # Model the residuals with ARIMA
            arima_result = fit_arima_model(residuals)
            if arima_result is not None:
                # Use standardized residuals from the ARIMA model
                adjusted_residuals = arima_result.resid
                standardized_residuals = (adjusted_residuals - adjusted_residuals.mean()) / adjusted_residuals.std()
            else:
                # Fallback to original residuals if ARIMA modeling fails
                print(f"ARIMA modeling failed for {column_name}, using original residuals")
                standardized_residuals = (residuals - residuals.mean()) / residuals.std()
        else:
            # Use original residuals if no significant autocorrelation
            standardized_residuals = (residuals - residuals.mean()) / residuals.std()

        # Detect outliers
        outliers = abs(standardized_residuals) > threshold

        return outliers, stl_result, standardized_residuals

    except Exception as e:
        print(f"Error in outlier detection for {column_name}: {e}")
        # Return empty results in case of error
        return pd.Series(False, index=data.index), None, pd.Series(0, index=data.index)


def plot_stl_outliers(data, column_name, outliers, stl_result, standardized_residuals):
    """
    Plot the original data, STL decomposition, and outliers.
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    # Plot original data with outliers
    axes[0].plot(data.index, data[column_name], 'b-', label='Original')
    outlier_points = data[column_name][outliers]
    axes[0].scatter(outlier_points.index, outlier_points, color='red', label='Outliers')
    axes[0].set_title(f'{column_name} Time Series with Outliers')
    axes[0].legend()

    # Plot trend
    axes[1].plot(data.index, stl_result.trend)
    axes[1].set_title('Trend')

    # Plot seasonal
    axes[2].plot(data.index, stl_result.seasonal)
    axes[2].set_title('Seasonal')

    # Plot standardized residuals with threshold lines
    axes[3].plot(data.index, standardized_residuals)
    axes[3].axhline(y=3, color='r', linestyle='--')
    axes[3].axhline(y=-3, color='r', linestyle='--')
    axes[3].set_title('Standardized Residuals')

    plt.tight_layout()
    return fig


def main():
    # Set up paths and variables
    data_folder = 'Filled_Data_2016_2023/'
    variables = ['ATMP', 'WTMP', 'WVHT']
    # target_stations = ['41008_filled_2016_2023.csv', '41009_filled_2016_2023.csv',
    #                    '41010_filled_2016_2023.csv', '42022_filled_2016_2023.csv',
    #                    '42036_filled_2016_2023.csv','fwyf1_filled_2016_2023.csv',
    #                    'smkf1_filled_2016_2023.csv','venf1_filled_2016_2023.csv']
    target_stations = ['41009_filled_2016_2023.csv','41010_filled_2016_2023.csv', '42022_filled_2016_2023.csv']

    # Create output directory for plots
    output_dir = 'STL_Outlier_Analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Process each station
    for file_name in target_stations:
        file_path = os.path.join(data_folder, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"\nProcessing {file_name}")
        df = pd.read_csv(file_path)
        df['Date/Time'] = pd.to_datetime(df['Date/Time'])
        df.set_index('Date/Time', inplace=True)

        # Process each variable
        for variable in variables:
            if variable not in df.columns:
                continue

            print(f"\nAnalyzing {variable}")

            # Detect outliers
            outliers, stl_result, standardized_residuals = detect_outliers_stl(df, variable)

            if stl_result is not None:
                # Calculate outlier statistics
                outlier_ratio = outliers.mean()
                print(f"Outlier ratio for {variable}: {outlier_ratio:.2%}")

                # Plot results
                fig = plot_stl_outliers(df, variable, outliers, stl_result, standardized_residuals)

                # Save plot
                plot_path = os.path.join(output_dir, f"{file_name.replace('.csv', '')}_{variable}_stl_analysis.png")
                fig.savefig(plot_path)
                plt.close(fig)

                # Save outlier information
                df[f'Outlier_{variable}_STL'] = outliers
            else:
                print(f"STL decomposition failed for {variable}")

        # Save processed data
        output_path = os.path.join(output_dir, f"{file_name.replace('.csv', '')}_with_outliers.csv")
        df.to_csv(output_path)
        print(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    main()