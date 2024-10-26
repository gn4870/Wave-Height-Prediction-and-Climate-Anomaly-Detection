import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 定义函数以检测异常值
def detect_outliers_ma(data, column_name, window_size=24, threshold_multiplier=3):
    # Calculate moving average and moving standard deviation
    rolling_mean = data[column_name].rolling(window=window_size, center=True).mean()
    rolling_std = data[column_name].rolling(window=window_size, center=True).std()

    # Calculate the upper and lower bounds for outliers
    upper_bound = rolling_mean + (threshold_multiplier * rolling_std)
    lower_bound = rolling_mean - (threshold_multiplier * rolling_std)

    # Mark outliers and return only the outlier column
    outlier_col = (data[column_name] > upper_bound) | (data[column_name] < lower_bound)
    return outlier_col

# 定义绘制异常值的函数
def plot_outliers(data, column_name):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[column_name], color='blue', label=f'{column_name}')
    
    # 标记异常值
    outliers = data[data['Outlier_' + column_name] == True]
    plt.scatter(outliers.index, outliers[column_name], color='red', s=60, marker='o', label='Outliers')
    
    # 添加标签和标题
    plt.xlabel('Date/Time')
    plt.ylabel(column_name)
    plt.title(f'{column_name} with Outliers Detected Using Moving Average')
    plt.legend()
    plt.show()

# 处理所有的.csv文件
data_folder = 'Processed_Data/'
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

results = {}
for file_name in csv_files:
    file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(file_path)

    # 将'Date/Time'列转换为日期时间格式
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])

    # 设置'Date/Time'为索引
    df.set_index('Date/Time', inplace=True)

    # 将数据重采样为每两小时，并填充缺失值
    df = df.resample('2H').mean()
    df = df.fillna(method='ffill')

    # 对'ATMP'、'WTMP'和'WVHT'进行异常值检测
    for col in ['ATMP', 'WTMP', 'WVHT']:
        if col in df.columns:
            df['Outlier_' + col] = detect_outliers_ma(df, col)

    # 存储处理后的数据
    results[file_name] = df

# 只展示41008和41009的数据
for file_name in ['41008.csv', '41009.csv']:
    if file_name in results:
        df_result = results[file_name]
        print(f"\nOutlier Ratios for {file_name}:")
        for col in ['ATMP', 'WTMP', 'WVHT']:
            if 'Outlier_' + col in df_result.columns:
                outlier_ratio = df_result['Outlier_' + col].mean()
                print(f"{col} Outlier Ratio: {outlier_ratio:.2%}")

        # 绘制异常值的图
        for col in ['ATMP', 'WTMP', 'WVHT']:
            if col in df_result.columns:
                plot_outliers(df_result, col)
