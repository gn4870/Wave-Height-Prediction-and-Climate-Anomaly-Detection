import os
import pandas as pd
import lightgbm as lgb
from datetime import timedelta
import numpy as np

files = ['Hourly_Data/41008.csv', 'Hourly_Data/41009.csv', 'Hourly_Data/41010.csv', 'Hourly_Data/42022.csv',
         'Hourly_Data/42036.csv', 'Hourly_Data/fwyf1.csv', 'Hourly_Data/smkf1.csv', 'Hourly_Data/venf1.csv']

threshold_large = timedelta(days=60)
threshold_small = timedelta(hours=6)

# 定义目标列（要填补缺失值的列）
target_columns = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP']

# 创建输出文件夹，如果不存在则创建
output_folder = 'Processed_Data'
os.makedirs(output_folder, exist_ok=True)


# 定义一个函数来处理每个文件
def process_file(file_path):
    print(f"Processing file: {file_path}")

    # 1. 加载数据
    data = pd.read_csv(file_path)
    data['Date/Time'] = pd.to_datetime(data['Date/Time'])

    # 2. 提取时间特征
    data['timestamp'] = data['Date/Time'].apply(lambda x: x.timestamp())
    data['year'] = data['Date/Time'].dt.year
    data['month'] = data['Date/Time'].dt.month
    data['day'] = data['Date/Time'].dt.day
    data['hour'] = data['Date/Time'].dt.hour

    # 3. 按时间排序
    data = data.sort_values(by='Date/Time').reset_index(drop=True)

    # 4. 初始化 LightGBM 模型
    model = lgb.LGBMRegressor()

    # 5. 对每个目标列进行填补
    for target_column in target_columns:
        print(f"  Filling missing values for column: {target_column}")

        # 识别缺失值，并标记缺失段
        data['missing'] = data[target_column].isna()
        data['group'] = (data['missing'] != data['missing'].shift()).cumsum()

        # 获取缺失段总数
        total_missing_segments = data['group'].nunique() - data['missing'].sum()

        # 遍历每个缺失段
        for group_id, group in data.groupby('group'):
            if group['missing'].iloc[0]:  # 如果该段是缺失值
                # 计算该缺失段的时间跨度
                time_span = group['Date/Time'].max() - group['Date/Time'].min()

                if time_span > threshold_large:
                    # 如果时间跨度超过60天，保持为 NaN，不进行填充
                    data.loc[group.index, target_column] = np.nan
                    print(f"    Skipping group {group_id} (larger than 60 days)")
                elif time_span <= threshold_small:
                    # 如果时间跨度小于等于6小时，使用线性插值填补
                    data.loc[group.index, target_column] = data[target_column].interpolate(
                        method='linear', limit_direction='both', limit_area='inside'
                    )
                    print(f"    Interpolated group {group_id} (less than 6 hours)")
                else:
                    # 其他情况使用 LightGBM 进行预测
                    # 提取训练集，非缺失的观测值
                    X_train = data[~data[target_column].isna()].drop(
                        columns=[target_column, 'missing', 'group', 'Date/Time'])
                    y_train = data[~data[target_column].isna()][target_column]

                    # 提取需要预测的缺失段
                    X_missing = group.drop(columns=[target_column, 'missing', 'group', 'Date/Time'])

                    # 模型训练
                    model.fit(X_train, y_train)

                    # 预测缺失段
                    y_pred = model.predict(X_missing)

                    # 填充预测值
                    data.loc[group.index, target_column] = y_pred
                    print(f"    Filled group {group_id} using LightGBM")

            # 输出剩余的未处理缺失段数量
            total_missing_segments -= 1
            print(f"    Remaining missing segments for {target_column}: {total_missing_segments}")

        # 删除临时列
        data = data.drop(columns=['missing', 'group'])

    # 6. 保存处理后的数据
    output_file = os.path.join(output_folder, os.path.basename(file_path))
    data.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")


# 处理每个文件
for file_path in files:
    process_file(file_path)

print("All files processed.")



