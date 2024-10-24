import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import h5py
import cupy as cp

files = ['Hourly_Data/41008.csv', 'Hourly_Data/41009.csv', 'Hourly_Data/41010.csv', 'Hourly_Data/42022.csv',
         'Hourly_Data/42036.csv','Hourly_Data/fwyf1.csv','Hourly_Data/smkf1.csv','Hourly_Data/venf1.csv']

# files = ['Hourly_Data/41008.csv', 'Hourly_Data/41009.csv', 'Hourly_Data/41010.csv']

# 用于存储每个表格的数据
data_frames = []

# 从CSV文件中读取每个表格并存储到列表中
for file in files:
    df = pd.read_csv(file)

    # 保留数值型数据列，并用0填充NaN
    df_numeric = df.select_dtypes(include=[np.number]).fillna(0)

    # 如果长度不一致，补齐至相同长度
    max_length = max(df_numeric.shape[0] for df_numeric in data_frames) if data_frames else df_numeric.shape[0]
    df_numeric = df_numeric.reindex(range(max_length), fill_value=0)

    data_frames.append(df_numeric.mean(axis=1).values)  # 每行取均值作为时间序列向量

# 将所有表格的数据转化为矩阵
combined_data = pd.DataFrame(data_frames).T  # 转置使其成为表格间相似度计算的矩阵

# 将矩阵转为 CuPy 数组，放到 GPU 上
combined_data_gpu = cp.asarray(combined_data)

# 分块大小
chunk_size = 500
n_rows = combined_data_gpu.shape[0]

# 打开HDF5文件，用于存储分块结果
with h5py.File('cosine_similarity.h5', 'w') as f:
    # 创建一个大矩阵用于存储分块结果
    cosine_sim = f.create_dataset('cosine_sim', shape=(n_rows, n_rows), dtype='float32')

    # 分块计算并逐步存储
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk_gpu = combined_data_gpu[start:end]

        # 计算该块与整个数据的余弦相似度（使用CuPy在GPU上加速）
        chunk_sim_gpu = cp.dot(chunk_gpu, combined_data_gpu.T) / (
                    cp.linalg.norm(chunk_gpu, axis=1)[:, None] * cp.linalg.norm(combined_data_gpu, axis=1))

        # 将结果从GPU内存转回CPU内存并存储到HDF5文件
        chunk_sim = cp.asnumpy(chunk_sim_gpu)
        cosine_sim[start:end, :] = chunk_sim

print("Cosine Similarity Matrix stored in 'cosine_similarity.h5'")

## 查看相似度矩阵
# 打开 HDF5 文件
h5_file = h5py.File("cosine_similarity_8x8.h5", "r")

# 读取存储的相似度矩阵
dataset = h5_file['cosine_sim']

# 获取矩阵的大小
n_rows, n_cols = dataset.shape

# 设置分块大小
chunk_size = 1000

# 逐步读取并处理矩阵
for start in range(0, n_rows, chunk_size):
    end = min(start + chunk_size, n_rows)
    # 读取部分数据
    chunk = dataset[start:end, :]

    # 打印块的信息或对其进行处理
    print(f"Chunk {start}:{end}:\n", chunk)

# 关闭 HDF5 文件
h5_file.close()
