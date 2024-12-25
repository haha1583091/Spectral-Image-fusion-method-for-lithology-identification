import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

from scipy.io import loadmat, savemat
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA


def applyPCA(data, n_components):
    """
    应用PCA到数据并返回PCA处理后的数据
    """
    pca = PCA(n_components=n_components)
    reshaped_data = data.reshape(-1, data.shape[-1])
    data_pca = pca.fit_transform(reshaped_data)
    data_pca = data_pca.reshape(data.shape[0], data.shape[1], n_components)
    return data_pca, np.sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_


def plot_pca_components(data_pca, n_components):
    """
    绘制PCA分量图
    """
    fig, axes = plt.subplots(1, n_components, figsize=(30, 8))
    for i, ax in enumerate(axes):
        ax.imshow(data_pca[:, :, i], cmap='gray')
        ax.set_title(f'PC{i + 1}')
        ax.axis('off')
    plt.show()


def save_texture_to_mat(texture_image, X_path):
    """
    将纹理特征图像保存为.mat文件
    """
    output_path = X_path.replace('.mat', '_texture.mat')
    savemat(output_path, {'texture': texture_image})
    return output_path


def save_pca_components(data_pca, n_components, output_dir, dpi=1600):
    """
    保存到指定文件夹
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(n_components):
        plt.figure(figsize=(8, 8))
        plt.imshow(data_pca[:, :, i], cmap='gray')
        # plt.title(f'PC{i+1}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'PC{i + 1}.png'))
        plt.close()


output_directory = r"D:\Jupyter_project\01_HSI data classification\PC_images"
save_pca_components(data_pca, 10, output_directory)

# 加载.mat文件
X_path = r'D:\Code_data\光谱+纹理+标签_mat\9.mat'
data = loadmat(X_path)['data']

print('初始data维度：', data.shape)

# PCA分析
pca_components = 10
data_pca, cumulative_variance, variance_ratio = applyPCA(data, pca_components)

print('PCA处理后data维度：', data_pca.shape)
print('累积可解释性方差：', cumulative_variance)

# 打印每个PC的方差占比
for i, variance in enumerate(variance_ratio):
    print(f"PC{i+1}: {variance * 100:.2f}%")

# 调用函数来绘制和保存
plot_pca_components(data_pca, pca_components)

# 选择进行特征提取的主成分（例如PC1）
pc1 = data_pca[:, :, 0]
height, width = pc1.shape
print('pc1图像维度：', pc1.shape)

print('纹理特征提取', '\n')
# 对PC1进行纹理特征提取, 将数据转换为8位整数
pc1 = (pc1 / pc1.max() * 255).astype(np.uint8)

# 初始化存储纹理特征图像的数组 (高度 x 宽度 x 4个特征)
texture_features = np.zeros((height, width, 4))  # 对比度, 均匀性, 能量, ASM

# 定义窗口大小
window_size = 5

for row in range(pc1.shape[0]):
    if row % 100 == 0:
        print(f"Processing row {row} in PC1")
    for col in range(pc1.shape[1]):
        row_start = max(row - window_size // 2, 0)
        row_end = min(row + window_size // 2 + 1, pc1.shape[0])
        col_start = max(col - window_size // 2, 0)
        col_end = min(col + window_size // 2 + 1, pc1.shape[1])

        window = pc1[row_start:row_end, col_start:col_end]

        glcm = graycomatrix(window, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

        texture_features[row, col, 0] = graycoprops(glcm, 'contrast')[0, 0]
        texture_features[row, col, 1] = graycoprops(glcm, 'homogeneity')[0, 0]
        texture_features[row, col, 2] = graycoprops(glcm, 'energy')[0, 0]
        texture_features[row, col, 3] = graycoprops(glcm, 'ASM')[0, 0]

print('纹理特征提取完成', '\n')

# 对每个特征通道进行归一化
for i in range(texture_features.shape[2]):
    texture_features[:, :, i] = (texture_features[:, :, i] - texture_features[:, :, i].min()) / (
                texture_features[:, :, i].max() - texture_features[:, :, i].min())

print('纹理特征归一化完成', '\n')

# 保存纹理特征图像到.mat文件
output_path = save_texture_to_mat(texture_features, X_path)
print(f"纹理特征成功保存至: {output_path}")


# 加载.mat文件
X_path = r'D:\Code_data\光谱+纹理+标签_mat\9_texture.mat'
data = loadmat(X_path)['texture']

print('初始data维度：', data.shape)

# 遍历每个通道，计算并打印最大和最小值
num_channels = data.shape[2]  # 假设通道是在第三维
for i in range(num_channels):
    channel_data = data[:, :, i]
    min_val = np.min(channel_data)
    max_val = np.max(channel_data)
    print(f"通道 {i+1} - 最小值: {min_val}, 最大值: {max_val}")