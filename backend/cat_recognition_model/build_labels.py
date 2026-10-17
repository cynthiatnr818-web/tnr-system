import os
import pandas as pd

# 指定训练集和测试集的目录路径
train_dir = '/Users/apple/Desktop/research/TNRproject/CatFaceDataset/train'
test_dir = '/Users/apple/Desktop/research/TNRproject/CatFaceDataset/test'

# 用于存储标签和文件名的列表
train_labels = []
test_labels = []
train_file_names = []
test_file_names = []

# 处理训练集
for breed_folder in os.listdir(train_dir):
    breed_folder_path = os.path.join(train_dir, breed_folder)
    if os.path.isdir(breed_folder_path):  # 只处理文件夹
        for image_file in os.listdir(breed_folder_path):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):  # 可根据需要调整文件类型
                train_file_names.append(image_file)
                train_labels.append(breed_folder)  # 文件夹名称作为标签

# 处理测试集
for breed_folder in os.listdir(test_dir):
    breed_folder_path = os.path.join(test_dir, breed_folder)
    if os.path.isdir(breed_folder_path):  # 只处理文件夹
        for image_file in os.listdir(breed_folder_path):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):  # 可根据需要调整文件类型
                test_file_names.append(image_file)
                test_labels.append(breed_folder)  # 文件夹名称作为标签

# 创建训练集和测试集的 DataFrame
train_data = pd.DataFrame({
    'file_name': train_file_names,
    'label': train_labels
})

test_data = pd.DataFrame({
    'file_name': test_file_names,
    'label': test_labels
})

# 保存为CSV文件
train_csv_file = '/Users/apple/Desktop/research/TNRproject/CatFaceDataset/train_labels.csv'
test_csv_file = '/Users/apple/Desktop/research/TNRproject/CatFaceDataset/test_labels.csv'

train_data.to_csv(train_csv_file, index=False)
test_data.to_csv(test_csv_file, index=False)

print(f"训练集标签文件已保存为 {train_csv_file}")
print(f"测试集标签文件已保存为 {test_csv_file}")
