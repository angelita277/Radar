"""
划分训练集和测试集

USAGE: split the train_data and test_data

- src_dir_path：源文件主路径
- sub_dir：源文件子路径
- dst_dir_path：目标文件夹路径
- test_rate：测试集所占比例

"""


import os
import random
import shutil

def split_data(src_dir_path, sub_dir, dst_dir_path, test_rate):
    for sd in sub_dir:
        full_src_path = src_dir_path + sd

        # 获取源文件夹中所有文件的列表
        files = [os.path.join(full_src_path, f) for f in os.listdir(full_src_path)]
        #print(len(files))

        # 随机选择文件
        random_files = random.sample(files, int(len(files) * test_rate))
        #print(random_files)

        # 如果目标文件夹不存在，则创建它
        if not os.path.exists(dst_dir_path + 'train/' + sd):
            os.makedirs(dst_dir_path + 'train/' + sd)
        if not os.path.exists(dst_dir_path + 'test'):
            os.makedirs(dst_dir_path + 'test')

        # 将文件复制到训练/测试文件夹中
        for file_path in os.listdir(full_src_path):
            #print(file_path)
            if full_src_path + '\\' + file_path in random_files:
                shutil.copy(full_src_path + '\\' + file_path, dst_dir_path + 'test')
            else:
                shutil.copy(full_src_path + '\\' + file_path, dst_dir_path + 'train/' + sd)


if __name__ == "__main__":
    src_dir_path = "D:/研究生/研一/跌倒检测/1017/data_6.16_dynamic_txt/"
    sub_dir = ['fall', 'lie', 'sit', 'walk']
    dst_dir_path = "D:/研究生/研一/跌倒检测/1017/data_6.16_dynamic_txt/"

    split_data(src_dir_path, sub_dir, dst_dir_path, 0.3)