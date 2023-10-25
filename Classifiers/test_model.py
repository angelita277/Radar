from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from LSTM import one_hot_encoding
import tensorflow as tf
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# tf.config.set_visible_devices([], 'GPU')

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print("TensorFlow Version: ", tf.__version__)

frames_together = 20
sub_dirs = ['fall', 'else']


def softmax_(a):
    length = len(a)
    max_ = 0
    maxnum = -1
    for i in range(length):
        if a[i] >= max_:
            max_ = a[i]
            maxnum = i
    return maxnum


def classi_acc(pre, label):
    length = len(label)
    #print(length)
    right = 0
    wrong = 0
    wrong_count = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    for i in range(length):
        #print(label[i], " ", sub_dirs[softmax_(pre[i])])
        if label[i] == sub_dirs[softmax_(pre[i])]:
            right = right + 1
        else:
            wrong = wrong + 1
            #print(label[i], "to", sub_dirs[softmax_(pre[i])])

    return right / (right + wrong)


def acc(pre, label):
    TP = FN = FP = TN = 0
    M = len(label)
    for i in range(M):
        # print(softmax_(pre[i]))
        # print(label[i])
        if label[i] == "fall":
            if softmax_(pre[i]) == 0:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if softmax_(pre[i]) == 0:
                FP = FP + 1
            else:
                TN = TN + 1
    acc = (TP + TN) / M
    recall = TP / (TP + FN)
    false_alarm = FP / (TP + FP)
    missing_alarm = FN / (TP + FN)
    print("TP:", TP, "FN:", FN, "FP:", FP, "TN:", TN)
    print("acc:", acc)
    print("recall:", recall)
    print("false_alarm:", false_alarm)
    print("missing_alarm:", missing_alarm)
    return


# 加载训练好的模型
checkpoint_model_path = "D:/研究生/研一/跌倒检测/1017/Radar/Classifiers/LSTM_dynamic"
model = load_model(checkpoint_model_path)

# 加载测试数据
data_file = "D:\\研究生\研一\\跌倒检测\\1017\\data_6.16_dynamic_txt\\test_voxel_data.npz"
data = np.load(data_file)  # 根据模型的输入要求准备测试数据
test_data = data["arr_0"]
test_data = np.reshape(test_data, (-1, frames_together, 10 * 32 * 32))
print(test_data.shape)
test_label = data["arr_1"]
for i in range(len(test_label)):
    if test_label[i] != 'fall':
        test_label[i] = 'else'

# 进行预测
#print(test_label)
# print(one_hot_encoding(test_label,sub_dirs))

predictions = model.predict(test_data)
pre = []

for i in range(len(predictions)):
    pre.append(sub_dirs[softmax_(predictions[i])])

print("classification acc:")
print(classi_acc(predictions, test_label))

acc(predictions, test_label)

C = confusion_matrix(test_label, pre, labels=['fall', 'else'])
plt.matshow(C, cmap=plt.cm.Greens)

for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#print(test_label)

# acc(predictions, test_label)
# 输出预测结果
# print(predictions)
# print(np.shape(test_data["arr_0"]))
