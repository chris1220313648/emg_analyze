import rospy
import sys
import pandas as pd
from joblib import dump, load
from collections import deque
import math
import feature_fn
import numpy as np
sys.path.append("/home/chenpy/code/io_mocap/build/lib/python3/dist-packages/")
from io_mocap.msg import emg  # 假设消息类型为 Emg，根据实际情况修改
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import MinMaxScaler
# 创建一个列表来累积数据
# emg_data_buffer = []
# 设置阈值
threshold = 0.5
emg_data_buffer = deque(maxlen=150)
new_data_count = 0
force_rf = load('model_save/modelStre_force.joblib')
gesture_rf= load('model_save/modelStre.joblib')
force_scaler = load('model_save/scaler_force.joblib')
gesture_scaler = load('model_save/scaler.joblib')
# # 加载最大和最小值数据
# max_min_values = pd.read_csv('./max/min_max_values.csv')
# max_vals = max_min_values['max_value'].values
# min_vals = max_min_values['min_value'].values

# 启用matplotlib的交互模式
plt.ion()

def show_image_based_on_prediction(prediction):
    prediction_to_img_path = {
        0: './gesture_img/0.JPEG',
        1: './gesture_img/1.JPEG',
        2: './gesture_img/2.JPEG',
        3: './gesture_img/3.JPEG',
        4: './gesture_img/4.JPEG',
        5: './gesture_img/5.JPEG',
    }

    # 使用字典的get方法，如果prediction不存在，则返回'default'对应的路径
    prediction = int(prediction)
    img_path = prediction_to_img_path.get(prediction, 'path_to_default_image.jpg')

    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('off')  # 如果不想显示坐标轴
    plt.draw()  # 更新当前图形
    plt.pause(0.01)  # 暂停一会儿，允许图形更新
# def show_name_based_on_prediction(prediction):
#     label_to_gesture_name = {
#         '0': '握拳',
#         '1': '握水杯',
#         '2': '伸开五指',
#         '3': '握笔',
#         '4': '捏卡',
#         '5': '竖大拇指',
#         'a': '小力度',
#         'b': '中等力度',
#         'c': '大力度'
#
#         # 添加更多手势名称和对应的标签
#     }
#     # 使用字典的get方法，如果prediction不存在，则返回'default'对应的路径
#     name = label_to_gesture_name.get(prediction, 'unkbown')
#     print(name)
def show_name_based_on_prediction(prediction_ges, prediction_force):
    label_to_gesture_name = {
        '0': '握拳',
        '1': '握水杯',
        '2': '伸开五指',
        '3': '握笔',
        '4': '捏卡',
        '5': '竖大拇指',
        'a': '小力度',
        'b': '中等力度',
        'c': '大力度'

        # 添加更多手势名称和对应的标签
    }
    # 使用字典的get方法，如果prediction不存在，则返回'default'对应的路径
    name_ges = label_to_gesture_name.get(prediction_ges, 'unknown')
    name_force = label_to_gesture_name.get(prediction_force, 'unknown')
    print(name_ges, name_force)
def emg_callback(data):
    global emg_data_buffer,new_data_count
    new_data_count += 1
    # 将新的数据添加到缓冲区
    emg_data_buffer.append(data.data)
    # print(type(emg_data_buffer))

    # 保持缓冲区大小为150，如果超过了，就移除最早的数据

    # 当缓冲区中有150条数据时，处理这些数据
    if new_data_count >= 10 and len(emg_data_buffer) == 150:
        # 在这里执行你的特征提取逻辑，例如：
        feature_vector = np.array(extract_features(list(emg_data_buffer))).reshape(1, -1)
        feature_force = force_scaler.transform(feature_vector)
        feature_gesture = gesture_scaler.transform(feature_vector)
        # print(feature_vector[0].shape)
        # probabilities = model_rf.predict_proba(feature_vector)
        # print(probabilities)
        prob_gesture = gesture_rf.predict_proba(feature_gesture)[0]
        # print(prob_gesture)
        max_prob = max(prob_gesture)
        if max_prob < threshold:
            # 如果最高置信度低于阈值，则标记为null
            prediction_gesture= 100
        else:
            # 否则，选择置信度最高的手势

            prediction_gesture=np.argmax(prob_gesture)
        # prediction_gesture = gesture_rf.predict(feature_gesture)[0]
        prediction_force = force_rf.predict(feature_force)[0]
        # print(prediction_gesture)
        show_name_based_on_prediction(str(int(prediction_gesture)),prediction_force)

        # rospy.loginfo("Extracted features: %s", feature_vector)
        new_data_count = 0

        # 注意：这里不清空emg_data_buffer，以保持最新的150条数据
def extract_features(data_buffer):
    feature = []
    # 使用numpy计算一些基本的统计特征作为例子
    data_np = np.array(data_buffer).astype(np.float32)[:,:-1]#最后一列不取
    # # 规范化数据
    # for i in range(data_np.shape[1]):  # 遍历所有通道
    #     range_val = max_vals[i] - min_vals[i]
    #     if range_val != 0:
    #         data_np[:, i] = (data_np[:, i] - min_vals[i]) / range_val
    #     else:
    #         data_np[:, i] = 0  # 如果最大值等于最小值，设为0
    col_means = data_np.mean(axis=0)
    # 从每一列中减去其平均值去除直流偏置
    data = data_np - col_means
    data = data.T
    # print(data.shape)
    window_size = 50  # 滑动窗口大小
    window_num_in_raw = 3  # 每个样本有几个窗口
    raw_length = window_num_in_raw * window_size  # 每个样本有几条数据
    for k in range(window_num_in_raw):  # 每个raw有多少个窗口
        window_start = k * window_size  # 每个窗口起始点
        window_end = (k + 1) * window_size  # 每个窗口结束点
        for i in range(0, 6):  # 遍历六个通道
            temp = data[i][window_start:window_end]
            f1 = np.std(temp)  # 计算窗口数据的标准差，反映了数据的变异程度
            feature.append(f1)
            feature.append(feature_fn.get_rms(temp))  # 计算窗口数据的均方根值
            feature.append(feature_fn.sampEn(temp, std=f1))
            feature.append(feature_fn.Mav(temp))  # 平均值
            feature.append(feature_fn.waveLength(temp))  # 波长
            feature.append(feature_fn.mf(temp))
            feature.append(feature_fn.mpf(temp))
    # print(np.array(feature).shape)
    return feature
    # 根据需要添加更多特征计算
def main():
    # 初始化ROS节点
    rospy.init_node('emg_subscriber', anonymous=True)

    # 创建一个订阅者，订阅名为 "emg_left_arm" 的主题
    rospy.Subscriber("emg_left_arm", emg, emg_callback)

    # 保持程序运行，直到节点被关闭
    rospy.spin()


if __name__ == '__main__':
    main()
