import tensorflow as tf
import pandas as pd
from numpy import *

stage1_epochs = 50  # stage1训练次数
stage2_epochs = 3
# stage2训练次数

# 读入数据
in_filename = 'SDData10.csv'
data = pd.read_csv(in_filename)
# 输出数据
out_filename = "result.csv"

j = 4
data = data[:j * 100000]                    # 对数据集进行切片，截取前j十w数据
data_count = j * 100000                     # 数据量

with open(out_filename, 'a') as file_object:
    file_object.write(str(j) + "00000, ")

stage1_predict = []                         # 存放第一次预测位置
stage2_predict = []                         # 存放第二次预测位置
v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10 = [], [], [], [], [], [], [], [], [], [], []     # 存放第一次分类后的值
p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = [], [], [], [], [], [], [], [], [], [], []     # 存放第一次分类后的原位置

stage1_loss = []                            # stage1误差
stage2_loss = []                            # stage2误差
stage1_sum_loss = 0                                # 误差和
stage1_max_loss = 0                                # 最大误差
stage1_min_loss = 1000000                          # 最小误差
stage2_sum_loss = 0                                # 误差和
stage2_max_loss = 0                                # 最大误差
stage2_min_loss = 1000000                          # 最小误差

value = data.value
position = data.position

# stage1
# 双隐层神经网络
model = tf.keras.Sequential([tf.keras.layers.Dense(32,input_shape=(1,),activation='relu'),
                             tf.keras.layers.Dense(32,activation='relu'),
                             tf.keras.layers.Dense(1)]
)

# 训练数据
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
model.fit(value, position, epochs=stage1_epochs)

# 预测数据
stage1_predict = model.predict(value)

for i in range(data_count):
    stage1_loss.append(abs(stage1_predict[i] - position[i]))

stage1_max_loss = max(stage1_loss)
stage1_min_loss = min(stage1_loss)
stage1_avg_loss = sum(stage1_loss) / data_count

with open(out_filename, 'a') as file_object:
    file_object.write(str(stage1_epochs) + ", ")
    file_object.write(str("{:.2f}".format(stage1_max_loss[0]))+", ")
    file_object.write(str("{:.2f}".format(stage1_min_loss[0]))+", ")
    file_object.write(str("{:.2f}".format(stage1_avg_loss))+", ")

# stage2

# 根据范围将各个数据放进各个数组中
for i in range(data_count):
    # row = (int)(predict_position[i] / (10000 * j))
    # values[row].append(value[i])
    # positions[row].append(position[i])
    if stage1_predict[i] < data_count / 10 * 1:
        v0.append(value[i])
        p0.append(position[i])
    elif data_count / 10 * 1 <= stage1_predict[i] < data_count / 10 * 2:
        v1.append(value[i])
        p1.append(position[i])
    elif data_count / 10 * 2 <= stage1_predict[i] < data_count / 10 * 3:
        v2.append(value[i])
        p2.append(position[i])
    elif data_count / 10 * 3 <= stage1_predict[i] < data_count / 10 * 4:
        v3.append(value[i])
        p3.append(position[i])
    elif data_count / 10 * 4 <= stage1_predict[i] < data_count / 10 * 5:
        v4.append(value[i])
        p4.append(position[i])
    elif data_count / 10 * 5 <= stage1_predict[i] < data_count / 10 * 6:
        v5.append(value[i])
        p5.append(position[i])
    elif data_count / 10 * 6 <= stage1_predict[i] < data_count / 10 * 7:
        v6.append(value[i])
        p6.append(position[i])
    elif data_count / 10 * 7 <= stage1_predict[i] < data_count / 10 * 8:
        v7.append(value[i])
        p7.append(position[i])
    elif data_count / 10 * 8 <= stage1_predict[i] < data_count / 10 * 9:
        v8.append(value[i])
        p8.append(position[i])
    elif data_count / 10 * 9 <= stage1_predict[i] < data_count / 10 * 10:
        v9.append(value[i])
        p9.append(position[i])
    elif data_count / 10 * 10 <= stage1_predict[i]:
        v10.append(value[i])
        p10.append(position[i])

v = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]
p = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]

# 对于每一个模型进行操作
for val, pos in zip(v, p):
    if not val:
        continue
    # 双隐层神经网络
    model = tf.keras.Sequential([tf.keras.layers.Dense(32, input_shape=(1,), activation='relu'),
                                 tf.keras.layers.Dense(32, activation='relu'),
                                 tf.keras.layers.Dense(1)])
    # model.summary()

    # 训练数据
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
    val = list(map(int, val))
    pos = list(map(int, pos))
    model.fit(val, pos, epochs=stage2_epochs)
    stage2_predict = model.predict(val)
    # print(predict_positions)
    # print(positions)
    for i in range(0, len(stage2_predict)):
        stage2_loss.append(abs(stage2_predict[i] - pos[i]))

stage2_max_loss = max(stage2_loss)
stage2_min_loss = min(stage2_loss)
stage2_avg_loss = sum(stage2_loss) / data_count

with open(out_filename, 'a') as file_object:
    file_object.write(str(stage2_epochs) + ", ")
    file_object.write(str("{:.2f}".format(stage2_max_loss[0]))+", ")
    file_object.write(str("{:.2f}".format(stage2_min_loss[0]))+", ")
    file_object.write(str("{:.2f}".format(stage2_avg_loss))+"\n")
