import os
import numpy as np
import tensorflow as tf
import input_data
import model

# ------------------ 参数设置 ------------------
N_CLASSES = 4         # 花的种类数
IMG_W = 64            # 图像宽度
IMG_H = 64            # 图像高度
BATCH_SIZE = 20       # 每个批次图像数量
CAPACITY = 200        # shuffle 时的缓冲区大小
EPOCHS = 20            # 训练轮数（根据需要调整）
MAX_STEP = 200        # 每个 epoch 的步数（根据数据集大小调整）
learning_rate = 0.0001  # 学习率

# 数据和日志路径
train_dir = r"D:\Neuron\Bsdata\flower\input_data"  # 训练数据目录
logs_train_dir = r"D:\Neuron\Bsdata\flower\save"     # 日志及模型保存目录

# ------------------ 数据加载 ------------------
# 获取图片路径和标签，并拆分为训练集与验证集（验证集比例 0.3）
train_images, train_labels, val_images, val_labels = input_data.get_files(train_dir, val_ratio=0.3)

# 利用 tf.data 创建数据集
train_dataset = input_data.get_batch(train_images, train_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
# 添加 .repeat()，确保训练数据集可以无限生成批次
train_dataset = train_dataset.repeat()

val_dataset = input_data.get_batch(val_images, val_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
# 如果验证数据也不足以支撑验证步数，可以选择 repeat()，否则一般不加

# ------------------ 模型构建 ------------------
cnn_model = model.build_model(IMG_W, IMG_H, N_CLASSES)

# 编译模型：使用 Adam 优化器、稀疏分类交叉熵损失（标签为整数）、以及准确率指标
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

# ------------------ TensorBoard 回调 ------------------
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_train_dir, histogram_freq=1)

# ------------------ 模型训练 ------------------
# 注意：如果 MAX_STEP 过大而数据量不足，则会出现数据用尽的警告
cnn_model.fit(train_dataset,
              epochs=EPOCHS,
              steps_per_epoch=MAX_STEP,
              validation_data=val_dataset,
              callbacks=[tensorboard_callback])

