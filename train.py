import tensorflow as tf
import input_data
import model

# ------------------ 参数设置 ------------------
N_CLASSES = 4         # 花的种类数
IMG_W = 64            # 图像宽度
IMG_H = 64            # 图像高度
BATCH_SIZE = 20       # 每个批次图像数量
CAPACITY = 200        # shuffle 时的缓冲区大小
EPOCHS = 8           # 训练轮数（根据需要调整）
MAX_STEP = 1000        # 每个 epoch 的步数（根据数据集大小调整）
LEARNING_RATE = 0.0001  # 学习率

# 数据和日志路径
TRAIN_DIR = r"D:\Neuron\Bsdata\flower\input_data"  # 训练数据目录
LOGS_TRAIN_DIR = r"D:\Neuron\Bsdata\flower\save"     # 日志及模型保存目录

# ------------------ 数据加载 ------------------
# 获取图片路径和标签，并拆分为训练集与验证集（验证集比例 0.3）
train_images, train_labels, val_images, val_labels = input_data.get_files(TRAIN_DIR, val_ratio=0.3)

# 利用 tf.data 创建数据集，并增加 repeat 与 prefetch 加速数据读取
train_dataset = input_data.get_batch(train_images, train_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
train_dataset = train_dataset.repeat().prefetch(tf.data.AUTOTUNE)

val_dataset = input_data.get_batch(val_images, val_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

# ------------------ 模型构建 ------------------
cnn_model = model.build_model(IMG_W, IMG_H, N_CLASSES)

# 编译模型：使用 Adam 优化器、稀疏分类交叉熵损失（标签为整数）、以及准确率指标
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

# ------------------ 回调函数 ------------------
# TensorBoard 用于可视化训练过程
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGS_TRAIN_DIR, histogram_freq=1)
# ModelCheckpoint 保存验证集上最优模型权重
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{LOGS_TRAIN_DIR}/model_epoch{{epoch:02d}}_valacc{{val_accuracy:.2f}}.h5",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)
# EarlyStopping 提前停止训练，防止过拟合（监控验证损失，patience 可根据情况调整）
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

callbacks_list = [tensorboard_callback, model_checkpoint_callback, early_stopping_callback]

# ------------------ 模型训练 ------------------
cnn_model.fit(train_dataset,
              epochs=EPOCHS,
              steps_per_epoch=MAX_STEP,
              validation_data=val_dataset,
              callbacks=callbacks_list)


