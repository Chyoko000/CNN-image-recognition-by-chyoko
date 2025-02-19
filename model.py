import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(img_w, img_h, n_classes):
    """
    构建一个简单的卷积神经网络模型
    参数:
        img_w: 图像宽度
        img_h: 图像高度
        n_classes: 分类数量
    返回:
        model: 一个 tf.keras.Model 对象
    """
    model = models.Sequential([
        # 输入层：指定图片大小及通道数
        layers.InputLayer(input_shape=(img_w, img_h, 3)),
        # 卷积块1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # 卷积块2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        # 展平层
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(n_classes)
    ])
    return model

