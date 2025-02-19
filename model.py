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
        # 卷积层1：64 个 3x3 卷积核，ReLU 激活
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # 卷积层2：16 个 3x3 卷积核，ReLU 激活
        layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # 展平层
        layers.Flatten(),
        # 全连接层1：128 个神经元
        layers.Dense(128, activation='relu'),
        # 全连接层2：128 个神经元
        layers.Dense(128, activation='relu'),
        # 输出层：不使用 softmax（配合 loss 中的 from_logits=True）
        layers.Dense(n_classes)
    ])
    return model

