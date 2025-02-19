import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import model  # 这里的 model 模块中应包含 build_model() 函数，用于构建模型
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 使用黑体显示中文
matplotlib.rcParams["axes.unicode_minus"] = False    # 正确显示负号


def get_one_image(image_path):
    """
    读取并展示指定路径的图片，返回 numpy 数组格式的图片数据
    """
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title("原始图片")
    plt.axis('off')
    plt.show()
    return np.array(img)


def evaluate_one_image(image_array):
    """
    基于 TensorFlow 2.x 进行单张图片预测：
      1. 对图片做预处理（类型转换、标准化、调整尺寸）
      2. 构建模型并加载训练好的权重（若检查点不存在，则保存初始权重作为检查点）
      3. 执行前向推理，输出各类别的预测概率
      4. 返回预测结果字符串
    """
    BATCH_SIZE = 1
    N_CLASSES = 4
    IMG_SIZE = 64  # 假设训练时图片大小为 64x64

    # 预处理图片：转换数据类型、调整大小、标准化，并 reshape 为模型输入形状 [1, 64, 64, 3]
    image = tf.cast(image_array, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3])

    # 构建模型（请确保 model.build_model() 返回 tf.keras.Model 对象）
    cnn_model = model.build_model(IMG_SIZE, IMG_SIZE, N_CLASSES)

    # 定义检查点保存目录
    logs_train_dir = 'D:/Neuron/Bsdata/flower/save/'
    # 尝试查找最新的检查点文件
    latest_checkpoint = tf.train.latest_checkpoint(logs_train_dir)
    if latest_checkpoint:
        # 若找到检查点，则加载权重
        cnn_model.load_weights(latest_checkpoint)
        print("加载成功：", latest_checkpoint)
    else:
        # 若没有找到检查点，则保存当前（初始）权重作为检查点文件
        print("没有找到检查点文件，保存初始权重作为检查点")
        if not os.path.exists(logs_train_dir):
            os.makedirs(logs_train_dir)
        # 指定初始检查点文件保存路径及名称
        initial_ckpt_path = os.path.join(logs_train_dir, "model_initial.ckpt")
        cnn_model.save_weights(initial_ckpt_path)
        print("已保存初始权重到：", initial_ckpt_path)
        # 加载刚刚保存的初始权重
        cnn_model.load_weights(initial_ckpt_path)

    # 进行预测（前向推理）
    logits = cnn_model(image, training=False)
    prediction = tf.nn.softmax(logits).numpy()

    max_index = np.argmax(prediction)
    # 根据预测结果输出对应信息
    if max_index == 0:
        result = ('这是玫瑰花的可能性为： %.6f' % prediction[0, 0])
    elif max_index == 1:
        result = ('这是郁金香的可能性为： %.6f' % prediction[0, 1])
    elif max_index == 2:
        result = ('这是蒲公英的可能性为： %.6f' % prediction[0, 2])
    else:
        result = ('这是向日葵的可能性为： %.6f' % prediction[0, 3])

    return result



if __name__ == '__main__':
    # 指定测试图片路径，可根据需要修改
    image_path = r"C:\Users\c2543\Desktop\th-3342646652.jpg"

    # 读取并展示图片
    image_array = get_one_image(image_path)
    # 调整图片大小为模型输入大小（如果图片大小不为 64x64，则调整）
    img = Image.fromarray(image_array)
    img_resized = img.resize((64, 64))
    image_array = np.array(img_resized)

    # 评估图片并打印预测结果
    result = evaluate_one_image(image_array)
    print(result)

