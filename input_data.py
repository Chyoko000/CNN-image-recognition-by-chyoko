import os
import math
import numpy as np
import tensorflow as tf


def get_files(file_dir, val_ratio=0.3):
    """
    获取图片文件路径和标签，并按验证集比例拆分
    参数:
        file_dir: 数据根目录，每个类别存放在一个子文件夹中（如 'roses', 'tulips', 等）
        val_ratio: 验证集占比（例如 0.3 表示 30% 用于验证）
    返回:
        train_images, train_labels, val_images, val_labels
    """
    # 定义类别及其对应标签
    categories = ['roses', 'tulips', 'dandelion', 'sunflowers']
    image_list = []
    label_list = []

    # 遍历每个类别文件夹，收集图片路径和标签
    for i, category in enumerate(categories):
        category_path = os.path.join(file_dir, category)
        files = os.listdir(category_path)
        for file in files:
            image_list.append(os.path.join(category_path, file))
            label_list.append(i)

    # 打乱数据顺序
    data = list(zip(image_list, label_list))
    np.random.shuffle(data)
    image_list, label_list = zip(*data)
    image_list = list(image_list)
    label_list = list(label_list)

    # 按比例拆分训练集和验证集
    n_samples = len(label_list)
    n_val = int(math.ceil(n_samples * val_ratio))
    n_train = n_samples - n_val
    train_images = image_list[:n_train]
    train_labels = label_list[:n_train]
    val_images = image_list[n_train:]
    val_labels = label_list[n_train:]

    return train_images, train_labels, val_images, val_labels


def get_batch(image_list, label_list, img_w, img_h, batch_size, capacity):
    """
    利用 tf.data 创建数据集，自动读取、解码和预处理图片
    参数:
        image_list: 图片路径列表
        label_list: 对应标签列表
        img_w: 图片宽度
        img_h: 图片高度
        batch_size: 每个批次的样本数
        capacity: shuffle 缓冲区大小
    返回:
        dataset: 一个 tf.data.Dataset 对象，每个元素为 (image, label)
    """
    # 通过 from_tensor_slices 构建数据集
    dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))

    def _parse_function(filename, label):
        # 读取文件内容
        image_string = tf.io.read_file(filename)
        # 解码 JPEG 图片（确保所有图片都是 JPEG 格式）
        image = tf.image.decode_jpeg(image_string, channels=3)
        # 调整图片大小
        image = tf.image.resize(image, [img_w, img_h])
        # 将像素值归一化到 [0,1]
        image = image / 255.0
        return image, label

    # 映射预处理函数，shuffle、batch 以及预取数据以加速训练
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=capacity)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


