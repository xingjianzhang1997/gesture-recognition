import tensorflow as tf
import numpy as np
import os


def get_files(file_dir):
    """
    输入： 存放训练照片的文件地址
    返回:  图像列表， 标签列表
    """
    # 建立空列表
    gesture_1 = []          # 剪刀
    label_gesture_1 = []
    gesture_2 = []          # 石头
    label_gesture_2 = []
    gesture_3 = []          # 布
    label_gesture_3 = []
    gesture_4 = []          # OK
    label_gesture_4 = []
    gesture_5 = []          # good
    label_gesture_5 = []

    # 读取标记好的图像和加入标签
    for file in os.listdir(file_dir):   # file就是要读取的照片
        name = file.split(sep='.')      # 因为照片的格式是gesture_1.1.jpg/gesture_1.2.jpg
        if name[0] == 'gesture_1':            # 所以只用读取 . 前面这个字符串
            gesture_1.append(file_dir + file)
            label_gesture_1.append(0)        # 把图像和标签加入列表
        elif name[0] == 'gesture_2':
            gesture_2.append(file_dir + file)
            label_gesture_2.append(1)
        elif name[0] == 'gesture_3':
            gesture_3.append(file_dir + file)
            label_gesture_3.append(2)
        elif name[0] == 'gesture_4':
            gesture_4.append(file_dir + file)
            label_gesture_4.append(3)
        else:
            gesture_5.append(file_dir + file)
            label_gesture_5.append(4)

    print('There are %d scissors\nThere are %d paper\nThere are %d rock\nThere are %d ok\nThere are %d good'
          % (len(gesture_1), len(gesture_3), len(gesture_2), len(gesture_4), len(gesture_5)))

    image_list = np.hstack((gesture_1, gesture_2, gesture_3, gesture_4, gesture_5))  # 在水平方向平铺合成一个行向量
    label_list = np.hstack((label_gesture_1, label_gesture_2, label_gesture_3, label_gesture_4, label_gesture_5))

    temp = np.array([image_list, label_list])  # 生成一个两行数组列表，大小是2 X 2500
    temp = temp.transpose()   # 转置向量，大小变成2500 X 2
    np.random.shuffle(temp)   # 乱序，打乱这2500个例子的顺序，一个乱序的效果不太明显
    np.random.shuffle(temp)
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])  # 所有行，列=0
    label_list = list(temp[:, 1])  # 所有行，列=1
    label_list = [int(float(i)) for i in label_list]  # 把标签列表转化为int类型

    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    输入：
    image,label ：要生成batch的图像和标签
    image_W，image_H: 图像的宽度和高度
    batch_size: 每个batch（小批次）有多少张图片数据
    capacity: 队列的最大容量
    返回：
    image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
    label_batch: 1D tensor [batch_size], dtype=tf.int32
    """
    # 将列表转换成tf能够识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 生成队列(牵扯到线程概念，便于batch训练）
    """
    队列的理解：每次训练时，从队列中取一个batch送到网络进行训练，
               然后又有新的图片从训练库中注入队列，这样循环往复。
               队列相当于起到了训练库到网络模型间数据管道的作用，
               训练数据通过队列送入网络。
    """
    input_queue = tf.train.slice_input_producer([image, label])

    # 图像的读取需要tf.read_file()，标签则可以直接赋值
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)  # 解码彩色的.jpg图像
    label = input_queue[1]

    # 统一图片大小
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)  # 标准化图片，因为前两行代码已经处理过了，所以可要可不要

    # 打包batch的大小
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,  # 涉及到线程，配合队列
                                              capacity=capacity)

    # 下面两行代码应该也多余了，放在这里确保一下格式不会出问题
    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.cast(label_batch, tf.int32)

    return image_batch, label_batch
