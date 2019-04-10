import tensorflow as tf


def cnn_inference(images, batch_size, n_classes, keep_prob):

    """
    使用AlexNet结构
    输入
    images      输入的图像
    batch_size  每个批次的大小
    n_classes   n分类
    keep_prob   droupout保存的比例（ 设置神经元被选中的概率）
    返回
    softmax_linear 还差一个softmax
    """
    # 第一层的卷积层conv1，卷积核为3X3，有16个
    with tf.variable_scope('conv1') as scope:
        # 建立weights和biases的共享变量
        # conv1, shape = [kernel size, kernel size, channels, kernel numbers]
        weights = tf.get_variable('weights',
                                  shape=[11, 11, 3, 96],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))  # stddev标准差
        biases = tf.get_variable('biases',
                                 shape=[96],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # 卷积层 strides = [1, x_movement, y_movement, 1], padding填充周围有valid和same可选择
        conv = tf.nn.conv2d(images, weights, strides=[1, 4, 4, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)         # 加入偏差
        conv1 = tf.nn.relu(pre_activation, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间

    # 第一层的池化层pool1和规范化norm1(特征缩放）
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm1')
        # ksize是池化窗口的大小=[1,height,width,1]，一般height=width=池化窗口的步长
        # 池化窗口的步长一般是比卷积核多移动一位
        # tf.nn.lrn是Local Response Normalization，（局部响应归一化）

    # 第二层的卷积层cov2，这里的命名空间和第一层不一样，所以可以和第一层取同名
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5, 5, 96, 256],  # 这里只有第三位数字96需要等于上一层的tensor维度
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # 第二层的池化层pool2和规范化norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID',name='pooling2')
        # 这里选择了先规范化再池化

    # conv3
    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 256, 384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name='conv3')

    # conv4
    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 384, 384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name='conv4')

    # conv5
    with tf.variable_scope('conv5') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 384, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv4, weights, strides=[1, 1, 1, 1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name='conv5')

    # 池化
    with tf.variable_scope('pooling') as scope:
        pooling = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                 padding='VALID', name='pooling1')

    # 第三层为全连接层local3
    with tf.variable_scope('local3') as scope:
        # flatten-把卷积过的多维tensor拉平成二维张量（矩阵）
        reshape = tf.reshape(pooling, shape=[batch_size, -1])  # batch_size表明了有多少个样本

        dim = reshape.get_shape()[1].value  # 知道-1(代表任意)这里具体是多少个
        weights = tf.get_variable('weights',
                                  shape=[dim, 1024],  # 连接1024个神经元
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[1024],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)  # 矩阵相乘加上bias
        local3 = tf.nn.dropout(local3, keep_prob)  # 设置神经元被选中的概率

    # 第四层为全连接层local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[1024, 1024],  # 再连接1024个神经元
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[1024],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
        local4 = tf.nn.dropout(local4, keep_prob)

    # 第五层为输出层softmax_linear
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights',
                                  shape=[1024, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
        # 这里只是命名为softmax_linear，真正的softmax函数放在下面的losses函数里面和交叉熵结合在一起了，这样可以提高运算速度。
        # softmax_linear的行数=local4的行数，列数=weights的列数=bias的行数=需要分类的个数
        # 经过softmax函数用于分类过程中，它将多个神经元的输出，映射到（0,1）区间内，可以看成概率来理解
        softmax_linear = tf.nn.dropout(softmax_linear, keep_prob)

    return softmax_linear


def losses(logits, labels):
    """
    输入
    logits: 经过cnn_inference处理过的tensor
    labels: 对应的标签
    返回
    loss： 损失函数（交叉熵）
    """
    with tf.variable_scope('loss') as scope:
        # 下面把交叉熵和softmax合到一起写是为了通过spares提高计算速度
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='loss_per_eg')
        loss = tf.reduce_mean(cross_entropy, name='loss')  # 求所有样本的平均loss
    return loss


def training(loss, learning_rate):
    """
    输入
    loss: 损失函数（交叉熵）
    learning_rate： 学习率
    返回
    train_op: 训练的最优值
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # global_step不是共享变量，初始值为0，设定trainable=False 可以防止该变量被数据流图的 GraphKeys.TRAINABLE_VARIABLES 收集,
        # 这样我们就不会在训练的时候尝试更新它的值。
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op


def evaluation(logits, labels):
    """
     输入
    logits: 经过cnn_inference处理过的tensor
    labels:
    返回
    accuracy：正确率
    """
    with tf.variable_scope('accuracy') as scope:
        prediction = tf.nn.softmax(logits)  # 这个logits有n_classes列
        # prediction每行的最大元素（1）的索引和label的值相同则为1 否则为0。
        correct = tf.nn.in_top_k(prediction, labels, 1)
        # correct = tf.nn.in_top_k(logits, labels, 1)   也可以不需要prediction过渡，因为最大值的索引没变，这里这样写是为了更好理解
        correct = tf.cast(correct, tf.float16)  # 记得要转换格式
        accuracy = tf.reduce_mean(correct)
    return accuracy
