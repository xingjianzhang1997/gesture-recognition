import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import Train_model
import Train_inputdata


N_CLASSES = 5  # 5种手势
IMG_W = 227  # resize图像，太大的话训练时间久
IMG_H = 227
BATCH_SIZE = 32  # 由于数据集较小，所以选值较大  一次训练512张已经是该电脑的极限了
CAPACITY = 320
MAX_STEP = 1000
learning_rate = 0.0001  # 一般小于0.0001

train_dir = 'D:/python/deep-learning/Gesture-recognition/data/train/'
logs_train_dir = 'D:/python/deep-learning/Gesture-recognition/log/'  # 记录训练过程与保存模型

train, train_label = Train_inputdata.get_files(train_dir)
train_batch, train_label_batch = Train_inputdata.get_batch(train,
                                                           train_label,
                                                           IMG_W,
                                                           IMG_H,
                                                           BATCH_SIZE,
                                                           CAPACITY)

train_logits = Train_model.cnn_inference(train_batch, BATCH_SIZE, N_CLASSES, keep_prob=0.5)
train_loss = Train_model.losses(train_logits, train_label_batch)
train_op = Train_model.training(train_loss, learning_rate)
train__acc = Train_model.evaluation(train_logits, train_label_batch)


summary_op = tf.summary.merge_all()  # 这个是log汇总记录

# 可视化为了画折线图
step_list = list(range(50))  # 因为后来的cnn_list有20个
cnn_list1 = []   # tra_acc
cnn_list2 = []   # tra_loss

fig = plt.figure()  # 建立可视化图像框
ax = fig.add_subplot(2, 1, 1)  # 子图总行数、列数，位置
ax.yaxis.grid(True)
ax.set_title('cnn_accuracy ', fontsize=14, y=1.02)
ax.set_xlabel('step')
ax.set_ylabel('accuracy')
bx = fig.add_subplot(2, 1, 2)
bx.yaxis.grid(True)
bx.set_title('cnn_loss ', fontsize=14, y=1.02)
bx.set_xlabel('step')
bx.set_ylabel('loss')


# 初始化，如果存在变量则是必不可少的操作
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 产生一个writer来写log文件
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    # 产生一个saver来存储训练好的模型
    saver = tf.train.Saver()

    # 队列监控
    # batch训练法用到了队列，不想用队列也可以用placeholder
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        # 执行MAX_STEP步的训练，一步一个batch
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            # 启动以下操作节点，这里不能用train_op，因为它在第二次迭代是None，会导致session出错，改为_
            _op, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            # tes_acc, tes_loss = sess.run([test_acc, test_loss])
            # 每隔20步打印一次当前的loss以及acc，同时记录log，写入writer,并画个图
            if step % 20 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
                cnn_list1.append(tra_acc)
                cnn_list2.append(tra_loss)
            # 每隔40步，保存一次训练好的模型
            if step % 60 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        ax.plot(step_list, cnn_list1, color="r", label=train)
        bx.plot(step_list, cnn_list2, color="r", label=train)

        plt.tight_layout()
        plt.show()

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()



