from PIL import Image
import matplotlib.pyplot as plt
import Train_inputdata
import Train_model
import os
import numpy as np
import tensorflow as tf


def get_one_image(train):
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]

    image = Image.open(img_dir)
    image = image.resize([227, 227])
    image = np.array(image)
    return image


def evaluate_one_image():
    """
    Test one image against the saved models and parameters
    返回字符串1~5
    """
    train_dir = 'D:/python/deep-learning/Gesture-recognition/data/testImage/roi/'
    train, train_label = Train_inputdata.get_files(train_dir)
    image_array = get_one_image(train)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 5

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 227, 227, 3])
        logit = Train_model.cnn_inference(image, BATCH_SIZE, N_CLASSES, keep_prob=1)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[227, 227, 3])

        logs_train_dir = 'D:/python/deep-learning/Gesture-recognition/log/'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('This is a scissor with possibility %.6f' % prediction[:, 0])
                return 1
            elif max_index == 1:
                print('This is a rock with possibility %.6f' % prediction[:, 1])
                return 2
            elif max_index == 2:
                print('This is a paper with possibility %.6f' % prediction[:, 2])
                return 3
            elif max_index == 3:
                print('This is a ok with possibility %.6f' % prediction[:, 3])
                return 4
            elif max_index == 4:
                print('This is a good with possibility %.6f' % prediction[:, 4])
                return 5






