import Train_inputdata
import Train_model
import tensorflow as tf
import numpy as np

with tf.Graph().as_default():  # 不加的话graph就会冲突
    IMG_W = 227
    IMG_H = 227
    BATCH_SIZE = 500
    CAPACITY = 2000
    N_CLASSES = 5

    test_dir = 'D:/python/deep-learning/Gesture-recognition/data/test/'

    test, test_label = Train_inputdata.get_files(test_dir)
    test_batch, test_label_batch = Train_inputdata.get_batch(test,
                                                             test_label,
                                                             IMG_W,
                                                             IMG_H,
                                                             BATCH_SIZE,
                                                             CAPACITY)

    test_logit = Train_model.cnn_inference(test_batch, BATCH_SIZE, N_CLASSES, keep_prob=1)
    test_acc = Train_model.evaluation(test_logit, test_label_batch)
    test_loss = Train_model.losses(test_logit, test_label_batch)

    logs_train_dir = 'D:/python/deep-learning/Gesture-recognition/log/'

    saver = tf.train.Saver()

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)

        # 对最新的模型进行测验
        # if ckpt and ckpt.model_checkpoint_path:
        #     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        #     print('Loading success, global_step is %s' % global_step)
        # else:
        #     print('No checkpoint file found')

        # 对所有保存的模型进行测验
        if ckpt and ckpt.all_model_checkpoint_paths:
            for path in ckpt.all_model_checkpoint_paths:
                saver.restore(sess, path)
                global_step = path.split('/')[-1].split('-')[-1]
                print('Loading success, global_step is %s' % global_step)
                accuracy, loss = sess.run([test_acc, test_loss])
                print("测试集正确率是：%.2f%%" % (accuracy * 100))
                print("测试集损失率：%.2f" % loss)
        else:
            print('No checkpoint file found')




