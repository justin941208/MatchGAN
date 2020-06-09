from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import models
import argparse
import traceback
import numpy as np
import tflib as tl
import tensorflow as tf

def ToOnehot(labels, att_dim):
    batch_size = labels.shape[0]
    out = np.zeros([batch_size, att_dim])
    out[np.arange(batch_size), labels] = 1
    return out

def mean_accuracy_multi_binary_label_with_logits(att, logits):
    return tf.reduce_mean(tf.to_float(tf.equal(tf.to_int64(tf.greater(logits, 0.0)), att)))

def mean_accuracy_one_hot_label_with_logits(att, logits):
    return tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, axis=1), tf.argmax(att, axis=1))))

def train_on_fake(dataset, c_dim, result_dir, gpu_id, use_real=0, epoch_=200):
    """ param """
    batch_size = 64
    batch_size_fake = batch_size
    lr = 0.0002

    ''' data '''
    if use_real == 1:
        print('======Using real data======')
        batch_size_real = batch_size // 2
        batch_size_fake = batch_size - batch_size_real
        train_tfrecord_path_real = './tfrecords/celeba_tfrecord_train'
        train_data_pool_real = tl.TfrecordData(train_tfrecord_path_real, batch_size_real, shuffle=True)
    train_tfrecord_path_fake = os.path.join(result_dir, 'synthetic_tfrecord')
    train_data_pool_fake = tl.TfrecordData(train_tfrecord_path_fake, batch_size_fake, shuffle=True)
    if dataset == 'CelebA':
        test_tfrecord_path = './tfrecords/celeba_tfrecord_test'
    elif dataset == 'RaFD':
        test_tfrecord_path = './tfrecords/rafd_test'
    test_data_pool = tl.TfrecordData(test_tfrecord_path, 120)
    att_dim = c_dim

    """ graphs """
    with tf.device('/gpu:{}'.format(gpu_id)):

        ''' models '''
        classifier = models.classifier

        ''' graph '''
        # inputs
        x_255 = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
        x = x_255 / 127.5 - 1
        if dataset == 'CelebA':
            att = tf.placeholder(tf.int64, shape=[None, att_dim])
        elif dataset == 'RaFD':
            att = tf.placeholder(tf.float32, shape=[None, att_dim])

        # classify
        logits = classifier(x, att_dim=att_dim, reuse=False)

        # loss
        reg_loss = tf.losses.get_regularization_loss()
        if dataset == 'CelebA':
            loss = tf.losses.sigmoid_cross_entropy(att, logits) + reg_loss
            acc = mean_accuracy_multi_binary_label_with_logits(att, logits)
        elif dataset == 'RaFD':
            loss = tf.losses.softmax_cross_entropy(att, logits) + reg_loss
            acc = mean_accuracy_one_hot_label_with_logits(att, logits)

        lr_ = tf.placeholder(tf.float32, shape=[])

        # optim
        #with tf.variable_scope('Adam', reuse=tf.AUTO_REUSE):
        step = tf.train.AdamOptimizer(lr_, beta1=0.9).minimize(loss)

        # test
        test_logits = classifier(x, att_dim=att_dim, training=False)
        if dataset == 'CelebA':
            test_acc = mean_accuracy_multi_binary_label_with_logits(att, test_logits)
        elif dataset == 'RaFD':
            test_acc = mean_accuracy_one_hot_label_with_logits(att, test_logits)
        mean_acc = tf.placeholder(tf.float32, shape=())

    # summary
    summary = tl.summary({loss: 'loss', acc: 'acc'})
    test_summary = tl.summary({mean_acc: 'test_acc'})

    """ train """
    ''' init '''
    # session
    sess = tf.Session()
    # iteration counter
    it_cnt, update_cnt = tl.counter()
    # saver
    saver = tf.train.Saver(max_to_keep=None)
    # summary writer
    sum_dir = os.path.join(result_dir, 'summaries_train_on_fake')
    if use_real == 1:
        sum_dir += '_real'
    summary_writer = tf.summary.FileWriter(sum_dir, sess.graph)

    ''' initialization '''
    ckpt_dir = os.path.join(result_dir, 'checkpoints_train_on_fake')
    if use_real == 1:
        ckpt_dir += '_real'
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir + '/')
    if not tl.load_checkpoint(ckpt_dir, sess):
        sess.run(tf.global_variables_initializer())

    ''' train '''
    try:
        batch_epoch = len(train_data_pool_fake) // batch_size
        max_it = epoch_ * batch_epoch
        for it in range(sess.run(it_cnt), max_it):
            bth = it//batch_epoch - 8
            lr__ = lr*(1-max(bth, 0)/epoch_)**0.75
            if it % batch_epoch == 0:
                print('======learning rate:', lr__, '======')
            sess.run(update_cnt)

            # which epoch
            epoch = it // batch_epoch
            it_epoch = it % batch_epoch + 1

            x_255_ipt, att_ipt = train_data_pool_fake.batch(['img', 'attr'])
            if dataset == 'RaFD':
                att_ipt = ToOnehot(att_ipt, att_dim)
            if use_real == 1:
                x_255_ipt_real, att_ipt_real = train_data_pool_real.batch(['img', 'class'])
                x_255_ipt = np.concatenate([x_255_ipt, x_255_ipt_real])
                att_ipt = np.concatenate([att_ipt, att_ipt_real])
            summary_opt, _ = sess.run([summary, step], feed_dict={x_255: x_255_ipt, att: att_ipt, lr_:lr__})
            summary_writer.add_summary(summary_opt, it)

            # display
            if (it + 1) % batch_epoch == 0:
                print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

            # save
            if (it + 1) % (batch_epoch * 50) == 0:
                save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
                print('Model saved in file: % s' % save_path)

            # sample
            if it % 100 == 0:
                test_it = 100 if dataset == 'CelebA' else 7
                test_acc_opt_list = []
                for i in range(test_it):
                    key = 'class' if dataset == 'CelebA' else 'attr'
                    x_255_ipt, att_ipt = test_data_pool.batch(['img', key])
                    if dataset == 'RaFD':
                        att_ipt = ToOnehot(att_ipt, att_dim)

                    test_acc_opt = sess.run(test_acc, feed_dict={x_255: x_255_ipt, att: att_ipt})
                    test_acc_opt_list.append(test_acc_opt)
                test_summary_opt = sess.run(test_summary, feed_dict={mean_acc: np.mean(test_acc_opt_list)})
                summary_writer.add_summary(test_summary_opt, it)

    except Exception:
        traceback.print_exc()
    finally:
        print(" [*] Close main session!")
        sess.close()
