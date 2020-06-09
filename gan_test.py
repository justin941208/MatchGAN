from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import models
import traceback
import numpy as np
import imlib as im
import tflib as tl
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

from glob import glob

def ToOnehot(labels, att_dim):
    batch_size = labels.shape[0]
    out = np.zeros([batch_size, att_dim])
    out[np.arange(batch_size), labels] = 1
    return out

def mean_accuracy_multi_binary_label_with_logits(label, logits):
    return tf.reduce_mean(tf.to_float(tf.equal(tf.to_int64(tf.greater(logits, 0.0)), label)),
                          axis=0, keep_dims=True)

def mean_accuracy_one_hot_label_with_logits(att, logits):
    return tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, axis=1), tf.argmax(att, axis=1))))

def test_train_on_fake(dataset, c_dim, result_dir, gpu_id, epoch_=200):
    img_size = 128

    ''' data '''
    if dataset == 'CelebA':
        ckpt_file = 'checkpoints_train_on_fake/Epoch_({})_(2513of2513).ckpt'.format(epoch_-1)
        test_tfrecord_path = './tfrecords_test/celeba_tfrecord_test'
        test_data_pool = tl.TfrecordData(test_tfrecord_path, 18, shuffle=False)
    elif dataset == 'RaFD':
        ckpt_file = 'checkpoints_train_on_fake/Epoch_({})_(112of112).ckpt'.format(epoch_-1)
        test_tfrecord_path = './tfrecords_test/rafd_test'
        test_data_pool = tl.TfrecordData(test_tfrecord_path, 120, shuffle=False)
    ckpt_file = os.path.join(result_dir, ckpt_file)

    """ graphs """
    with tf.device('/gpu:{}'.format(gpu_id)):
        ''' models '''
        classifier = models.classifier

        ''' graph '''
        # inputs
        x_255 = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
        x = x_255 / 127.5 - 1
        if dataset == 'CelebA':
            label = tf.placeholder(tf.int64, shape=[None, c_dim])
        elif dataset == 'RaFD':
            label = tf.placeholder(tf.float32, shape=[None, c_dim])

        # classify
        logits = classifier(x, att_dim=c_dim, reuse=False, training=False)

        if dataset == 'CelebA':
            accuracy = mean_accuracy_multi_binary_label_with_logits(label, logits)
        elif dataset == 'RaFD':
            accuracy = mean_accuracy_one_hot_label_with_logits(label, logits)

    """ train """
    ''' init '''
    # session
    sess = tl.session()

    ''' initialization '''
    tl.load_checkpoint(ckpt_file, sess)

    ''' train '''
    try:
        all_accuracies = []
        denom = 18 if dataset == 'CelebA' else 120
        key = 'class' if dataset == 'CelebA' else 'attr'
        test_iter = len(test_data_pool) // denom
        for iter in range(test_iter):
            img, label_gt = test_data_pool.batch(['img', key])
            if dataset == 'RaFD':
                label_gt = ToOnehot(label_gt, c_dim)
            print('Test batch {}'.format(iter), end='\r')
            batch_accuracy = sess.run(accuracy, feed_dict={x_255: img, label: label_gt})
            all_accuracies.append(batch_accuracy)

        if dataset == 'CelebA':
            mean_accuracies = np.mean(np.concatenate(all_accuracies), axis=0)
            mean_accuracy = np.mean(mean_accuracies)
            print('\nIndividual accuracies: {} Average: {:.4f}'.format(mean_accuracies, mean_accuracy))
            with open(os.path.join(result_dir, 'GAN_train.txt'), 'w') as f:
                for attr, acc in zip(['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'], mean_accuracies):
                    f.write('{}: {}\n'.format(attr, acc))
                f.write('Average: {}'.format(mean_accuracy))
        elif dataset == 'RaFD':
            mean_accuracy = np.mean(all_accuracies)
            print('\nAverage accuracies: {:.4f}'.format(mean_accuracy))
            with open(os.path.join(result_dir, 'GAN_train.txt'), 'w') as f:
                f.write('Average accuracy: {}'.format(mean_accuracy))

    except Exception:
        traceback.print_exc()
    finally:
        print(" [*] Close main session!")
        sess.close()
