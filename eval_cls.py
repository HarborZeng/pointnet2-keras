import tensorflow as tf
from keras import backend as K
import numpy as np

from modelnet_h5_dataset import ModelNetH5Dataset
from tf_cls import batch_size, num_point, nb_classes, summary_dir
from model_cls import pointnet2

shuffle = False

train_dataset = ModelNetH5Dataset('./data/modelnet40_ply_hdf5_2048/train_files.txt',
                                  batch_size=batch_size, npoints=num_point, shuffle=shuffle)

test_dataset = ModelNetH5Dataset('data/modelnet40_ply_hdf5_2048/test_files.txt',
                                 batch_size=batch_size, npoints=num_point, shuffle=shuffle)

point_cloud = K.placeholder(dtype=np.float32, shape=(batch_size, num_point, 3))
labels = K.placeholder(dtype=np.int32, shape=batch_size)
is_training_pl = K.placeholder(dtype=np.bool, shape=())

logits = pointnet2(point_cloud, nb_classes, is_training_pl)
top_k_op = tf.nn.in_top_k(logits, labels, 1)

saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(summary_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found')

    cur_batch_data = np.zeros((batch_size, num_point, train_dataset.num_channel()))
    cur_batch_label = np.zeros(batch_size, dtype=np.int32)
    true_count = 0
    total_batch_count = 0

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    while train_dataset.has_next_batch():
        batch_data, batch_label = train_dataset.next_batch()
        bsize = batch_data.shape[0]
        total_batch_count += bsize
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        preditions = sess.run([top_k_op], feed_dict={
            point_cloud: cur_batch_data,
            labels: cur_batch_label,
            is_training_pl: False,
        })
        true_count += np.sum(preditions)

        precision = true_count / total_batch_count
        print('train dataset cumulative precision is {:.4%}'.format(precision))

    while test_dataset.has_next_batch():
        batch_data, batch_label = test_dataset.next_batch()
        bsize = batch_data.shape[0]
        total_batch_count += bsize
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        preditions = sess.run([top_k_op],  feed_dict={
            point_cloud: cur_batch_data,
            labels: cur_batch_label,
            is_training_pl: False,
        })
        true_count += np.sum(preditions)

        precision = true_count / total_batch_count
        print('test dataset cumulative precision is {:.4%}'.format(precision))

    train_dataset.reset()
    test_dataset.reset()
