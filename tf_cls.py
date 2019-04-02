import tensorflow as tf
from model_cls import pointnet2
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
from data_loader import DataGenerator
import os
from keras import backend as K
from modelnet_h5_dataset import ModelNetH5Dataset
import numpy as np


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()


def get_learning_rate():
    return 0.001


def train():
    nb_classes = 40

    train_file = './ModelNet40/ply_data_train.h5'
    test_file = './ModelNet40/ply_data_test.h5'

    epochs = 100
    batch_size = 32
    num_point = 1024

    TRAIN_DATASET = ModelNetH5Dataset('./data/modelnet40_ply_hdf5_2048/train_files.txt',
                                      batch_size=batch_size, npoints=num_point, shuffle=True)
    TEST_DATASET = ModelNetH5Dataset('data/modelnet40_ply_hdf5_2048/test_files.txt',
                                     batch_size=batch_size, npoints=num_point, shuffle=False)

    train_dg = DataGenerator(train_file, batch_size, nb_classes, train=True)
    validate_dg = DataGenerator(test_file, batch_size, nb_classes, train=False)
    point_cloud = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pred = pointnet2(point_cloud, nb_classes)
    labels = tf.placeholder(tf.float32, shape=(nb_classes))
    from keras.objectives import categorical_crossentropy
    loss = tf.reduce_mean(categorical_crossentropy(labels, pred))
    train_op = tf.train.AdamOptimizer(get_learning_rate()).minimize(loss)
    init_op = tf.global_variables_initializer()
    session = K.get_session()
    session.run(init_op)

    with session.as_default():
        with tf.device('/gpu:0'):
            for i in range(epochs):
                # Make sure batch data is of same size
                cur_batch_data = np.zeros((batch_size, num_point, TRAIN_DATASET.num_channel()))
                cur_batch_label = np.zeros((batch_size), dtype=np.int32)
                while TRAIN_DATASET.has_next_batch():
                    batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True)
                    bsize = batch_data.shape[0]
                    cur_batch_data[0:bsize, ...] = batch_data
                    cur_batch_label[0:bsize] = batch_label
                    train_op.run(feed_dict={
                        point_cloud: cur_batch_data,
                        labels: cur_batch_label
                    })


if __name__ == '__main__':
    train()
