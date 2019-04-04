import tensorflow as tf
from model_cls import pointnet2
import matplotlib

from pointnet2_cls_msg import placeholder_inputs, get_loss, get_model

matplotlib.use('AGG')
import matplotlib.pyplot as plt
from data_loader import DataGenerator
import os
from keras import backend as K
from modelnet_h5_dataset import ModelNetH5Dataset
import numpy as np

nb_classes = 40

train_file = './ModelNet40/ply_data_train.h5'
test_file = './ModelNet40/ply_data_test.h5'

epochs = 150
batch_size = 16
num_point = 1024

EPOCH_CNT = 0

DECAY_STEP = 200000
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
DECAY_RATE = 0.7


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


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(0.001,  # Base learning rate.
                                               batch * batch_size,  # Current index into the dataset.
                                               DECAY_STEP,  # Decay step.
                                               DECAY_RATE,  # Decay rate.
                                               staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,
                                             batch * batch_size,
                                             BN_DECAY_DECAY_STEP,
                                             BN_DECAY_DECAY_RATE,
                                             staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    TRAIN_DATASET = ModelNetH5Dataset('./data/modelnet40_ply_hdf5_2048/train_files.txt',
                                      batch_size=batch_size, npoints=num_point, shuffle=True)
    TEST_DATASET = ModelNetH5Dataset('data/modelnet40_ply_hdf5_2048/test_files.txt',
                                     batch_size=batch_size, npoints=num_point, shuffle=False)

    point_cloud, labels = placeholder_inputs(batch_size, num_point)
    is_training_pl = tf.placeholder(tf.bool, shape=())

    # Note the global_step=batch parameter to minimize.
    # That tells the optimizer to helpfully increment the 'batch' parameter
    # for you every time it trains.
    batch = tf.get_variable('batch', [],
                            initializer=tf.constant_initializer(0), trainable=False)
    bn_decay = get_bn_decay(batch)
    tf.summary.scalar('bn_decay', bn_decay)

    pred = pointnet2(point_cloud, nb_classes, is_training_pl)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    losses = tf.get_collection('losses')
    total_loss = tf.add_n(losses, name='total_loss')
    tf.summary.scalar('total_loss', total_loss)
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name, l)

    correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels))
    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(batch_size)
    tf.summary.scalar('accuracy', accuracy)

    train_op = tf.train.AdamOptimizer(get_learning_rate(batch)).minimize(total_loss)

    init_op = tf.global_variables_initializer()
    session = K.get_session()
    session.run(init_op)

    with session.as_default():
        with tf.device('/gpu:0'):
            for i in range(epochs):
                # TODO: add early stop to prevent overfitting
                print('**** EPOCH %03d ****' % i)

                # Make sure batch data is of same size
                cur_batch_data = np.zeros((batch_size, num_point, TRAIN_DATASET.num_channel()))
                cur_batch_label = np.zeros((batch_size), dtype=np.int32)

                total_correct = 0
                total_seen = 0
                loss_sum = 0
                batch_idx = 0

                while TRAIN_DATASET.has_next_batch():
                    batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True)
                    bsize = batch_data.shape[0]
                    cur_batch_data[0:bsize, ...] = batch_data
                    cur_batch_label[0:bsize] = batch_label
                    _, loss_val, pred_val = session.run([train_op, total_loss, pred], feed_dict={
                        point_cloud: cur_batch_data,
                        labels: cur_batch_label,
                        is_training_pl: True,
                        batch: i
                    })

                    pred_val = np.argmax(pred_val, 1)
                    correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
                    total_correct += correct
                    total_seen += bsize
                    loss_sum += loss_val
                    if (batch_idx + 1) % 50 == 0:
                        print(' ---- batch: %03d ----' % (batch_idx + 1))
                        print('mean loss: %f' % (loss_sum / 50))
                        print('accuracy: %f' % (total_correct / float(total_seen)))
                        total_correct = 0
                        total_seen = 0
                        loss_sum = 0
                    batch_idx += 1
                TRAIN_DATASET.reset()

                # Make sure batch data is of same size
                cur_batch_data = np.zeros((batch_size, num_point, TEST_DATASET.num_channel()))
                cur_batch_label = np.zeros((batch_size), dtype=np.int32)

                total_correct = 0
                total_seen = 0
                loss_sum = 0
                batch_idx = 0
                total_seen_class = [0 for _ in range(nb_classes)]
                total_correct_class = [0 for _ in range(nb_classes)]

                print('---- EPOCH %03d EVALUATION ----' % i)

                while TEST_DATASET.has_next_batch():
                    batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
                    bsize = batch_data.shape[0]
                    # for the last batch in the epoch, the bsize:end are from last batch
                    cur_batch_data[0:bsize, ...] = batch_data
                    cur_batch_label[0:bsize] = batch_label

                    _, loss_val, pred_val = session.run([train_op, total_loss, pred], feed_dict={
                        point_cloud: cur_batch_data,
                        labels: cur_batch_label,
                        is_training_pl: False,
                        batch: i
                    })
                    pred_val = np.argmax(pred_val, 1)
                    correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
                    total_correct += correct
                    total_seen += bsize
                    loss_sum += loss_val
                    batch_idx += 1
                    for i in range(0, bsize):
                        l = batch_label[i]
                        total_seen_class[l] += 1
                        total_correct_class[l] += (pred_val[i] == l)

                print('eval mean loss: %f' % (loss_sum / float(batch_idx)))
                print('eval accuracy: %f' % (total_correct / float(total_seen)))
                print('eval avg class acc: %f' % (
                    np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

                TEST_DATASET.reset()


if __name__ == '__main__':
    train()
