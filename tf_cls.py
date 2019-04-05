import tensorflow as tf
from model_cls import pointnet2
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
import os
from keras import backend as K
from modelnet_h5_dataset import ModelNetH5Dataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sn
import pandas as pd

nb_classes = 40

epochs = 100
batch_size = 16
num_point = 1024

decay_step = 200000
bn_init_decay = 0.5
bn_decay_decay_rate = 0.5
bn_decay_decay_step = float(decay_step)
bn_decay_clip = 0.99
decay_rate = 0.7

summary_dir = 'summary'
image_dir = 'result_image'
train_log_dir = 'train_out'

classes = []

with open('data/modelnet40_ply_hdf5_2048/shape_names.txt', 'r') as shapeName:
    for line in shapeName.readlines():
        classes.append(line.strip())


def plot_history(history, result_dir, show_on_train=True):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    plt.plot(history['acc'], marker='.')
    plt.plot(history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')

    if os.path.exists(os.path.join(result_dir, 'model_accuracy.png')):
        os.remove(os.path.join(result_dir, 'model_accuracy.png'))
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))

    if show_on_train:
        plt.show()

    plt.close()

    plt.plot(history['loss'], marker='.')
    plt.plot(history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')

    if os.path.exists(os.path.join(result_dir, 'model_loss.png')):
        os.remove(os.path.join(result_dir, 'model_loss.png'))

    plt.savefig(os.path.join(result_dir, 'model_loss.png'))

    if show_on_train:
        plt.show()

    plt.close()


def save_history(history, result_dir):
    loss = history['loss']
    acc = history['acc']
    val_loss = history['val_loss']
    val_acc = history['val_acc']
    nb_epoch = len(acc)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()


def get_learning_rate(epoch_index):
    learning_rate = tf.train.exponential_decay(0.001,  # Base learning rate.
                                               epoch_index,  # Current index into the dataset.
                                               decay_step,  # Decay step.
                                               decay_rate,  # Decay rate.
                                               staircase=True)
    learning_rate = K.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(epoch_index):
    bn_momentum = tf.train.exponential_decay(bn_init_decay,
                                             epoch_index,
                                             bn_decay_decay_step,
                                             bn_decay_decay_rate,
                                             staircase=True)
    bn_decay = K.minimum(bn_decay_clip, 1 - bn_momentum)
    return bn_decay


def train():
    train_dataset = ModelNetH5Dataset('./data/modelnet40_ply_hdf5_2048/train_files.txt',
                                      batch_size=batch_size, npoints=num_point, shuffle=True)
    test_dataset = ModelNetH5Dataset('data/modelnet40_ply_hdf5_2048/test_files.txt',
                                     batch_size=batch_size, npoints=num_point, shuffle=False)

    point_cloud = K.placeholder(dtype=np.float32, shape=(batch_size, num_point, 3))
    labels = K.placeholder(dtype=np.int32, shape=batch_size)
    is_training_pl = K.placeholder(dtype=np.bool, shape=())

    # Note the global_step=batch parameter to minimize.
    # That tells the optimizer to helpfully increment the 'batch' parameter
    # for you every time it trains.
    current_epoch = tf.Variable(0)
    bn_decay = get_bn_decay(current_epoch)
    tf.summary.scalar('bn_decay', bn_decay)

    pred = pointnet2(point_cloud, nb_classes, is_training_pl)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify_loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    losses = tf.get_collection('losses')
    total_loss = tf.add_n(losses, name='total_loss')
    tf.summary.scalar('total_loss', total_loss)
    for the_lable in losses + [total_loss]:
        tf.summary.scalar(the_lable.op.name, the_lable)

    correct = K.equal(K.argmax(pred, axis=1), tf.to_int64(labels))
    accuracy = tf.reduce_sum(K.cast(correct, 'float32')) / batch_size
    tf.summary.scalar('accuracy', accuracy)

    learning_rate = get_learning_rate(current_epoch)
    tf.summary.scalar('learning_rate', learning_rate)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=current_epoch)

    saver = tf.train.Saver()

    session = K.get_session()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train'), session.graph)
    test_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'test'), session.graph)

    init_op = tf.global_variables_initializer()
    session.run(init_op)

    with session.as_default():
        with tf.device('/gpu:0'):

            history = {
                "acc": [],
                "val_acc": [],
                "loss": [],
                "val_loss": []
            }

            for epoch in range(epochs):
                # TODO: add early stop to prevent overfitting
                print('**** EPOCH {} ****'.format(epoch))

                # Make sure batch data is of same size
                cur_batch_data = np.zeros((batch_size, num_point, train_dataset.num_channel()))
                cur_batch_label = np.zeros(batch_size, dtype=np.int32)

                train_total_correct = 0
                train_total_seen = 0
                train_loss_sum = 0
                train_batch_idx = 0

                while train_dataset.has_next_batch():
                    batch_data, batch_label = train_dataset.next_batch(augment=True)
                    bsize = batch_data.shape[0]
                    cur_batch_data[0:bsize, ...] = batch_data
                    cur_batch_label[0:bsize] = batch_label
                    _, loss_val, pred_val, summary = session.run([train_op, total_loss, pred, merged],
                                                                 feed_dict={
                                                                     point_cloud: cur_batch_data,
                                                                     labels: cur_batch_label,
                                                                     is_training_pl: True,
                                                                     current_epoch: epoch
                                                                 })

                    pred_val = np.argmax(pred_val, 1)
                    correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
                    train_total_correct += correct
                    train_total_seen += bsize
                    train_loss_sum += loss_val
                    train_batch_idx += 1

                train_loss = train_loss_sum / train_batch_idx
                print('mean loss:\t{:.4f}'.format(train_loss))
                train_acc = train_total_correct / train_total_seen
                print('accuracy:\t{:.2%}'.format(train_acc))

                # only save these parameter on every epoch
                history['acc'].append(train_acc)
                history['loss'].append(train_loss)

                train_writer.add_summary(summary, epoch)

                train_dataset.reset()

                # Make sure batch data is of same size
                cur_batch_data = np.zeros((batch_size, num_point, test_dataset.num_channel()))
                cur_batch_label = np.zeros(batch_size, dtype=np.int32)

                total_correct = 0
                total_seen = 0
                loss_sum = 0
                batch_idx = 0
                total_seen_class = [0 for _ in range(nb_classes)]
                total_correct_class = [0 for _ in range(nb_classes)]

                print('---- EPOCH {} EVALUATION ----'.format(epoch))

                while test_dataset.has_next_batch():
                    batch_data, batch_label = test_dataset.next_batch(augment=False)
                    bsize = batch_data.shape[0]
                    # for the last batch in the epoch, the bsize:end are from last batch
                    cur_batch_data[0:bsize, ...] = batch_data
                    cur_batch_label[0:bsize] = batch_label

                    _, loss_val, test_pred_val, summary = session.run([train_op, total_loss, pred, merged],
                                                                      feed_dict={
                                                                          point_cloud: cur_batch_data,
                                                                          labels: cur_batch_label,
                                                                          is_training_pl: False,
                                                                          current_epoch: epoch
                                                                      })

                    test_pred_val = np.argmax(test_pred_val, 1)
                    correct = np.sum(test_pred_val[0:bsize] == batch_label[0:bsize])
                    total_correct += correct
                    total_seen += bsize
                    loss_sum += loss_val
                    batch_idx += 1
                    for bindex in range(0, bsize):
                        the_lable = batch_label[bindex]
                        total_seen_class[the_lable] += 1
                        total_correct_class[the_lable] += (test_pred_val[bindex] == the_lable)

                val_loss = loss_sum / batch_idx
                print('eval mean loss:\t{:.4f}'.format(val_loss))
                val_acc = total_correct / total_seen
                print('eval accuracy:\t{:.2%}'.format(val_acc))
                print('eval avg class acc:\t{:.2%}'.format(
                    np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

                # only save these parameter on every epoch
                history['val_acc'].append(val_acc)
                history['val_loss'].append(val_loss)

                test_writer.add_summary(summary, epoch)

                test_dataset.reset()

                if epoch % 10 == 0:
                    save_path = saver.save(session, os.path.join(summary_dir, "model.ckpt"))
                    print("Model saved in file: {}".format(save_path))

            plot_history(history, image_dir)
            save_history(history, train_log_dir)

            y_pred = test_pred_val
            _, y_labels = test_dataset.next_batch()
            y_true = np.argmax(y_labels, 1)
            # 计算模型的 metrics
            print("Precision", precision_score(y_true.tolist(), y_pred.tolist(), average='weighted'))
            print("Recall", recall_score(y_true, y_pred, average='weighted'))
            print("f1_score", f1_score(y_true, y_pred, average='weighted'))
            print("confusion_matrix")
            conf_mat = confusion_matrix(y_true, y_pred)
            print(conf_mat)

            df_cm = pd.DataFrame(conf_mat, index=[i for i in classes],
                                 columns=[i for i in classes])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
            plt.title('confusion matrix')
            plt.show()


if __name__ == '__main__':
    train()
