import tensorflow as tf
from model_cls import pointnet2
import matplotlib.pyplot as plt
import os
from keras import backend as K
from modelnet_h5_dataset import ModelNetH5Dataset
import numpy as np
from pointnet2_cls_msg import get_model
from tqdm import tqdm
import time
import pandas as pd
import seaborn as sn
from sklearn.metrics import precision_score, recall_score, f1_score

# specify to use keras model (implemented by HarborZeng)
# or tensorflow model (implemented by CharlesQi)
use_keras_model = True

# total number of classes
nb_classes = 40

# the epoch count to train
epochs = 200
# the size of ever mini-batch
batch_size = 16
# the number of points in a train/eval data
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

# the classes list
classes = []

with open('data/modelnet40_ply_hdf5_2048/shape_names.txt', 'r') as shapeName:
    for line in shapeName.readlines():
        classes.append(line.strip())


def plot_history(history, result_dir, show_on_train=True):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    plt.plot(history['acc'], marker='.')
    plt.plot(history['test_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'test_acc'], loc='lower right')

    if os.path.exists(os.path.join(result_dir, 'model_accuracy.png')):
        os.remove(os.path.join(result_dir, 'model_accuracy.png'))
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))

    if show_on_train:
        plt.show()

    plt.close()

    plt.plot(history['loss'], marker='.')
    plt.plot(history['test_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'test_loss'], loc='upper right')

    if os.path.exists(os.path.join(result_dir, 'model_loss.png')):
        os.remove(os.path.join(result_dir, 'model_loss.png'))

    plt.savefig(os.path.join(result_dir, 'model_loss.png'))

    if show_on_train:
        plt.show()

    plt.close()


def save_history(history, result_dir):
    loss = history['loss']
    acc = history['acc']
    val_loss = history['test_loss']
    val_acc = history['test_acc']
    ps = history['precision_score']
    recall = history['recall_score']
    fs = history['f1_score']
    nb_epoch = len(acc)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\ttest_loss\ttest_acc\tprecision_score\trecall\tf1_score\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i], ps[i], recall[i], fs[i]))
        fp.close()


def get_learning_rate(step):
    learning_rate = tf.train.exponential_decay(0.001,  # Base learning rate.
                                               step * batch_size,  # Current index into the dataset.
                                               decay_step,  # Decay step.
                                               decay_rate,  # Decay rate.
                                               staircase=True)
    learning_rate = K.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        bn_init_decay,
        batch * batch_size,
        bn_decay_decay_step,
        bn_decay_decay_rate,
        staircase=True)
    bn_decay = tf.minimum(bn_decay_clip, 1 - bn_momentum)
    return bn_decay


def train():
    train_dataset = ModelNetH5Dataset('./data/modelnet40_ply_hdf5_2048/train_files.txt',
                                      batch_size=batch_size, npoints=num_point, shuffle=True)
    test_dataset = ModelNetH5Dataset('./data/modelnet40_ply_hdf5_2048/test_files.txt',
                                     batch_size=batch_size, npoints=num_point, shuffle=False)

    point_cloud = K.placeholder(dtype=np.float32, shape=(batch_size, num_point, 3), name='x')
    labels = K.placeholder(dtype=np.int32, shape=batch_size, name='y')
    is_training_pl = K.placeholder(dtype=np.bool, shape=())

    # Note the global_step=global_step parameter to minimize.
    # That tells the optimizer to helpfully increment the 'global_step' parameter
    # for you every time it trains.
    global_step = tf.train.get_or_create_global_step()

    if use_keras_model:
        logits = pointnet2(point_cloud, nb_classes, is_training_pl)
    else:
        logits = get_model(point_cloud, is_training_pl, get_bn_decay(global_step))

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    losses = tf.get_collection('losses')
    total_loss = tf.add_n(losses, name='total_loss')
    tf.summary.scalar('total_loss', total_loss)
    for the_lable in losses + [total_loss]:
        tf.summary.scalar(the_lable.op.name, the_lable)

    correct = K.equal(K.argmax(logits, axis=1), tf.to_int64(labels))
    accuracy = tf.reduce_sum(K.cast(correct, 'float32')) / batch_size
    tf.summary.scalar('accuracy', accuracy)

    learning_rate = get_learning_rate(global_step)
    tf.summary.scalar('learning_rate', learning_rate)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step, name='train_op')

    saver = tf.train.Saver()

    session = K.get_session()

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # delete previous summary and ckpt file
    tf.gfile.DeleteRecursively(summary_dir)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train'), session.graph)

    init_op = tf.global_variables_initializer()
    session.run(init_op)

    with session.as_default():
        with tf.device('/gpu:0'):

            history = {
                "acc": [],
                "test_acc": [],
                "loss": [],
                "test_loss": [],
                "precision_score": [],
                "recall_score": [],
                "f1_score": [],
            }

            for epoch in range(epochs):
                print('**** EPOCH {} ****'.format(epoch))
                time.sleep(4)

                # Make sure batch data is of same size
                cur_batch_data = np.zeros((batch_size, num_point, train_dataset.num_channel()))
                cur_batch_label = np.zeros(batch_size, dtype=np.int32)

                train_total_correct = 0
                train_total_seen = 0
                train_loss_sum = 0
                train_batch_idx = 0

                with tqdm(total=train_dataset.total_batch(), unit='batches') as pbar:
                    while train_dataset.has_next_batch():
                        batch_data, batch_label = train_dataset.next_batch(augment=True)
                        bsize = batch_data.shape[0]
                        cur_batch_data[0:bsize, ...] = batch_data
                        cur_batch_label[0:bsize] = batch_label
                        _, loss_val, pred_val, summary, step = session.run(
                            [train_op, total_loss, logits, merged, global_step],
                            feed_dict={
                                point_cloud: cur_batch_data,
                                labels: cur_batch_label,
                                is_training_pl: True,
                            })

                        train_writer.add_summary(summary, step)

                        pred_val = np.argmax(pred_val, 1)
                        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
                        train_total_correct += correct
                        train_total_seen += bsize
                        train_loss_sum += loss_val
                        train_batch_idx += 1

                        train_acc = correct / bsize
                        pbar.set_description('train_acc:\t{:.2%}, train_loss:\t{:.4f}'.format(train_acc, loss_val))
                        pbar.update()

                time.sleep(4)
                train_loss = train_loss_sum / train_batch_idx
                print('mean loss:\t\t{:.4f}'.format(train_loss))
                train_acc = train_total_correct / train_total_seen
                print('accuracy:\t\t{:.2%}'.format(train_acc))

                # only save these parameter on every epoch
                history['acc'].append(train_acc)
                history['loss'].append(train_loss)

                train_dataset.reset()

                # Make sure global_step data is of same size
                cur_batch_data = np.zeros((batch_size, num_point, test_dataset.num_channel()))
                cur_batch_label = np.zeros(batch_size, dtype=np.int32)

                total_correct = 0
                total_seen = 0
                loss_sum = 0
                batch_idx = 0
                total_seen_class = [0 for _ in range(nb_classes)]
                total_correct_class = [0 for _ in range(nb_classes)]
                total_test_pred_vals = []
                total_batch_labels = []

                print('---- EPOCH {} EVALUATION ----'.format(epoch))
                time.sleep(4)

                with tqdm(total=test_dataset.total_batch(), unit='batches') as pbar:
                    while test_dataset.has_next_batch():
                        batch_data, batch_label = test_dataset.next_batch(augment=True)
                        bsize = batch_data.shape[0]
                        # for the last global_step in the epoch, the bsize:end are from last global_step
                        cur_batch_data[0:bsize, ...] = batch_data
                        cur_batch_label[0:bsize] = batch_label

                        _, loss_val, test_pred_val, summary = session.run(
                            [train_op, total_loss, logits, merged],
                            feed_dict={
                                point_cloud: cur_batch_data,
                                labels: cur_batch_label,
                                is_training_pl: False,
                            })
                        test_pred_val = np.argmax(test_pred_val, 1)
                        correct = np.sum(test_pred_val[0:bsize] == batch_label[0:bsize])
                        total_test_pred_vals = np.concatenate((total_test_pred_vals, test_pred_val[0:bsize]))
                        total_batch_labels = np.concatenate((total_batch_labels, batch_label[0:bsize]))
                        total_correct += correct
                        total_seen += bsize
                        loss_sum += loss_val
                        batch_idx += 1
                        for bindex in range(0, bsize):
                            the_lable = batch_label[bindex]
                            total_seen_class[the_lable] += 1
                            total_correct_class[the_lable] += (test_pred_val[bindex] == the_lable)

                        test_acc = correct / bsize
                        pbar.set_description('test_acc:\t{:.2%}, test_loss:\t{:.4f}. '.format(test_acc, loss_val))
                        pbar.update()

                time.sleep(4)
                test_loss = loss_sum / batch_idx
                print('eval mean loss:\t\t{:.4f}'.format(test_loss))
                test_acc = total_correct / total_seen
                print('eval accuracy:\t\t{:.2%}'.format(test_acc))
                print('eval avg class acc:\t{:.2%}'.format(
                    np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

                # draw confusion_matrix
                if epoch == epochs - 1:
                    plot_cm(session, total_batch_labels, total_test_pred_vals, image_dir)
                # 计算模型的 metrics
                ps = precision_score(total_batch_labels.tolist(),
                                     total_test_pred_vals.tolist(),
                                     average='weighted',
                                     labels=np.unique(total_test_pred_vals))
                print("Precision:\t{:.2%}".format(ps))

                recall = recall_score(total_batch_labels,
                                      total_test_pred_vals,
                                      average='weighted')
                print("Recall:\t\t{:.2%}".format(recall))

                fs = f1_score(total_batch_labels,
                              total_test_pred_vals,
                              average='weighted',
                              labels=np.unique(total_test_pred_vals))
                print("f1_score:\t{:.2%}".format(fs))

                # only save these parameter on every epoch
                history['test_acc'].append(test_acc)
                history['test_loss'].append(test_loss)
                history['precision_score'].append(ps)
                history['recall_score'].append(recall)
                history['f1_score'].append(fs)

                test_dataset.reset()

                if (epoch + 1) % 10 == 0:
                    save_path = saver.save(session, os.path.join(summary_dir, "model.ckpt"))
                    print("\nModel saved in file: {}".format(save_path))

            plot_history(history, image_dir)
            save_history(history, train_log_dir)


def plot_cm(session, total_batch_labels, total_test_pred_vals, result_dir):
    confusion_matrix_tensor = tf.confusion_matrix(total_batch_labels, total_test_pred_vals, 40)
    confusion_matrix = session.run(confusion_matrix_tensor)
    print(confusion_matrix)
    df_cm = pd.DataFrame(confusion_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(10, 8), dpi=200)
    sn.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('confusion matrix')

    if os.path.exists(os.path.join(result_dir, 'confusion_matrix.png')):
        os.remove(os.path.join(result_dir, 'confusion_matrix.png'))
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))

    plt.show()


if __name__ == '__main__':
    train()
