import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from model_cls import pointnet2
from util.provider import rotate_point_cloud_by_angle
import modelnet_h5_dataset

num_classes = 40
num_point = 1024
batch_size = 16
shape_names = [line.rstrip() for line in open('data/modelnet40_ply_hdf5_2048/shape_names.txt')]

test_dataset = modelnet_h5_dataset.ModelNetH5Dataset(
    'data/modelnet40_ply_hdf5_2048/test_files.txt', batch_size=batch_size,
    npoints=num_point, shuffle=False)


def evaluate(num_votes=1):
    with tf.device('/gpu:0'):
        point_cloud = tf.placeholder(dtype=np.float32, shape=(batch_size, num_point, 3))
        labels = tf.placeholder(dtype=np.int32, shape=batch_size)
        is_training_pl = tf.placeholder(dtype=np.bool, shape=())

        # simple model
        logits = pointnet2(point_cloud, num_classes, is_training_pl)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', cross_entropy_mean)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, 'summary')
    print("Model restored.")

    ops = {'pointclouds_pl': point_cloud,
           'labels_pl': labels,
           'is_training_pl': is_training_pl,
           'pred': logits,
           'loss': total_loss}

    eval_one_epoch(sess, ops, num_votes)


def eval_one_epoch(sess, ops, num_votes=1):
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((batch_size, num_point, test_dataset.num_channel()))
    cur_batch_label = np.zeros(batch_size, dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(num_classes)]
    total_correct_class = [0 for _ in range(num_classes)]

    time.sleep(4)
    with tqdm(total=test_dataset.total_batch(), unit='batches') as pbar:
        while test_dataset.has_next_batch():
            batch_data, batch_label = test_dataset.next_batch(augment=False)
            bsize = batch_data.shape[0]
            # for the last batch in the epoch, the bsize:end are from last batch
            cur_batch_data[0:bsize, ...] = batch_data
            cur_batch_label[0:bsize] = batch_label

            batch_pred_sum = np.zeros((batch_size, num_classes))  # score for classes
            for vote_idx in range(num_votes):
                # Shuffle point order to achieve different farthest samplings
                shuffled_indices = np.arange(num_point)
                np.random.shuffle(shuffled_indices)
                rotated_data = rotate_point_cloud_by_angle(cur_batch_data[:, shuffled_indices, :],
                                                           vote_idx / num_votes * np.pi * 2)
                feed_dict = {ops['pointclouds_pl']: rotated_data,
                             ops['labels_pl']: cur_batch_label,
                             ops['is_training_pl']: is_training}
                loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
                batch_pred_sum += pred_val
            pred_val = np.argmax(batch_pred_sum, 1)
            correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
            total_correct += correct
            total_seen += bsize
            loss_sum += loss_val
            batch_idx += 1
            for i in range(bsize):
                l = batch_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i] == l)
            test_acc = correct / bsize
            pbar.set_description('eval_acc:\t{:.2%}, eval_loss:\t{:.4f}. '.format(test_acc, loss_val))
            pbar.update()

    time.sleep(4)
    test_loss = loss_sum / batch_idx
    print('eval mean loss:\t\t{:.4f}'.format(test_loss))
    test_acc = total_correct / total_seen
    print('eval accuracy:\t\t{:.2%}'.format(test_acc))
    print('eval avg class acc:\t{:.2%}'.format(
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    for i, name in enumerate(shape_names):
        print('%10s:\t%0.3f' % (name, class_accuracies[i]))


if __name__ == '__main__':
    with tf.Graph().as_default():
        evaluate()
