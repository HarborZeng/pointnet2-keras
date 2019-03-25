from keras.layers import Conv2D, Flatten, Dropout, Input, BatchNormalization, Dense, InputLayer
from keras.models import Model
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point


class MatMul(Layer):

    def __init__(self, **kwargs):
        super(MatMul, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('`MatMul` layer should be called '
                             'on a list of inputs')
        if len(input_shape) != 2:
            raise ValueError('The input of `MatMul` layer should be a list containing 2 elements')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError('The dimensions of each element of inputs should be 3')

        if input_shape[0][-1] != input_shape[1][1]:
            raise ValueError('The last dimension of inputs[0] should match the dimension 1 of inputs[1]')

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A `MatMul` layer should be called '
                             'on a list of inputs.')
        return tf.matmul(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0][0], input_shape[0][1], input_shape[1][-1]]
        return tuple(output_shape)


def pointnet2(nb_classes):
    input_points = tf.placeholder(tf.float32, shape=(32, 2048, 3))

    sa1_xyz, sa1_points = set_abstraction_msg(input_points,
                                              None,
                                              512,
                                              [0.1, 0.2, 0.4],
                                              [16, 32, 128],
                                              [[32, 32, 64], [64, 64, 128], [64, 96, 128]])

    sa2_xyz, sa2_points = set_abstraction_msg(sa1_xyz,
                                              sa1_points,
                                              128,
                                              [0.2, 0.4, 0.8],
                                              [32, 64, 128],
                                              [[64, 64, 128], [128, 128, 256], [128, 128, 256]])

    sa3_xyz, sa3_points = set_abstraction(sa2_xyz,
                                          sa2_points,
                                          [256, 512, 1024])

    # point_net_cls
    c = Dense(512, activation='relu')(sa3_points)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    c = Dense(256, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    c = Dense(nb_classes, activation='softmax')(c)
    prediction = Flatten()(c)

    model = Model(inputs=Input(shape=(2048, 3)), outputs=prediction)

    # turn tf tensor to keras
    return model(input_points)


def set_abstraction_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list):
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
    new_points_list = []
    for i in range(len(radius_list)):
        radius = radius_list[i]
        nsample = nsample_list[i]
        group_idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = group_point(xyz, group_idx)
        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])
        if points is not None:
            grouped_points = group_point(points, group_idx)
            grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
        else:
            grouped_points = grouped_xyz
        grouped_points = tf.transpose(grouped_points, [0, 3, 1, 2])
        for j, num_out_channel in enumerate(mlp_list[i]):
            grouped_points = Conv2D(num_out_channel, 1, activation="relu")(grouped_points)
            grouped_points = BatchNormalization()(grouped_points)
        grouped_points = tf.transpose(grouped_points, [0, 2, 3, 1])
        new_points = tf.reduce_max(grouped_points, axis=[2])
        new_points_list.append(new_points)
    new_points_concat = tf.concat(new_points_list, axis=-1)
    return new_xyz, new_points_concat


def set_abstraction(xyz, points, mlp):
    # Sample and Grouping
    new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points)

    # Point Feature Embedding
    new_points = tf.transpose(new_points, [0, 3, 1, 2])
    for i, num_out_channel in enumerate(mlp):
        new_points = Conv2D(num_out_channel, 1, activation="relu")(new_points)
        new_points = BatchNormalization()(new_points)
    new_points = tf.transpose(new_points, [0, 2, 3, 1])

    # Pooling in Local Regions
    new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')

    new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
    return new_xyz, new_points


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    """
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    """

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))  # (batch_size, npoint, 3)
    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    """
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0, 0, 0]).reshape((1, 1, 3)), (batch_size, 1, 1)),
                          dtype=tf.float32)  # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1, 1, nsample)), (batch_size, 1, 1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3))  # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)  # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)  # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz
