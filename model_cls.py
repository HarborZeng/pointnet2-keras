from keras.layers import Conv2D, Dropout, Input, BatchNormalization, Dense, Lambda
from keras import backend as K
import numpy as np
import tensorflow as tf

sess = tf.Session()
K.set_session(sess)

from tf_ops.grouping.tf_grouping import query_ball_point, group_point
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point


def pointnet2(input_points, nb_classes):
    model_input = Input(tensor=input_points)

    sa1_xyz, sa1_points = set_abstraction_msg(model_input,
                                              None,
                                              512,
                                              [0.1, 0.2, 0.4],
                                              [16, 32, 128],
                                              [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                                              use_nchw=True)

    sa2_xyz, sa2_points = set_abstraction_msg(sa1_xyz,
                                              sa1_points,
                                              128,
                                              [0.2, 0.4, 0.8],
                                              [32, 64, 128],
                                              [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                                              use_nchw=False)

    sa3_xyz, sa3_points = set_abstraction(sa2_xyz,
                                          sa2_points,
                                          [256, 512, 1024])

    c = Lambda(lambda x: K.reshape(x, [16, -1]))(sa3_points)
    # point_net_cls
    c = Dense(512, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    c = Dense(256, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    prediction = Dense(nb_classes, activation='softmax')(c)
    return prediction


def set_abstraction_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, use_nchw):
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
    new_points_list = []
    for i in range(len(radius_list)):
        radius = radius_list[i]
        nsample = nsample_list[i]
        group_idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = group_point(xyz, group_idx[0])
        grouped_xyz -= K.tile(Lambda(lambda x: K.expand_dims(x, axis=2))(new_xyz), [1, 1, nsample, 1])
        if points is not None:
            grouped_points = group_point(points, group_idx[0])
            grouped_points = Lambda(lambda x: K.concatenate(x, axis=-1))([grouped_points, grouped_xyz])
        else:
            grouped_points = grouped_xyz
        if use_nchw: grouped_points = Lambda(lambda x: K.permute_dimensions(x, [0, 3, 1, 2]))(grouped_points)
        for j, num_out_channel in enumerate(mlp_list[i]):
            grouped_points = Conv2D(num_out_channel, 1, activation="relu")(grouped_points)
            grouped_points = BatchNormalization()(grouped_points)
        if use_nchw: grouped_points = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 3, 1]))(grouped_points)
        new_points = Lambda(lambda x: K.max(x, axis=2))(grouped_points)
        new_points_list.append(new_points)
    new_points_concat = Lambda(lambda x: K.concatenate(x, axis=-1))(new_points_list)
    return new_xyz, new_points_concat


def set_abstraction(xyz, points, mlp):
    # Sample and Grouping
    new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points)

    # Point Feature Embedding
    for i, num_out_channel in enumerate(mlp):
        new_points = Conv2D(num_out_channel, 1, activation="relu")(new_points)
        new_points = BatchNormalization()(new_points)

    # Pooling in Local Regions
    new_points = Lambda(lambda x: K.max(x, axis=2, keepdims=True))(new_points)

    new_points = Lambda(lambda x: K.squeeze(x, 2))(new_points)  # (batch_size, npoints, mlp2[-1])
    return new_xyz, new_points


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
    new_xyz = tf.constant(np.tile(np.array([0, 0, 0]).reshape((1, 1, 3)), (batch_size, 1, 1)), dtype=tf.float32)  # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1, 1, nsample)), (batch_size, 1, 1)))
    grouped_xyz = Lambda(lambda x: K.reshape(x, (batch_size, 1, nsample, 3)))(xyz)  # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = Lambda(lambda x: K.concatenate(x, axis=2))([xyz, points])  # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = Lambda(lambda x: K.expand_dims(x, 1))(new_points)  # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz
