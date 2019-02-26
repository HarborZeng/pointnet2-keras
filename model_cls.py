from keras.layers import Conv2D, Flatten, Dropout, Input, BatchNormalization, Dense
from keras.models import Model
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf


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
    input_points = Input(shape=(2048, 3))

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

    model = Model(inputs=input_points, outputs=prediction)

    return model


def set_abstraction_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list):
    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    new_points_list = []
    for i in range(len(radius_list)):
        radius = radius_list[i]
        nsample = nsample_list[i]
        group_idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])
        if points is not None:
            grouped_points = index_points(points, group_idx)
            grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
        else:
            grouped_points = grouped_xyz
        grouped_points = tf.transpose(grouped_points, [0, 3, 1, 2])
        for j, num_out_channel in enumerate(mlp_list[i]):
            grouped_points = Conv2D(activation="relu")(grouped_points)
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
        new_points = Conv2D(activation="relu")(new_points)
        new_points = BatchNormalization()(new_points)
    new_points = tf.transpose(new_points, [0, 2, 3, 1])

    # Pooling in Local Regions
    new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')

    new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
    return new_xyz, new_points


def square_distance(src, dst):
    """
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * np.matmul(src, dst.permute(0, 2, 1))
    dist += np.sum(src ** 2, -1).view(B, N, 1)
    dist += np.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, D1, D2, ..., Dn]
    Return:
        new_points:, indexed points data, [B, D1, D2, ..., Dn, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = np.arange(B, dtype=np.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud data, [B, npoint, C]
    """
    B, N, C = xyz.shape
    S = npoint
    centroids = np.zeros((B, S), dtype=np.long)
    distance = np.ones((B, N), dtype=np.long) * 1e10
    farthest = np.random.randint(0, N, (B,), dtype=np.long)
    batch_indices = np.arange(B, dtype=np.long)
    for i in range(S):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    K = nsample
    group_idx = np.arange(N, dtype=np.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :K]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, K])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint

    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz -= new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = tf.concat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = np.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = tf.concat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points
