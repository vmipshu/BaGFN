# -coding:utf-8-
"""
File Name:util
Description:
        Author:Tony
        Date:2020/1/2 14:24
"""
import os
import shutil
import logging
import tensorflow as tf


def create_dir(dir):
    """
    dirs - a directory to create if it is not found
    :param dir: directory path, string
    :return exit_code: 0:success -1:failed and exit
    """
    try:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def createlog(log_dir):
    """
    Setting logger. Save log to a local file and show log in screen.
    :param log_dir: logger file save path.
    :return: logger object.
    """
    log = os.path.join(log_dir, "log.txt")
    os.mknod(log)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log)
    formatter = logging.Formatter('%(asctime)s = %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def pairwise_distance(input):
    """Compute pairwise distance of a point cloud.
 
    Args:
      input: tensor (batch_size, num_points, num_dims)
  
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = input.get_shape().as_list()[0]
    input = tf.squeeze(input)
    if og_batch_size == 1:
        input = tf.expand_dims(input, 0)

    input_transpose = tf.transpose(input, perm=[0, 2, 1])
    input_inner = tf.matmul(input, input_transpose)
    input_inner = -2 * input_inner

    input_square = tf.reduce_sum(tf.square(input), axis=-1, keepdims=True)
    input_square_tranpose = tf.transpose(input_square, perm=[0, 2, 1])

    return input_square + input_inner + input_square_tranpose


def knn_top(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int
    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    neg_adj = -adj_matrix
    _, nn_idx = tf.math.top_k(neg_adj, k=k)

    return nn_idx


def get_edge_feature(input, nn_idx, k=20):
    """Construct edge feature for each point
    Args:
      input: (batch_size, num_points, 1, num_dims)
      nn_idx: (batch_size, num_points, k)
      k: int

    Returns:
      edge features: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = input.get_shape().as_list()[0]
    input = tf.squeeze(input)
    if og_batch_size == 1:
        input = tf.expand_dims(input, 0)

    input_central = input

    input_shape = input.get_shape()
    batch_size = input_shape[0].value
    num_points = input_shape[1].value
    num_dims = input_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    input_flat = tf.reshape(input, [-1, num_dims])
    input_neighbors = tf.gather(input_flat, nn_idx + idx_)
    input_central = tf.expand_dims(input_central, axis=-2)

    input_central = tf.tile(input_central, [1, 1, k, 1])

    edge_feature = tf.concat([input_central, input_neighbors - input_central], axis=-1)

    return edge_feature


def knn_with_RBF_dist(adj_matrix, k=20, r=0.05):
    """Get KNN based on the pairwise distance.
	Args:
	  pairwise distance: (batch_size, num_points, num_points)
	  k: int
  
	Returns:
	  nearest neighbors: (batch_size, num_points, k)
	"""
    neg_adj = -adj_matrix
    dist, nn_idx = tf.nn.top_k(neg_adj, k=k + 1)
    dist = dist[..., 1:k + 1]
    nn_idx = nn_idx[..., 1:k + 1]
    #  mean_dist = tf.reduce_mean(-dist, axis = -1, keep_dims=True)
    #  mean_dist = tf.tile(mean_dist, [1,1,k])
    dist = tf.exp(dist / (2 * r * r))
    dist = tf.expand_dims(dist, axis=-1)
    return dist, nn_idx


def knn_random(adj_matrix, max_k=40, k=20):
    """Get KNN based on the pairwise distance.
	Args:
	  pairwise distance: (batch_size, num_points, num_points)
	  k: int
  
	Returns:
	  nearest neighbors: (batch_size, num_points, k)
	"""
    neg_adj = -adj_matrix
    _, nn_idx = tf.nn.top_k(neg_adj, k=max_k + 1)

    nn_idx = nn_idx[..., 1:max_k + 1]
    indices = tf.random_shuffle(tf.range(max_k))
    indices, _ = tf.nn.top_k(indices[0:k], k=k)
    return tf.gather(nn_idx, indices, axis=-1)


def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
	Args:
	  pairwise distance: (batch_size, num_points, num_points)
	  k: int
  
	Returns:
	  nearest neighbors: (batch_size, num_points, k)
	"""
    neg_adj = -adj_matrix
    _, nn_idx = tf.nn.top_k(neg_adj, k=k)
    return nn_idx


def get_new_edge_feature(point_cloud, nn_idx, k=20, r=0.05):
    """Construct edge feature for each point
	Args:
	  point_cloud: (batch_size, num_points, 1, num_dims)
	  nn_idx: (batch_size, num_points, k)
	  k: int
  
	Returns:
	  edge features: (batch_size, num_points, k, num_dims)
	"""
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])
    edge_feature = tf.concat([point_cloud_central,
                              point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature


def get_triangle_edge_feature(point_cloud, nn_idx, k=20):
    """Construct edge feature for each point
	Args:
	  point_cloud: (batch_size, num_points, 1, num_dims)
	  nn_idx: (batch_size, num_points, k)
	  k: int
  
	Returns:
	  edge features: (batch_size, num_points, k, num_dims)
	"""
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)
    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    point_cloud_neighbors_reverse = tf.reverse_v2(point_cloud_neighbors, axis=[-2])

    edge_feature = tf.concat([point_cloud_central,
                              point_cloud_neighbors - point_cloud_central,
                              point_cloud_neighbors_reverse - point_cloud_central], axis=-1)
    return edge_feature


def get_edge_cross_feature(point_cloud, nn_idx, k=20):
    """Construct edge feature for each point
	Args:
	  point_cloud: (batch_size, num_points, 1, num_dims)
	  nn_idx: (batch_size, num_points, k)
	  k: int
  
	Returns:
	  edge features: (batch_size, num_points, k, num_dims)
	"""
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors - point_cloud_central,
                              tf.cross(point_cloud_central,
                                       point_cloud_neighbors - point_cloud_central)], axis=-1)
    return edge_feature
