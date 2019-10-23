"""
This code reproduces the MNIST results from the paper
Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning
https://arxiv.org/abs/1803.04765

The LSH backend used in the paper is FALCONN. This script also demonstrates
how to use an alternative backend called FAISS.

Code source: https://github.com/tensorflow/cleverhans/tree/master/cleverhans/model_zoo
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from bisect import bisect_left
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange
from enum import Enum
import tensorflow as tf
import time



class NearestNeighbor:
  class BACKEND(Enum):
    FALCONN = 1
    FAISS = 2

  def __init__(
    self,
    backend,
    dimension,
    neighbors,
    number_bits,
    nb_tables=None,
  ):
    assert backend in NearestNeighbor.BACKEND

    self._NEIGHBORS = neighbors
    self._BACKEND = backend

    if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
      self._init_falconn(
        dimension,
        number_bits,
        nb_tables
      )
    elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
      self._init_faiss(
        dimension,
      )
    else:
      raise NotImplementedError

  def _init_falconn(
    self,
    dimension,
    number_bits,
    nb_tables,
  ):
    import falconn

    assert nb_tables >= self._NEIGHBORS

    # LSH parameters
    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = dimension
    params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    params_cp.l = nb_tables
    params_cp.num_rotations = 2  # for dense set it to 1; for sparse data set it to 2
    params_cp.seed = 5721840
    # we want to use all the available threads to set up
    params_cp.num_setup_threads = 0
    params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable

    # we build number_bits-bit hashes so that each table has
    # 2^number_bits bins; a rule of thumb is to have the number
    # of bins be the same order of magnitude as the number of data points
    falconn.compute_number_of_hash_functions(number_bits, params_cp)
    self._falconn_table = falconn.LSHIndex(params_cp)
    self._falconn_query_object = None
    self._FALCONN_NB_TABLES = nb_tables

  def _init_faiss(
    self,
    dimension,
  ):
    import faiss

    res = faiss.StandardGpuResources()

    self._faiss_index = faiss.GpuIndexFlatL2(
      res,
      dimension,
    )

  def _find_knns_falconn(self, x, output):
    # Late falconn query_object construction
    # Since I suppose there might be an error
    # if table.setup() will be called after
    if self._falconn_query_object is None:
      self._falconn_query_object = self._falconn_table.construct_query_object()
      self._falconn_query_object.set_num_probes(
        self._FALCONN_NB_TABLES
      )

    missing_indices = np.zeros(output.shape, dtype=np.bool)

    print('Finding k-nearest neighbors with FALCONN...')
    start = time.time()
    for i in range(x.shape[0]):
      query_res = self._falconn_query_object.find_k_nearest_neighbors(
        x[i],
        self._NEIGHBORS
      )
      try:
        output[i, :] = query_res
      except:  # pylint: disable-msg=W0702
        # mark missing indices
        missing_indices[i, len(query_res):] = True

        output[i, :len(query_res)] = query_res
    end = time.time()
    print('Search completed! Time cost:', end-start, 's')
    return missing_indices

  def _find_knns_faiss(self, x, output):
    neighbor_distance, neighbor_index = self._faiss_index.search(
      x,
      self._NEIGHBORS
    )

    missing_indices = neighbor_distance == -1

    d1 = neighbor_index.reshape(-1)

    output.reshape(-1)[
      np.logical_not(missing_indices.flatten())
    ] = d1[
      np.logical_not(missing_indices.flatten())
    ]

    return missing_indices

  def add(self, x):
    if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
      self._falconn_table.setup(x)
    elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
      self._faiss_index.add(x)
    else:
      raise NotImplementedError

  def find_knns(self, x, output):
    if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
      return self._find_knns_falconn(x, output)
    elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
      return self._find_knns_faiss(x, output)
    else:
      raise NotImplementedError


class DkNNModel(object):
  def __init__(self, neighbors, layers, get_activations, train_data, train_labels,
               nb_classes, nb_tables=200, number_bits=18):
    """
    Implements the DkNN algorithm. See https://arxiv.org/abs/1803.04765 for more details.

    :param neighbors: number of neighbors to find per layer.
    :param layers: a list of layer names to include in the DkNN.
    :param get_activations: a callable that takes a np array and a layer name and returns its activations on the data.
    :param train_data: a np array of training data.
    :param train_labels: a np vector of training labels.
    :param nb_classes: the number of classes in the task.
    :param nb_tables: number of tables used by FALCONN to perform locality-sensitive hashing.
    :param number_bits: number of hash bits used by LSH Index.
    """
    self.neighbors = neighbors
    self.layers = layers
    self.get_activations = get_activations
    self.nb_classes = nb_classes
    self.nb_tables = nb_tables
    self.number_bits = number_bits

    self.nb_cali = -1
    self.calibrated = False

    # Compute training data activations
    self.nb_train = train_labels.shape[0]
    assert self.nb_train == train_data.shape[0]
    print('================================================================')
    print('Computing activations of the training data...')
    start = time.time()
    self.train_activations = get_activations(train_data)
    end = time.time()
    print('Computation done! Time cost:', end - start, 's.')
    self.train_labels = train_labels

    # Build locality-sensitive hashing tables for training representations
    self.train_activations_lsh = copy.copy(self.train_activations)
    print('================================================================')
    print('Initializing locality-sensitive hashing with FALCONN...')
    start = time.time()
    self.init_lsh()
    end = time.time()
    print('Initialization completed! Time cost:', end - start, 's.')

  def init_lsh(self):
    """
    Initializes locality-sensitive hashing with FALCONN to find nearest neighbors in training data.
    """
    self.query_objects = {
    }  # contains the object that can be queried to find nearest neighbors at each layer.
    # mean of training data representation per layer (that needs to be substracted before
    # NearestNeighbor).
    self.centers = {}
    for layer in self.layers:
      # Normalize all the lenghts, since we care about the cosine similarity.
      # The default order of the norm is Frobenius norm
      self.train_activations_lsh[layer] /= np.linalg.norm(
          self.train_activations_lsh[layer], axis=1).reshape(-1, 1)

      # Center the dataset and the queries: this improves the performance of LSH quite a bit.
      center = np.mean(self.train_activations_lsh[layer], axis=0)
      self.train_activations_lsh[layer] -= center
      self.centers[layer] = center
      print('-------------------For layer {}:-------------------'.format(layer))
      print('Constructing the NearestNeighbor table...')
      t1 = time.time()
      self.query_objects[layer] = NearestNeighbor(
        backend=NearestNeighbor.BACKEND.FALCONN,
        dimension=self.train_activations_lsh[layer].shape[1],
        neighbors=self.neighbors,
        number_bits=self.number_bits,
        nb_tables=self.nb_tables
      )

      self.query_objects[layer].add(self.train_activations_lsh[layer])
      t2 = time.time()
      print('Table constructed！Time cost:', t2-t1, 's.')

  def find_train_knns(self, data_activations):
    """
    Given a data_activation dictionary that contains a np array with activations for each layer,
    find the knns in the training data.
    """
    knns_ind = {}
    knns_labels = {}

    for layer in self.layers:
      # Pre-process representations of data to normalize and remove training data mean.
      data_activations_layer = copy.copy(data_activations[layer])
      nb_data = data_activations_layer.shape[0]
      data_activations_layer /= np.linalg.norm(
          data_activations_layer, axis=1).reshape(-1, 1)
      data_activations_layer -= self.centers[layer]

      # Use FALCONN to find indices of nearest neighbors in training data.
      knns_ind[layer] = np.zeros(
          (data_activations_layer.shape[0], self.neighbors), dtype=np.int32)
      # knn_errors = 0
      print('-------------------For layer {}:-------------------'.format(layer))
      knn_missing_indices = self.query_objects[layer].find_knns(
        data_activations_layer,
        knns_ind[layer],
      )

      # knn_errors += knn_missing_indices.flatten().sum()

      # Find labels of neighbors found in the training data.
      knns_labels[layer] = np.zeros((nb_data, self.neighbors), dtype=np.int32)

      knns_labels[layer].reshape(-1)[
        np.logical_not(
          knn_missing_indices.flatten()
        )
      ] = self.train_labels[
        knns_ind[layer].reshape(-1)[
          np.logical_not(
            knn_missing_indices.flatten()
          )
        ]
      ]

    return knns_ind, knns_labels

  def nonconformity(self, knns_labels):
    """
    Given an dictionary of nb_data x nb_classes dimension, compute the nonconformity of
    each candidate label for each data point: i.e. the number of knns whose label is
    different from the candidate label.
    """
    nb_data = knns_labels[self.layers[0]].shape[0]
    knns_not_in_class = np.zeros((nb_data, self.nb_classes), dtype=np.int32)
    for i in range(nb_data):
      # Compute number of nearest neighbors per class
      knns_in_class = np.zeros(
          (len(self.layers), self.nb_classes), dtype=np.int32)
      for layer_id, layer in enumerate(self.layers):
        # Count number of occurrences of each value in array of non-negative ints.
        knns_in_class[layer_id, :] = np.bincount(
            knns_labels[layer][i], minlength=self.nb_classes)

      # Compute number of knns in other class than class_id
      for class_id in range(self.nb_classes):
        knns_not_in_class[i, class_id] = np.sum(
            knns_in_class) - np.sum(knns_in_class[:, class_id])
    return knns_not_in_class

  def preds_conf_cred(self, knns_not_in_class):
    """
    Given an array of nb_data x nb_classes dimensions, use conformal prediction to compute
    the DkNN's prediction, confidence and credibility.
    """
    nb_data = knns_not_in_class.shape[0]
    preds_knn = np.zeros(nb_data, dtype=np.int32)
    confs = np.zeros((nb_data, self.nb_classes), dtype=np.float32)
    creds = np.zeros((nb_data, self.nb_classes), dtype=np.float32)

    for i in range(nb_data):
      # p-value of test input for each class
      p_value = np.zeros(self.nb_classes, dtype=np.float32)

      for class_id in range(self.nb_classes):
        # p-value of (test point, candidate label)
        p_value[class_id] = (float(self.nb_cali) - bisect_left(
            self.cali_nonconformity, knns_not_in_class[i, class_id])) / float(self.nb_cali)

      preds_knn[i] = np.argmax(p_value)
      confs[i, preds_knn[i]] = 1. - np.sort(p_value)[-2]
      creds[i, preds_knn[i]] = p_value[preds_knn[i]] #np.max(p_value)
    return preds_knn, confs, creds

  def fprop_np(self, data_np):
    """
    Performs a forward pass through the DkNN on an numpy array of data.
    """
    if not self.calibrated:
      raise ValueError(
          "DkNN needs to be calibrated by calling DkNNModel.calibrate method once before inferring.")
    data_activations = self.get_activations(data_np)
    _, knns_labels = self.find_train_knns(data_activations)
    knns_not_in_class = self.nonconformity(knns_labels)
    _, _, creds = self.preds_conf_cred(knns_not_in_class)
    return creds

  def calibrate(self, cali_data, cali_labels):
    """
    Runs the DkNN on holdout data to calibrate the credibility metric.
    :param cali_data: np array of calibration data.
    :param cali_labels: np vector of calibration labels.
    """
    self.nb_cali = cali_labels.shape[0]
    print('================================================================')
    print("Starting calibration of DkNN...")
    print('Computing activations of the calibration data...')
    start = time.time()
    self.cali_activations = self.get_activations(cali_data)
    end = time.time()
    print('Computation done! Time cost:', end - start, 's.')
    self.cali_labels = cali_labels

    cali_knns_ind, cali_knns_labels = self.find_train_knns(
        self.cali_activations)
    assert all([v.shape == (self.nb_cali, self.neighbors)
                for v in cali_knns_ind.values()])
    assert all([v.shape == (self.nb_cali, self.neighbors)
                for v in cali_knns_labels.values()])

    cali_knns_not_in_class = self.nonconformity(cali_knns_labels)
    cali_knns_not_in_l = np.zeros(self.nb_cali, dtype=np.int32)
    for i in range(self.nb_cali):
      cali_knns_not_in_l[i] = cali_knns_not_in_class[i, cali_labels[i]]
    cali_knns_not_in_l_sorted = np.sort(cali_knns_not_in_l)
    self.cali_nonconformity = np.trim_zeros(cali_knns_not_in_l_sorted, trim='f')
    self.nb_cali = self.cali_nonconformity.shape[0]
    print('nb_cali:', self.nb_cali)
    self.calibrated = True
    end = time.time()
    print("DkNN calibration completed! Time cost:", end - start, 's.')


def plot_reliability_diagram(confidence, labels, filepath):
  """
  Takes in confidence values for predictions and correct
  labels for the data, plots a reliability diagram.
  :param confidence: nb_samples x nb_classes (e.g., output of softmax)
  :param labels: vector of nb_samples
  :param filepath: where to save the diagram
  :return:
  """
  print('----------------------------------------------------------------')
  print('Plotting diagrams...')
  assert len(confidence.shape) == 2
  assert len(labels.shape) == 1
  assert confidence.shape[0] == labels.shape[0]
  if confidence.max() <= 1.:
        # confidence array is output of softmax
    bins_start = [b / 10. for b in xrange(0, 10)]
    bins_end = [b / 10. for b in xrange(1, 11)]
    bins_center = [(b + .5) / 10. for b in xrange(0, 10)]
    preds_conf = np.max(confidence, axis=1)
    preds_l = np.argmax(confidence, axis=1)
  else:
    raise ValueError('Confidence values go above 1.')

  #print(preds_conf.shape, preds_l.shape)

  # Create var for reliability diagram
  # Will contain mean accuracies for each bin
  reliability_diag = []
  num_points = []  # keeps the number of points in each bar

  # Find average accuracy per confidence bin
  for bin_start, bin_end in zip(bins_start, bins_end):
    above = preds_conf >= bin_start
    if bin_end == 1.:
      below = preds_conf <= bin_end
    else:
      below = preds_conf < bin_end
    mask = np.multiply(above, below)
    num_points.append(np.sum(mask))
    bin_mean_acc = max(0, np.mean(preds_l[mask] == labels[mask]))
    reliability_diag.append(bin_mean_acc)

  # Plot diagram
  assert len(reliability_diag) == len(bins_center)
  # print(reliability_diag)
  # print(bins_center)
  # print(num_points)
  fig, ax1 = plt.subplots()
  _ = ax1.bar(bins_center, reliability_diag, width=.1, alpha=0.8)
  plt.xlim([0, 1.])
  ax1.set_ylim([0, 1.])

  ax2 = ax1.twinx()
  #print(sum(num_points))
  ax2.plot(bins_center, num_points, color='r', linestyle='-', linewidth=7.0)
  ax2.set_ylabel('Number of points in the data', fontsize=16, color='r')

  if len(tf.where(confidence[0] != 0.)) == 1: #np.argwhere
    # This is a DkNN diagram
    ax1.set_xlabel('Prediction Credibility', fontsize=16)
  else:
    # This is a softmax diagram
    ax1.set_xlabel('Prediction Confidence', fontsize=16)
  ax1.set_ylabel('Prediction Accuracy', fontsize=16)
  ax1.tick_params(axis='both', labelsize=14)
  ax2.tick_params(axis='both', labelsize=14, colors='r')
  fig.tight_layout()
  plt.savefig(filepath, bbox_inches='tight')
  print('Saving reliability diagram at: ' + str(filepath))
