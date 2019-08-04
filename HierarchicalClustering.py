import tensorflow as tf
import numpy as np
import time
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from scipy.cluster.hierarchy import fcluster


class HierarchicalClustering(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, verbose=False):
        tf.reset_default_graph()
        self.verbose = verbose

    def fit(self, X):
        '''
        fit
        :return: linkage matrix
        '''
        self.__set_stuff(X)
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("graph/", sess.graph)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            sess.run(self.init)
            start = time.time()
            for i in range(self.steps - 1):
                with tf.name_scope("step_%d"%i) as scope:
                    print("step", i)
                    self.distances = self._distance(self.new_data)
                    self.n = self.distances.shape[0]
                    ##remove diagonal
                    with tf.name_scope('find_minimum_distances') as scope:
                        self.nddistances = tf.reshape(
                            tf.boolean_mask(self.distances,
                                            tf.logical_not(tf.equal(self.distances, tf.zeros_like(self.distances, name="zeros"), name="equal"), name="not")),
                            (self.n, self.n - 1))  # 1 is diagonal
                        self.actual_minimums = \
                            tf.sort(tf.sort(tf.where(tf.equal(tf.reduce_min(self.nddistances), self.distances), name="minimum_positions"), axis=1),
                                    axis=0,
                                    name="assignemts")[0]
                        self.original_cluster_indexes = tf.gather(self.assignments, tf.cast(self.actual_minimums, tf.int64),
                                                              name="correct_assignemts")
                    with tf.name_scope('merging') as scope:
                        if self.verbose:
                            print("merging..", self.original_cluster_indexes.eval())
                        self.min_distance = tf.cast(self.distances[self.actual_minimums[0]][self.actual_minimums[1]],
                                                    tf.float64,
                                                    name="minimum_distance")
                        ##mean position of new cluster
                        self.new_pos = self._get_linkage(self.new_data[self.actual_minimums[0]],
                                                   self.new_data[self.actual_minimums[1]], name="linkage")
                        self.assignments = np.delete(self.assignments, self.actual_minimums.eval())
                        self.n_actual_clusters -= 2
                        self.data = tf.concat([self.data, [self.new_pos]], axis=0, name="updated_data")
                        self.assignments = np.concatenate([self.assignments, [self.n_max_clusters]], axis=0)  ##new cluster
                        self.current_size = np.sum(self.sizes[np.array(self.original_cluster_indexes.eval()).astype(int)])
                        self.sizes = np.concatenate([self.sizes, [self.current_size]])
                    with tf.name_scope('update') as scope:
                        self.n_actual_clusters += 1
                        if self.verbose:
                            print("current clusters..", self.assignments)
                            print("current sizes..", self.sizes)
                        self.new_data = tf.Variable(tf.zeros((self.n_actual_clusters, self.data.shape[1]), dtype=tf.float64, name="zeros"),
                                                    dtype=tf.float64, name="new_data")
                        tf.assign(self.new_data, tf.gather(self.data, tf.cast(self.assignments, tf.int64)),
                                  validate_shape=False,name="assign_new_data").eval()
                        # new_data = tf.reshape(new_data, (n_actual_clusters,data.shape[1]))
                        if self.verbose:
                            print("data..", self.new_data.eval(), " with shape..", self.new_data.shape)
                        self.n_max_clusters = self.n_max_clusters + 1
                    with tf.name_scope('Z_matrix') as scope:
                        self.Z.append(
                            sess.run(tf.stack([self.original_cluster_indexes[0], self.original_cluster_indexes[1], self.min_distance,
                                      self.current_size],
                                     0, name="Z_linkage_matrix"),
                                     options=run_options,
                                      run_metadata=run_metadata))
                    writer.add_run_metadata(run_metadata, 'step%d' % i)
            self.Z = np.array(self.Z).astype(float)
            print("Z..", self.Z)
        print("runned in..", time.time() - start, " seconds")
        writer.close()
        return self.Z

    def fit_predict(self, X, y=None):
        '''
        Fit and predict data
        :return:
        '''
        self.fit(X)
        return fcluster(self.Z, t=self.t)

    def __set_stuff(self, data_list):
        with tf.name_scope('initializer') as scope:
            self.data = tf.constant(data_list, dtype=tf.float64, name="data")
            self.new_data = tf.Variable(self.data, name="data_variable")  #variable should change shape
            self.npoints = self.data.shape[0].value
            self.steps = self.npoints
            self.n_max_clusters = self.npoints  #max number
            self.n_actual_clusters = self.npoints  #currently considered
            self.assignments = np.linspace(0., self.npoints - 1, self.npoints)
            self.sizes = np.ones_like(self.assignments)
            self.orig_shape = self.data.shape[0]
            self.Z = []
            self.t = 0.8  # clustering param
            self.init = tf.global_variables_initializer()

    def _distance(self, data, name="distances"):
        return tf.map_fn(lambda A: tf.map_fn(lambda B: tf.norm(A - B), data), data, name=name)

    def _get_linkage(self, first, second, name="linkage"):
        return tf.reduce_mean([first, second], axis=0, name=name)

    def _drop_row(self, data, row=tf.constant(0, dtype=tf.int64), name="drop_row"):
        return tf.concat([tf.slice(data, [tf.constant(0, dtype=tf.int64), 0], [row, -1]),
                          tf.slice(data, [row + tf.constant(1, dtype=tf.int64), 0], [-1, -1])], axis=0, name=name)
