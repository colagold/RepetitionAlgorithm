import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter
from models.predictor import predict
from evaluation.metrics import evaluate


class RefineLabelAE(object):
    def __init__(self,
                 input_dim,
                 embed_dim,
                 batch_size,
                 lamb,
                 confidence,
                 norm_init_X,
                 norm_init_Y,
                 norm_init_xBias,
                 norm_init_yBias,
                 unif_init_X,
                 unif_init_Y,
                 unif_init_xBias,
                 unif_init_yBias,
                 gpu_on,
                 **unused):
        self.input_dim = self.output_dim = input_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.lamb = lamb
        self.confidence = confidence
        self.norm_init_X = norm_init_X
        self.norm_init_Y = norm_init_Y
        self.norm_init_xBias = norm_init_xBias
        self.norm_init_yBias = norm_init_yBias
        self.unif_init_X = unif_init_X
        self.unif_init_Y = unif_init_Y
        self.unif_init_xBias = unif_init_xBias
        self.unif_init_yBias = unif_init_yBias
        self.gpu_on = gpu_on
        self.get_graph()
        if self.gpu_on:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def get_graph(self):
        self.inputs = tf.placeholder(tf.float32, (None, self.input_dim))
        self.sample_idx = tf.placeholder(tf.float32, (None, self.input_dim))

        self.norm_encode_weights = tf.get_variable('norm_encode_weights', initializer=self.norm_init_X)
        self.norm_encode_bias = tf.get_variable('norm_encode_bias', initializer=self.norm_init_xBias)
        self.norm_decode_weights = tf.get_variable('norm_decode_weights', initializer=self.norm_init_Y)
        self.norm_decode_bias = tf.get_variable('norm_decode_bias', initializer=self.norm_init_yBias)

        self.unif_encode_weights = tf.get_variable('unif_encode_weights', initializer=self.unif_init_X, trainable=False)
        self.unif_encode_bias = tf.get_variable('unif_encode_bias', initializer=self.unif_init_xBias, trainable=False)
        self.unif_decode_weights = tf.get_variable('unif_decode_weights', initializer=self.unif_init_Y, trainable=False)
        self.unif_decode_bias = tf.get_variable('unif_decode_bias', initializer=self.unif_init_yBias, trainable=False)

        mask = tf.where(tf.not_equal(self.inputs, 0), tf.ones(tf.shape(self.inputs)), tf.zeros(tf.shape(self.inputs)))
        with tf.variable_scope("refine_label"):
            self.unif_encoded = tf.nn.relu(tf.matmul(self.inputs, self.unif_encode_weights) + self.unif_encode_bias)

            self.unif_prediction = tf.matmul(self.unif_encoded, self.unif_decode_weights) + self.unif_decode_bias

            predict_label = self.unif_prediction * mask

            self.predict_label = (predict_label - tf.reduce_min(predict_label, 1, keepdims=True)) / (
                    tf.reduce_max(predict_label, 1, keepdims=True) - tf.reduce_min(predict_label, 1, keepdims=True))

            self.refined_label = self.inputs + self.confidence * self.predict_label

        with tf.variable_scope('loss'):
            self.norm_encoded = tf.nn.relu(tf.matmul(self.inputs, self.norm_encode_weights) + self.norm_encode_bias)

            self.norm_prediction = tf.matmul(self.norm_encoded, self.norm_decode_weights) + self.norm_decode_bias

            l2_loss = tf.nn.l2_loss(self.norm_encode_weights) + tf.nn.l2_loss(self.norm_decode_weights)

            mf_loss = tf.square(self.refined_label - self.norm_prediction * mask)

            self.loss = tf.reduce_mean(mf_loss) + self.lamb*tf.reduce_mean(l2_loss)

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

    @staticmethod
    def get_batches(rating_matrix, batch_size):
        remaining_size = rating_matrix.shape[0]
        batch_index = 0
        batches = []
        while remaining_size > 0:
            if remaining_size < batch_size:
                batches.append(rating_matrix[batch_index*batch_size:])
            else:
                batches.append(rating_matrix[batch_index*batch_size:(batch_index+1)*batch_size])
            batch_index += 1
            remaining_size -= batch_size
        return batches

    def train_model(self, rating_matrix, matrix_valid, epoch, metric_names):
        batches = self.get_batches(rating_matrix, self.batch_size)

        # Training
        best_AUC, best_RQ, best_X, best_xBias, best_Y, best_yBias = 0, [], [], [], [], []
        for i in tqdm(range(epoch)):
            for step in range(len(batches)):
                feed_dict = {self.inputs: batches[step].todense()}
                _ = self.sess.run([self.optimizer], feed_dict=feed_dict)

            RQ, X, xBias = self.get_RQ(rating_matrix)
            Y = self.get_Y()
            yBias = self.get_yBias()
            prediction = predict(matrix_U=RQ, matrix_V=Y.T, matrix_Valid=matrix_valid, bias=yBias, gpu=self.gpu_on)
            result = evaluate(prediction, matrix_valid, metric_names, gpu=self.gpu_on)

            if result['AUC'][0] > best_AUC:
                best_AUC = result['AUC'][0]
                best_RQ, best_X, best_xBias, best_Y, best_yBias = RQ, X, xBias, Y, yBias

        return best_RQ, best_X, best_xBias, best_Y, best_yBias

    def get_RQ(self, rating_matrix):
        batches = self.get_batches(rating_matrix, self.batch_size)
        RQ = []
        for step in range(len(batches)):
            feed_dict = {self.inputs: batches[step].todense()}
            encoded = self.sess.run(self.norm_encoded, feed_dict=feed_dict)
            RQ.append(encoded)

        return np.vstack(RQ), self.sess.run(self.norm_encode_weights), self.sess.run(self.norm_encode_bias)

    def get_Y(self):
        return self.sess.run(self.norm_decode_weights)

    def get_yBias(self):
        return self.sess.run(self.norm_decode_bias)


def refinelabelae(matrix_train, matrix_valid, iteration=100, lam=0.01, confidence=0.9, rank=50, seed=0, batch_size=256,
                  dataset=None, gpu_on=True, **unused):
    progress = WorkSplitter()

    progress.section("RefineLabelAE: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("RefineLabelAE: Load the variables trained on S_c/S_t")

    norm_X = np.load('latent/' + dataset + 'X_AutoRec_200.npy')
    norm_xBias = np.load('latent/' + dataset + 'xB_AutoRec_200.npy')
    norm_Y = np.load('latent/' + dataset + 'Y_AutoRec_200.npy')
    norm_yBias = np.load('latent/' + dataset + 'yB_AutoRec_200.npy')

    unif_X = np.load('latent/' + dataset + 'unif_X_AutoRec_200.npy')
    unif_xBias = np.load('latent/' + dataset + 'unif_xB_AutoRec_200.npy')
    unif_Y = np.load('latent/' + dataset + 'unif_Y_AutoRec_200.npy')
    unif_yBias = np.load('latent/' + dataset + 'unif_yB_AutoRec_200.npy')

    progress.section("RefineLabelAE: Training")
    m, n = matrix_train.shape
    model = RefineLabelAE(n, rank, lamb=lam, confidence=confidence, batch_size=batch_size, gpu_on=gpu_on,
                          norm_init_X=norm_X, norm_init_Y=norm_Y, norm_init_xBias=norm_xBias, norm_init_yBias=norm_yBias,
                          unif_init_X=unif_X, unif_init_Y=unif_Y, unif_init_xBias=unif_xBias, unif_init_yBias=unif_yBias)
    metric_names = ['NLL', 'AUC']
    RQ, X, xBias, Y, yBias = model.train_model(matrix_train, matrix_valid, iteration, metric_names)

    model.sess.close()
    tf.reset_default_graph()

    return RQ, X, xBias, Y, yBias