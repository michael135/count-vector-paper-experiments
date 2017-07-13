import os
import numpy as np
from itertools import izip_longest
from sklearn.metrics import recall_score, precision_score, accuracy_score
from tensorflow.contrib import rnn
import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"]="0" # train on a specific GPU


class DataLoader(object):
    """
    Handling the data for training the network.
    """
    def __init__(self, data, labels=None):
        self.train_data = data['data_train']
        self.train_label = data['label_train']
        self.test_data = data['data_test']
        self.test_label = data['label_test']

    @staticmethod
    def change_scale(a, t):
        size = a.shape[1]
        b = np.zeros([a.shape[0], size / t + bool(size % t), a.shape[2]])
        for i in range(a.shape[0]):
            for j in range(0, size, t):
                b[i, j / t, :] = np.sum(a[i, j: j + t, :], axis=0)
        return b

    @staticmethod
    def normalize_per_line(a):
        for j in range(a.shape[0]):
            for i in range(a.shape[1]):
                a[j, i, :] /= np.sum(a[j, i, :])
        return a

    def scale_normalize(self, a, t):
        a_out = self.normalize_per_line(self.change_scale(a, t))
        z_out = a_out.shape[1] * np.ones(a_out.shape[0]).astype(np.int32)
        return a_out, z_out

    def loader(self, batch_size, phase='train'):
        span = range(len(getattr(self, phase + '_data')))
        if phase == 'train':
            np.random.shuffle(span)
        args = [iter(span)] * batch_size
        for indexes in izip_longest(*args):
            x, y = np.asarray([getattr(self, phase + '_data')[i] for i in indexes if i is not None]), \
                np.asarray([getattr(self, phase + '_label')[i] for i in indexes if i is not None])
            x, z = self.scale_normalize(x, 1)
            yield x, y, z 


def evaluation(true_labels, prediction):
    predicted_labels = [np.argmax(class_probs) for class_probs in prediction]
    recall = recall_score(true_labels, predicted_labels, average='macro')
    precision = precision_score(true_labels, predicted_labels, average='macro')
    return recall, precision


def create_net(x, z, hidden_units, weights, biases, keep_prob, bidirectional, concat):
    x = tf.nn.dropout(x, keep_prob)
    lstm_fw = rnn.BasicLSTMCell(hidden_units, forget_bias=1.0)
    if bidirectional:
        lstm_bw = rnn.BasicLSTMCell(hidden_units, forget_bias=1.0)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, x, sequence_length=z, dtype=tf.float32)
    else:
        outputs, _ = tf.nn.dynamic_rnn(lstm_fw, x, sequence_length=z, dtype=tf.float32)

    if bidirectional and concat:
        output = tf.concat([outputs[0][:, -1], outputs[1][:, -1]], axis=1)
    elif bidirectional:
        output = (outputs[0][:, -1] + outputs[1][:, -1]) / 2.
    else:
        output = outputs[:, -1]
    return tf.matmul(output, weights) + biases


def create_inception_like(x1, x2, x3, z1, z2, z3, hidden_units, weights, biases, keep_prob, concat):
    x1 = tf.nn.dropout(x1, keep_prob)
    x2 = tf.nn.dropout(x2, keep_prob)
    x3 = tf.nn.dropout(x3, keep_prob)
    lstm_fw = rnn.BasicLSTMCell(hidden_units, forget_bias=1.0)
    lstm_bw = rnn.BasicLSTMCell(hidden_units, forget_bias=1.0)
    with tf.variable_scope('siamese_network') as scope:
        with tf.name_scope('net_1'):
            outputs1, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, x1, sequence_length=z1, dtype=tf.float32)
        with tf.name_scope('net_2'):
            scope.reuse_variables()
            outputs2, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, x2, sequence_length=z2, dtype=tf.float32)
        with tf.name_scope('net_3'):
            scope.reuse_variables()
            outputs3, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, x3, sequence_length=z3, dtype=tf.float32)

    outputs = [outputs1[0][:, -1], outputs1[1][:, -1], outputs2[0][:, -1],
                    outputs2[1][:, -1], outputs3[0][:, -1], outputs3[1][:, -1]]
    if concat:
        output = tf.concat(outputs, axis=1)
    else:
        output = tf.mean(output)
    return tf.matmul(output, weights) + biases


def train(data, lr=0.001, epochs=50, batch_train=1024, batch_test=1024,
            hidden_units=64, beta=5e-4, dropout=1., inception=False, concat=False, bidirectional=False):
    tf.reset_default_graph()
    
    n_input = data['data_train'][0].shape[1]
    n_classes = len(data['label_train'][0])
    n_data = len(data['data_train'])
    
    data_load = DataLoader(data=data)

    # Create placeholders
    x1 = tf.placeholder("float32", [None, None, n_input], name='x1')
    z1 = tf.placeholder(tf.int32, [None])
    x2 = tf.placeholder("float32", [None, None, n_input], name='x2')
    z2 = tf.placeholder(tf.int32, [None])
    x3 = tf.placeholder("float32", [None, None, n_input], name='x3')
    z3 = tf.placeholder(tf.int32, [None])
    y = tf.placeholder("float32", [None, n_classes], name='y')
    keep_prob = tf.placeholder(tf.float32)

    if inception and concat:
        weights = tf.Variable(tf.random_normal([2 * 3 * hidden_units, n_classes]))
    elif bidirectional and concat:
        weights = tf.Variable(tf.random_normal([2 * hidden_units, n_classes]))
    else:
        weights = tf.Variable(tf.random_normal([hidden_units, n_classes]))
    biases = tf.Variable(tf.random_normal([n_classes]))

    if not inception:
        model = create_net(x1, z1, hidden_units, weights, biases, keep_prob, bidirectional, concat)
    else:
        model = create_inception_like(x1, x2, x3, z1, z2, z3, hidden_units, weights, biases, keep_prob, concat)

    l2_regularization =  beta * tf.reduce_sum(tf.nn.l2_loss(weights))
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y)
    cost = tf.reduce_mean(softmax) + l2_regularization
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    loss = []
    acc = []
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        print("--- Network training begins ---")
        for epoch in range(epochs):
            for _, (_x1, _y, _z1) in enumerate(data_load.loader(batch_train, phase='train')):
                if not inception:
                    feed_dict = {x1: _x1, y: _y, z1: _z1, keep_prob: dropout}
                else:
                    _x2, _z2 = data_load.scale_normalize(_x1, t=2)
                    _x3, _z3 = data_load.scale_normalize(_x1, t=3)
                    feed_dict = {x1: _x1, x2: _x2, x3: _x3, z1: _z1, z2: _z2, z3: _z3, y: _y, keep_prob: dropout}

                _, acc_, loss_ = sess.run([optimizer, accuracy, cost], feed_dict=feed_dict)
                loss.append(loss_)
                acc.append(acc_)
            else:
                print('** TRAIN ** [Epoch %i/%i] loss = %.4f, accuracy = %.4f' %
                    (epoch + 1, epochs, np.mean(loss), np.mean(acc)))

                loss, acc = [], []
                for _x1, _y, _z1 in data_load.loader(batch_test, phase='test'):
                    _x2, _z2 = data_load.scale_normalize(_x1, t=2)
                    _x3, _z3 = data_load.scale_normalize(_x1, t=3)
                    feed_dict = {x1: _x1, x2: _x2, x3: _x3, z1: _z1, z2: _z2, z3: _z3, y: _y, keep_prob: 1.}
                    acc_, loss_ = sess.run([accuracy, cost], feed_dict=feed_dict)
                    acc.append(acc_)
                    loss.append(loss_)
                print('** TEST ** [Epoch %i/%i] loss = %.4f, accuracy = %.4f' % (epoch + 1, epochs, np.mean(loss), np.mean(acc)))
                loss, acc = [], []

        print("Optimization Finished!")
        print('Network evaluation begins...')
        true_labels = []
        predictions = []
        for _x1, _y, _z1 in data_load.loader(batch_test, phase='test'):
            if not inception:
                feed_dict={x1: _x1, z1: _x1.shape[1] * np.ones(_x1.shape[0], dtype=np.int), keep_prob: 1.}
            else:
                _x2, _z2 = data_load.scale_normalize(_x1, t=2)
                _x3, _z3 = data_load.scale_normalize(_x1, t=3)
                feed_dict = {x1: _x1, x2: _x2, x3: _x3, z1: _z1, z2: _z2, z3: _z3, keep_prob: 1.}
            
            predictions_ = sess.run(model, feed_dict=feed_dict)
            predictions.extend(predictions_)
            true_labels.extend([np.argmax(label) for label in _y])

        predictions = np.vstack(predictions)
        recall, precision = evaluation(true_labels, predictions)
        print('recall %.3f, precision %.3f' % (recall, precision))
        return {'recall': recall, 'precision': precision, 'dropout': dropout,
                    'hidden_units': hidden_units, 'inception': inception, 'concat': concat, 'bidirectional': bidirectional}
