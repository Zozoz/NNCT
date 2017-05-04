#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from utils.config import *
from utils.data_helper import load_w2v, load_inputs_sentence, batch_index
from newbie_nn.nn_layer import cnn_layer, softmax_layer


class CNN_Sentence(object):

    def __init__(self, filter_list, filter_num, vocab_size=10000):
        self.config = FLAGS
        self.filter_list = filter_list
        self.filter_num = filter_num
        self.vocab_size = vocab_size

        self.add_placeholder()
        inputs = self.add_embedding()
        self.logits = self.create_model(inputs)
        self.predict_prob = tf.nn.softmax(self.logits)
        self.load_data()
        self.loss = self.add_loss(self.logits)
        self.accuracy, self.accuracy_num = self.add_accuracy(self.logits)
        self.train_op = self.add_train_op(self.loss)

    def add_placeholder(self):
        self.x = tf.placeholder(tf.int32, [None, self.config.max_sentence_len], 'input_data')
        self.y = tf.placeholder(tf.float32, [None, self.config.n_class], 'output_data')
        self.sen_len = tf.placeholder(tf.int32, [None], 'sentence_len')
        self.keep_prob1 = tf.placeholder(tf.float32)
        self.keep_prob2 = tf.placeholder(tf.float32)

    def add_embedding(self):
        if self.config.pre_trained == 'yes':
            self.word2id, w2v = load_w2v(self.config.embedding_file, self.config.embedding_dim, True)
            self.embedding = tf.constant(w2v, dtype=tf.float32, name='word_embedding')
        else:
            self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.config.embedding_dim], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(self.embedding, self.x)
        inputs = tf.expand_dims(inputs, -1)
        return inputs

    def load_data(self):
        self.train_x, self.train_sen_len, self.train_y = load_inputs_sentence(self.config.train_file, self.word2id,
                                                                              self.config.max_sentence_len)
        self.test_x, self.test_sen_len, self.test_y = load_inputs_sentence(self.config.test_file, self.word2id,
                                                                           self.config.max_sentence_len)
        self.val_x, self.val_sen_len, self.val_y = load_inputs_sentence(self.config.val_file, self.word2id,
                                                                        self.config.max_sentence_len)

    def add_cnn_layer(self, inputs):
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)
        pooling_outputs = []
        for i, filter_size in enumerate(self.filter_list):
            filter_shape = [filter_size, self.config.embedding_dim, 1, self.filter_num]
            # Convolution layer
            conv = cnn_layer(inputs, filter_shape, [1, 1, 1, 1], 'VALID', self.config.random_base,
                             self.config.l2_reg, tf.nn.relu, str(i))
            # Pooling layer
            pooling = tf.nn.max_pool(conv, ksize=[1, self.config.max_sentence_len - filter_size + 1, 1, 1],
                                     strides=[1, 1, 1, 1], padding='VALID', name='pooling')
            pooling_outputs.append(pooling)
        # combine all pooling outputs
        hiddens = tf.concat(pooling_outputs, 3)
        hiddens_flat = tf.reshape(hiddens, [-1, self.filter_num * len(self.filter_list)])
        return hiddens_flat

    def add_softmax_layer(self, inputs):
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob2)
        return softmax_layer(inputs, self.filter_num*len(self.filter_list), self.config.random_base,
                             self.keep_prob2, self.config.l2_reg, self.config.n_class)

    def add_loss(self, scores):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.y)
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.reduce_mean(loss) + sum(reg_loss)
        return loss

    def add_accuracy(self, scores):
        correct_predicts = tf.equal(tf.argmax(scores, 1), tf.argmax(self.y, 1))
        accuracy_num = tf.reduce_sum(tf.cast(correct_predicts, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_predicts, tf.float32), name='accuracy')
        return accuracy, accuracy_num

    def create_model(self, inputs):
        hiddens = self.add_cnn_layer(inputs)
        return self.add_softmax_layer(hiddens)

    def add_train_op(self, loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.config.lr, global_step, self.config.decay_steps,
                                             self.config.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(self.lr)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def create_feed_dict(self, x_batch, sen_len_batch, y_batch=None):
        if y_batch is None:
            holder_list = [self.x, self.sen_len]
            feed_list = [x_batch, sen_len_batch]
        else:
            holder_list = [self.x, self.sen_len, self.y, self.keep_prob1, self.keep_prob2]
            feed_list = [x_batch, sen_len_batch, y_batch, self.config.keep_prob1, self.config.keep_prob2]
        return dict(zip(holder_list, feed_list))

    def run_epoch(self, sess, data_x, data_len, data_y, verbose=10):
        total_loss = []
        total_acc_num = []
        total_num = []
        for step, indices in enumerate(batch_index(len(data_y), self.config.batch_size, 1)):
            feed_dict = self.create_feed_dict(data_x[indices], data_len[indices], data_y[indices])
            _, loss, acc_num, lr = sess.run([self.train_op, self.loss, self.accuracy_num, self.lr], feed_dict=feed_dict)
            total_loss.append(loss)
            total_acc_num.append(acc_num)
            total_num.append(len(indices))
            if verbose and step % verbose == 0:
                print '\n[INFO] {} : loss = {}, acc = {}, lr = {}'.format(
                    step,
                    np.mean(total_loss[-verbose:]),
                    sum(total_acc_num[-verbose:]) * 1.0 / sum(total_num[-verbose:]),
                    lr
                )
        return np.mean(total_loss), sum(total_acc_num) * 1.0 / sum(total_num)

    def run_op(self, sess, op, data_x, data_len, data_y=None):
        res_list = []
        len_list = []
        for indices in batch_index(len(data_x), self.config.batch_size, 1, False, False):
            if data_y is not None:
                feed_dict = self.create_feed_dict(data_x[indices], data_len[indices], data_y[indices])
            else:
                feed_dict = self.create_feed_dict(data_x[indices], data_len[indices])
            res = sess.run(op, feed_dict=feed_dict)
            res_list.append(res)
            len_list.append(len(indices))
        if type(res_list[0]) is list:
            res = np.concatenate(res_list, axis=0)
        elif op is self.accuracy_num:
            res = sum(res_list)
        else:
            res = sum(res_list) * 1.0 / len(len_list)
        return res


def test_case(sess, classifier, data_x, data_len, data_y):
    loss = classifier.run_op(sess, classifier.loss, data_x, data_len, data_y)
    acc_num = classifier.run_op(sess, classifier.accuracy_num, data_x, data_len, data_y)
    return acc_num * 1.0 / len(data_y), loss


def calculate_metrics(sess, classifier, data_x, data_len, data_y):
    pred_prob = classifier.run_op(sess, classifier.predict_prob, data_x, data_len, data_y)
    p = precision_score(np.argmax(data_y, 1), np.argmax(pred_prob, 1), average=None)
    r = recall_score(np.argmax(data_y, 1), np.argmax(pred_prob, 1), average=None)
    f1 = f1_score(np.argmax(data_y, 1), np.argmax(pred_prob, 1), average=None)
    print 'p:', p, 'avg=', sum(p) / len(data_y[0])
    print 'r:', r, 'avg=', sum(r) / len(data_y[0])
    print 'f1:', f1, 'avg=', sum(f1) / len(data_y[0])


def train_run(_):
    sys.stdout.write('Training start:\n')
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            classifier = CNN_Sentence([3, 4, 5], 100)
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            best_accuracy = 0
            best_val_epoch = 0
            train_x, train_y, train_sen_len = classifier.train_x, classifier.train_y, classifier.train_sen_len
            val_x, val_y, val_sen_len = classifier.val_x, classifier.val_y, classifier.val_sen_len
            for epoch in range(classifier.config.n_iter):
                print '=' * 20 + 'Epoch ', epoch, '=' * 20
                loss, acc = classifier.run_epoch(sess, train_x, train_sen_len, train_y)
                print '[INFO] Mean loss = {}, mean acc = {}'.format(loss, acc)
                print '=' * 50
                val_accuracy, loss = test_case(sess, classifier, val_x, val_sen_len, val_y)
                print '[INFO] test loss: {}, test acc: {}'.format(loss, val_accuracy)
                if best_accuracy > val_accuracy:
                    best_accuracy = val_accuracy
                    best_val_epoch = epoch
                    if not os.path.exists(classifier.config.weights_save_path):
                        os.makedirs(classifier.config.weights_save_path)
                    saver.save(sess, classifier.config.weight_save_path + '/weights')
                if epoch - best_val_epoch > classifier.config.early_stopping:
                    print 'Normal early stop!'
                    break
    print 'Training complete!'


if __name__ == '__main__':
    tf.app.run(train_run)











