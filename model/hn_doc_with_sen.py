#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())
import numpy as np

from utils.config import *
from utils.data_helper import load_w2v, load_inputs_document, load_word2id, batch_index
from newbie_nn.nn_layer import bi_dynamic_rnn, softmax_layer
from newbie_nn.att_layer import mlp_attention_layer


class HN_DOC_WITH_SEN(object):

    def __init__(self):
        self.config = FLAGS

        self.add_placeholder()
        inputs = self.add_embedding()
        self.sen_logits, self.doc_logits = self.create_model(inputs)
        self.predict_prob = tf.nn.softmax(self.doc_logits)
        self.load_data()
        self.sen_loss, self.doc_loss = self.add_loss(self.sen_logits, self.doc_logits)
        self.accuracy, self.accuracy_num = self.add_accuracy(self.doc_logits)
        self.train_op = self.add_train_op(self.sen_loss, self.doc_loss)

    def add_placeholder(self):
        self.x = tf.placeholder(tf.int32, [None, self.config.max_doc_len, self.config.max_sentence_len])
        self.doc_y = tf.placeholder(tf.float32, [None, self.config.n_class])
        self.sen_len = tf.placeholder(tf.int32, [None, self.config.max_doc_len])
        self.doc_len = tf.placeholder(tf.int32, [None])
        self.sen_y = tf.placeholder(tf.float32, [None, self.config.max_doc_len, 3])
        self.keep_prob1 = tf.placeholder(tf.float32)
        self.keep_prob2 = tf.placeholder(tf.float32)

    def add_embedding(self):
        if self.config.pre_trained == 'yes':
            self.word2id, w2v = load_w2v(self.config.embedding_file, self.config.embedding_dim, True)
        else:
            self.word2id = load_word2id(self.config.word2id_file)
            self.vocab_size = len(self.word2id) + 1
            w2v = tf.random_uniform([self.vocab_size, self.config.embedding_dim], -1.0, 1.0)
        if self.config.embedding_type == 'static':
            self.embedding = tf.constant(w2v, dtype=tf.float32, name='word_embedding')
        else:
            self.embedding = tf.Variable(w2v, dtype=tf.float32, name='word_embedding')
        inputs = tf.nn.embedding_lookup(self.embedding, self.x)
        return inputs

    def load_data(self):
        self.train_x, self.train_sen_len, self.train_doc_len, self.train_sen_y, self.train_doc_y = load_inputs_document(
            self.config.train_file, self.word2id, self.config.max_sentence_len, self.config.max_doc_len)
        self.test_x, self.test_sen_len, self.test_doc_len, self.test_sen_y, self.test_doc_y = load_inputs_document(
            self.config.test_file, self.word2id, self.config.max_sentence_len, self.config.max_doc_len)
        self.val_x, self.val_sen_len, self.val_doc_len, self.val_sen_y, self.val_doc_y = load_inputs_document(
            self.config.val_file, self.word2id, self.config.max_sentence_len, self.config.max_doc_len)

    def create_feed_dict(self, x_batch, sen_len_batch, doc_len_batch, sen_y_batch, y_batch=None):
        if y_batch is None:
            holder_list = [self.x, self.sen_len, self.doc_len, self.sen_y]
            feed_list = [x_batch, sen_len_batch, doc_len_batch, sen_y_batch]
        else:
            holder_list = [self.x, self.sen_len, self.doc_len, self.sen_y, self.y, self.keep_prob1, self.keep_prob2]
            feed_list = [x_batch, sen_len_batch, doc_len_batch, sen_y_batch, y_batch, self.config.keep_prob1, self.config.keep_prob2]
        return dict(zip(holder_list, feed_list))

    def create_model(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)
        cell = tf.contrib.rnn.LSTMCell
        # word to sentence
        sen_len = tf.reshape(self.sen_len, [-1])
        inputs = tf.reshape(inputs, [-1, self.config.max_sentence_len, self.config.embedding_dim])
        hiddens_sen = bi_dynamic_rnn(cell, inputs, self.config.n_hidden, sen_len, self.config.max_sentence_len, 'sentence', 'all')
        alpha_sen = mlp_attention_layer(hiddens_sen, sen_len, 2 * self.config.n_hidden, self.config.l2_reg, self.config.random_base, 1)
        outputs_sen = tf.matmul(alpha_sen, hiddens_sen)

        sen_logits = softmax_layer(outputs_sen, 2 * self.config.n_hidden, self.config.random_base, self.keep_prob2, self.config.l2_reg, 3)
        mask = tf.reshape(tf.cast(tf.sequence_mask(tf.constant([2] * batch_size), 3), tf.float32), tf.shape(sen_logits))
        tmp = sen_logits * mask
        alpha_sen = tf.reshape(tf.reduce_max(tmp, -1), [-1, 1, self.config.max_doc_len])

        outputs_sen = tf.reshape(outputs_sen, [-1, self.config.max_doc_len, 2 * self.config.n_hidden])
        outputs_doc = tf.matmul(alpha_sen, outputs_sen)
        # sentence to doc
        # hiddens_doc = bi_dynamic_rnn(cell, outputs_sen, self.config.n_hidden, self.doc_len, self.config.max_doc_len, 'doc', 'all')
        # alpha_doc = mlp_attention_layer(hiddens_doc, self.doc_len, 2 * self.config.n_hidden, self.config.l2_reg, self.config.random_base, 2)
        # outputs_doc = tf.reshape(tf.matmul(alpha_doc, hiddens_doc), [-1, 2 * self.config.n_hidden])

        logits = softmax_layer(outputs_doc, 2 * self.config.n_hidden, self.config.random_base, self.keep_prob2, self.config.l2_reg, self.config.n_class)
        return sen_logits, logits

    def add_loss(self, sen_scores, doc_scores):
        sen_loss = tf.nn.softmax_cross_entropy_with_logits(logits=sen_scores, labels=self.sen_y)
        doc_loss = tf.nn.softmax_cross_entropy_with_logits(logits=doc_scores, labels=self.doc_y)
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.reduce_mean(doc_loss) + sum(reg_loss)
        return sen_loss, loss

    def add_accuracy(self, scores):
        correct_predicts = tf.equal(tf.argmax(scores, 1), tf.argmax(self.y, 1))
        accuracy_num = tf.reduce_sum(tf.cast(correct_predicts, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_predicts, tf.float32), name='accuracy')
        return accuracy, accuracy_num

    def add_train_op(self, sen_loss, doc_loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.config.lr, global_step, self.config.decay_steps,
                                             self.config.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(self.lr)
        train_op1 = optimizer.minimize(sen_loss)
        train_op2 = optimizer.minimize(doc_loss, global_step=global_step)
        return train_op2

    def run_op(self, sess, op, data_x, sen_len, doc_len, sen_y, doc_y=None):
        res_list = []
        len_list = []
        for indices in batch_index(len(data_x), self.config.batch_size, 1, False, False):
            if doc_y is not None:
                feed_dict = self.create_feed_dict(data_x[indices], sen_len[indices], doc_len[indices], sen_y[indices], doc_y[indices])
            else:
                feed_dict = self.create_feed_dict(data_x[indices], sen_len[indices], doc_len[indices], sen_y[indices])
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

    def run_epoch(self, sess, verbose=10):
        total_loss = []
        total_acc_num = []
        total_num = []
        for step, indices in enumerate(batch_index(len(self.doc_y), self.config.batch_size, 1)):
            feed_dict = self.create_feed_dict(self.train_x[indices], self.train_sen_len[indices],
                                              self.train_doc_len[indices], self.train_sen_y[indices],
                                              self.train_doc_y[indices])
            _, loss, acc_num, lr = sess.run([self.train_op, self.doc_loss, self.accuracy_num, self.lr], feed_dict=feed_dict)
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


def test_case(sess, classifier, data_x, sen_len, doc_len, sen_y, doc_y):
    loss = classifier.run_op(sess, classifier.doc_loss, data_x, sen_len, doc_len, sen_y, doc_y)
    acc_num = classifier.run_op(sess, classifier.accuracy_num, data_x, sen_len, doc_len, sen_y, doc_y)
    return acc_num * 1.0 / len(doc_y), loss


def train_run(_):
    sys.stdout.write('Training start:\n')
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            classifier = HN_DOC_WITH_SEN()
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            best_accuracy = 0
            best_val_epoch = 0
            val_x, val_sen_len, val_doc_len, val_sen_y, val_doc_y = \
                classifier.val_x, classifier.val_sen_len, classifier.val_doc_len, classifier.sen_y, classifier.doc_y
            for epoch in range(classifier.config.n_iter):
                print '=' * 20 + 'Epoch ', epoch, '=' * 20
                loss, acc = classifier.run_epoch(sess)
                print '[INFO] Mean loss = {}, mean acc = {}'.format(loss, acc)
                print '=' * 50
                val_accuracy, loss = test_case(sess, classifier, val_x, val_sen_len, val_doc_len, val_sen_y, val_doc_y)
                print '[INFO] test loss: {}, test acc: {}'.format(loss, val_accuracy)
                if best_accuracy < val_accuracy:
                    best_accuracy = val_accuracy
                    best_val_epoch = epoch
                    if not os.path.exists(classifier.config.weights_save_path):
                        os.makedirs(classifier.config.weights_save_path)
                    saver.save(sess, classifier.config.weights_save_path + '/weights')
                if epoch - best_val_epoch > classifier.config.early_stopping:
                    print 'Normal early stop!'
                    break
            print 'Best acc = {}'.format(best_accuracy)
    print 'Training complete!'


if __name__ == '__main__':
    tf.app.run(train_run)







