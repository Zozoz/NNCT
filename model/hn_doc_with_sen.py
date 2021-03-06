#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())
import numpy as np

from utils.config import *
from utils.data_helper import load_w2v, load_inputs_document_sen_3, load_inputs_document_sen_2, load_word2id, batch_index
from newbie_nn.nn_layer import bi_dynamic_rnn, softmax_layer, reduce_mean_with_len, cnn_layer
from newbie_nn.att_layer import mlp_attention_layer, softmax_with_len


class HN_DOC_WITH_SEN(object):

    def __init__(self, filter_list=(3, 4, 5), filter_num=100):
        self.config = FLAGS
        self.filter_list = filter_list
        self.filter_num = filter_num

        self.add_placeholder()
        inputs = self.add_embedding()
        self.sen_logits, self.doc_logits = self.create_model_1(inputs)
        self.load_data()
        self.sen_loss, self.doc_loss = self.add_loss_sep(self.sen_logits, self.doc_logits)
        self.accuracy, self.accuracy_num = self.add_accuracy(self.doc_logits)
        self.sen_acc_num, self.sen_num = self.add_sen_acc(self.sen_logits)
        self.train_op1, self.train_op2 = self.add_train_op(self.sen_loss, self.doc_loss)

    def add_placeholder(self):
        self.x = tf.placeholder(tf.int32, [None, self.config.max_doc_len, self.config.max_sentence_len])
        self.doc_y = tf.placeholder(tf.float32, [None, self.config.n_class])
        self.sen_len = tf.placeholder(tf.int32, [None, self.config.max_doc_len])
        self.doc_len = tf.placeholder(tf.int32, [None])
        self.sen_y = tf.placeholder(tf.float32, [None, self.config.max_doc_len, self.config.n_sen_class])
        self.keep_prob1 = tf.placeholder(tf.float32)
        self.keep_prob2 = tf.placeholder(tf.float32)

    def add_embedding(self):
        if self.config.pre_trained == 'yes':
            self.word2id, w2v = load_w2v(self.config.embedding_file, self.config.embedding_dim, False)
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
        self.train_x, self.train_sen_len, self.train_doc_len, self.train_sen_y, self.train_doc_y = load_inputs_document_sen_3(
            self.config.train_file, self.word2id, self.config.max_sentence_len, self.config.max_doc_len)
        self.test_x, self.test_sen_len, self.test_doc_len, self.test_sen_y, self.test_doc_y = load_inputs_document_sen_3(
            self.config.test_file, self.word2id, self.config.max_sentence_len, self.config.max_doc_len)
        self.val_x, self.val_sen_len, self.val_doc_len, self.val_sen_y, self.val_doc_y = load_inputs_document_sen_3(
            self.config.val_file, self.word2id, self.config.max_sentence_len, self.config.max_doc_len)

    def create_feed_dict(self, x_batch, sen_len_batch, doc_len_batch, sen_y_batch, y_batch=None, kp1=1.0, kp2=1.0):
        if y_batch is None:
            holder_list = [self.x, self.sen_len, self.doc_len, self.sen_y, self.keep_prob1, self.keep_prob2]
            feed_list = [x_batch, sen_len_batch, doc_len_batch, sen_y_batch, kp1, kp2]
        else:
            holder_list = [self.x, self.sen_len, self.doc_len, self.sen_y, self.doc_y, self.keep_prob1, self.keep_prob2]
            feed_list = [x_batch, sen_len_batch, doc_len_batch, sen_y_batch, y_batch, kp1, kp2]
        return dict(zip(holder_list, feed_list))

    def add_bilstm_layer(self, inputs, seq_len, max_len, scope_name='bilstm'):
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)
        cell = tf.contrib.rnn.LSTMCell
        # word to sentence
        seq_len = tf.reshape(seq_len, [-1])
        hiddens_sen = bi_dynamic_rnn(cell, inputs, self.config.n_hidden, seq_len, max_len, scope_name, 'all')
        alpha_sen = mlp_attention_layer(hiddens_sen, seq_len, 2 * self.config.n_hidden, self.config.l2_reg, self.config.random_base, scope_name)
        outputs_sen = tf.squeeze(tf.matmul(alpha_sen, hiddens_sen))
        return outputs_sen

    def add_cnn_layer(self, inputs, scope_name='conv'):
        inputs = tf.expand_dims(inputs, -1)
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)
        pooling_outputs = []
        for i, filter_size in enumerate(self.filter_list):
            filter_shape = [filter_size, self.config.embedding_dim, 1, self.filter_num]
            # Convolution layer
            conv = cnn_layer(inputs, filter_shape, [1, 1, 1, 1], 'VALID', self.config.random_base,
                             self.config.l2_reg, tf.nn.relu, scope_name + str(i))
            # Pooling layer
            pooling = tf.nn.max_pool(conv, ksize=[1, self.config.max_sentence_len - filter_size + 1, 1, 1],
                                     strides=[1, 1, 1, 1], padding='VALID', name='pooling')
            pooling_outputs.append(pooling)
        # combine all pooling outputs
        hiddens = tf.concat(pooling_outputs, 3)
        hiddens_flat = tf.reshape(hiddens, [-1, self.filter_num * len(self.filter_list)])
        return hiddens_flat

    # 映射一层, 组合两层
    def create_model_1(self, inputs):
        inputs = tf.reshape(inputs, [-1, self.config.max_sentence_len, self.config.embedding_dim])

        # outputs_sen = self.add_cnn_layer(inputs)
        # outputs_sen_dim = self.filter_num * len(self.filter_list)

        outputs_sen = self.add_bilstm_layer(inputs, self.sen_len, self.config.max_sentence_len, 'doc_sen')  # doc_sen / sen
        outputs_sen_dim = 2 * self.config.n_hidden

        sen_logits = softmax_layer(outputs_sen, outputs_sen_dim, self.config.random_base, self.keep_prob2, self.config.l2_reg, self.config.n_sen_class, 'sen_softmax_')
        outputs_sen = tf.reshape(outputs_sen, [-1, self.config.max_doc_len, outputs_sen_dim])
        outputs_doc = reduce_mean_with_len(outputs_sen, self.doc_len)

        logits = softmax_layer(outputs_doc, outputs_sen_dim, self.config.random_base, self.keep_prob2, self.config.l2_reg, self.config.n_class, 'doc_softmax_')
        return sen_logits, logits

    # 映射两层, 组合两层
    def create_model_2(self, inputs):
        inputs = tf.reshape(inputs, [-1, self.config.max_sentence_len, self.config.embedding_dim])
        outputs_sen = self.add_bilstm_layer(inputs, self.sen_len, self.config.max_sentence_len, 'doc_sen')  # doc_sen / sen
        outputs_sen_dim = 2 * self.config.n_hidden
        sen_logits = softmax_layer(outputs_sen, outputs_sen_dim, self.config.random_base, self.keep_prob2, self.config.l2_reg, self.config.n_sen_class, 'sen_softmax_')

        outputs_sen = tf.reshape(outputs_sen, [-1, self.config.max_doc_len, outputs_sen_dim])
        outputs_doc = self.add_bilstm_layer(outputs_sen, self.doc_len, self.config.max_doc_len, 'doc')
        logits = softmax_layer(outputs_doc, outputs_sen_dim, self.config.random_base, self.keep_prob2, self.config.l2_reg, self.config.n_class, 'doc_softmax_')
        return sen_logits, logits

    # 一体化
    def add_loss(self, sen_scores, doc_scores):
        print 'I am ass_loss.'
        sen_y = tf.reshape(self.sen_y, [-1, self.config.n_sen_class])
        sen_loss = tf.nn.softmax_cross_entropy_with_logits(logits=sen_scores, labels=sen_y)
        sen_loss = tf.reduce_sum(sen_loss) / tf.cast(tf.reduce_sum(self.doc_len), dtype=tf.float32)

        doc_loss = tf.nn.softmax_cross_entropy_with_logits(logits=doc_scores, labels=self.doc_y)
        doc_loss = tf.reduce_mean(doc_loss)

        self.doc_vars = [var for var in tf.global_variables() if 'doc' in var.name or 'sen' in var.name]
        print self.doc_vars

        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='sen_softmax') +\
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='doc_softmax')
        print reg_loss

        loss = doc_loss + sen_loss + sum(reg_loss)
        return sen_loss, loss

    # 两步走
    def add_loss_sep(self, sen_scores, doc_scores):
        print 'I am ass_loss_sep.'
        sen_y = tf.reshape(self.sen_y, [-1, self.config.n_sen_class])
        sen_loss = tf.nn.softmax_cross_entropy_with_logits(logits=sen_scores, labels=sen_y)
        sen_loss = tf.reduce_sum(sen_loss) / tf.cast(tf.reduce_sum(self.doc_len), dtype=tf.float32)
        self.sen_vars = [var for var in tf.global_variables() if 'sen' in var.name]
        self.sen_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='sen_softmax')
        print self.sen_vars
        print self.sen_reg_loss
        # sen_loss = sen_loss + self.sen_reg_loss
        doc_loss = tf.nn.softmax_cross_entropy_with_logits(logits=doc_scores, labels=self.doc_y)
        doc_loss = tf.reduce_mean(doc_loss)
        self.doc_vars = [var for var in tf.global_variables() if 'doc' in var.name]
        self.doc_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='doc_softmax')
        print self.doc_vars
        print self.doc_reg_loss
        # doc_loss = doc_loss + self.doc_reg_loss
        return sen_loss, doc_loss

    # document accuracy
    def add_accuracy(self, scores):
        correct_predicts = tf.equal(tf.argmax(scores, 1), tf.argmax(self.doc_y, 1))
        accuracy_num = tf.reduce_sum(tf.cast(correct_predicts, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_predicts, tf.float32), name='accuracy')
        return accuracy, accuracy_num

    # sentence accuracy
    def add_sen_acc(self, scores):
        correct_predicts = tf.cast(tf.equal(tf.argmax(scores, 1), tf.argmax(tf.reshape(self.sen_y, [-1, self.config.n_sen_class]), 1)), tf.int32)
        mask = tf.reshape(tf.sequence_mask(self.doc_len, self.config.max_doc_len, dtype=tf.int32), [-1])
        acc_num = tf.reduce_sum(correct_predicts * mask)
        sen_num = tf.reduce_sum(self.doc_len)
        return acc_num, sen_num

    def add_train_op(self, sen_loss, doc_loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.config.lr, global_step, self.config.decay_steps,
                                             self.config.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(self.lr)
        train_op1 = optimizer.minimize(sen_loss, global_step=global_step, var_list=self.sen_vars)
        train_op2 = optimizer.minimize(doc_loss, global_step=global_step, var_list=self.doc_vars)
        return train_op1, train_op2

    def run_op(self, sess, op, data_x, sen_len, doc_len, sen_y, doc_y=None, kp1=1.0, kp2=1.0):
        res_list = []
        len_list = []
        for indices in batch_index(len(data_x), self.config.batch_size, 1, False, False):
            if doc_y is not None:
                feed_dict = self.create_feed_dict(data_x[indices], sen_len[indices], doc_len[indices], sen_y[indices], doc_y[indices], kp1, kp2)
            else:
                feed_dict = self.create_feed_dict(data_x[indices], sen_len[indices], doc_len[indices], sen_y[indices], None, kp1, kp2)
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
        total_sen_loss = []
        total_sen_acc_num = []
        total_sen_num = []
        for step, indices in enumerate(batch_index(len(self.train_doc_y), self.config.batch_size, 1), 1):
            feed_dict = self.create_feed_dict(self.train_x[indices], self.train_sen_len[indices],
                                              self.train_doc_len[indices], self.train_sen_y[indices],
                                              self.train_doc_y[indices], self.config.keep_prob1, self.config.keep_prob2)
            _, loss1, sen_y, sen_acc_num, sen_num = sess.run([self.train_op1, self.sen_loss, self.sen_logits, self.sen_acc_num, self.sen_num], feed_dict=feed_dict)
            # print sen_y
            _, loss, acc_num, lr = sess.run([self.train_op2, self.doc_loss, self.accuracy_num, self.lr], feed_dict=feed_dict)
            total_loss.append(loss)
            total_acc_num.append(acc_num)
            total_num.append(len(indices))
            total_sen_loss.append(loss1)
            total_sen_acc_num.append(sen_acc_num)
            total_sen_num.append(sen_num)
            if verbose and step % verbose == 0:
                print '\n[INFO] {} : sen loss = {}, sen acc = {}, doc loss = {}, doc acc = {}, lr = {}'.format(
                    step,
                    np.mean(total_sen_loss[-verbose:]),
                    sum(total_sen_acc_num[-verbose:]) * 1.0 / sum(total_sen_num[-verbose:]),
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
        best_test_acc = 0
        val_x, val_sen_len, val_doc_len, val_sen_y, val_doc_y = \
            classifier.val_x, classifier.val_sen_len, classifier.val_doc_len, classifier.val_sen_y, classifier.val_doc_y
        test_x, test_sen_len, test_doc_len, test_sen_y, test_doc_y = \
            classifier.test_x, classifier.test_sen_len, classifier.test_doc_len, classifier.test_sen_y, classifier.test_doc_y
        for epoch in range(classifier.config.n_iter):
            print '=' * 20 + 'Epoch ', epoch, '=' * 20
            loss, acc = classifier.run_epoch(sess)
            print '[INFO] Mean loss = {}, mean acc = {}'.format(loss, acc)
            if np.isnan(loss):
                print '[Error] loss is not a number!'
                break
            print '=' * 50
            val_accuracy, val_loss = test_case(sess, classifier, val_x, val_sen_len, val_doc_len, val_sen_y, val_doc_y)
            print '[INFO] val loss: {}, val acc: {}'.format(val_loss, val_accuracy)
            test_accuracy, test_loss = test_case(sess, classifier, test_x, test_sen_len, test_doc_len, test_sen_y, test_doc_y)
            print '[INFO] test loss: {}, test acc: {}'.format(test_loss, test_accuracy)
            if best_accuracy < val_accuracy:
                best_accuracy = val_accuracy
                best_val_epoch = epoch
                best_test_acc = test_accuracy
                if not os.path.exists(classifier.config.weights_save_path):
                    os.makedirs(classifier.config.weights_save_path)
                # saver.save(sess, classifier.config.weights_save_path + '/weights')
            if epoch - best_val_epoch > classifier.config.early_stopping:
                print 'Normal early stop!'
                break
        print 'Best val acc = {}'.format(best_accuracy)
        print 'Test acc = {}'.format(best_test_acc)
    print 'Training complete!'


if __name__ == '__main__':
    tf.app.run(train_run)







