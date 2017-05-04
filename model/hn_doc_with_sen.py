#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())

from utils.config import *
from utils.data_helper import load_w2v, load_inputs_document
from newbie_nn.nn_layer import bi_dynamic_rnn, softmax_layer
from newbie_nn.att_layer import mlp_attention_layer


class HN_DOC_WITH_SEN(object):

    def __init__(self):
        self.config = FLAGS

    def add_placeholder(self):
        self.x = tf.placeholder(tf.int32, [None, self.config.max_doc_len, self.config.max_sentence_len])
        self.y = tf.placeholder(tf.int32, [None, self.config.n_class])
        self.sen_len = tf.placeholder(tf.int32, [None, self.config.max_doc_len])
        self.doc_len = tf.placeholder(tf.int32, [None])
        self.sen_y = tf.placeholder(tf.int32, [None, self.config.max_doc_len])
        self.keep_prob1 = tf.placeholder(tf.float32)
        self.keep_prob2 = tf.placeholder(tf.float32)

    def add_embedding(self):
        if self.config.pre_trained == 'yes':
            self.word2id, w2v = load_w2v(self.config.embedding_file, self.config.embedding_dim, True)
            self.embedding = tf.Variable(w2v, dtype=tf.float32, name='word embedding')
        else:
            pass
        inputs = tf.nn.embedding_lookup(self.embedding, self.x)
        return inputs

    def load_data(self):
        self.train_x, self.train_sen_len, self.train_doc_len, self.train_sen_y, self.train_doc_y = load_inputs_document(
            self.config.train_file, self.config.train_sen_y, self.word2id,
            self.config.max_sentence_len, self.config.max_doc_len)
        self.test_x, self.test_sen_len, self.test_doc_len, self.test_sen_y, self.test_doc_y = load_inputs_document(
            self.config.test_file, self.config.test_sen_y, self.word2id,
            self.config.max_sentence_len, self.config.max_doc_len)
        self.val_x, self.val_sen_len, self.val_doc_len, self.val_sen_y, self.val_doc_y = load_inputs_document(
            self.config.val_file, self.config.val_sen_y, self.word2id,
            self.config.max_sentence_len, self.config.max_doc_len)

    def create_feed_dict(self, x_batch, sen_len_batch, doc_len_batch, sen_y_batch, y_batch=None):
        if y_batch is None:
            holder_list = [self.x, self.sen_len, self.doc_len, self.sen_y]
            feed_list = [x_batch, sen_len_batch, doc_len_batch, sen_y_batch]
        else:
            holder_list = [self.x, self.sen_len, self.doc_len, self.sen_y, self.y, self.keep_prob1, self.keep_prob2]
            feed_list = [x_batch, sen_len_batch, doc_len_batch, sen_y_batch, y_batch, self.config.keep_prob1, self.config.keep_prob2]
        return dict(zip(holder_list, feed_list))

    def add_model(self, inputs):
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)
        cell = tf.contrib.rnn.LSTMCell
        # word to sentence
        sen_len = tf.reshape(self.sen_len, [-1])
        hiddens_sen = bi_dynamic_rnn(cell, inputs, self.config.n_hidden, sen_len, self.config.max_sentence_len, 'sentence', 'all')
        alpha_sen = mlp_attention_layer(hiddens_sen, sen_len, 2 * self.config.n_hidden, self.config.l2_reg, self.config.random_base, 1)
        outputs_sen = tf.batch_matmul(alpha_sen, hiddens_sen)
        sen_logits = softmax_layer(outputs_sen, 2 * self.config.n_hidden, self.config.random_base, self.keep_prob2, self.config.l2_reg, self.config.n_class)
        outputs_sen = tf.reshape(outputs_sen, [-1, self.config.max_doc_len, 2 * self.config.n_hidden])

        # sentence to doc
        hiddens_doc = bi_dynamic_rnn(cell, outputs_sen, self.config.n_hidden, self.doc_len, self.config.max_doc_len, 'doc', 'all')
        alpha_doc = mlp_attention_layer(hiddens_doc, self.doc_len, 2 * self.config.n_hidden, self.config.l2_reg, self.config.random_base, 2)
        outputs_doc = tf.reshape(tf.batch_matmul(alpha_doc, hiddens_doc), [-1, 2 * self.config.n_hidden])

        logits = softmax_layer(outputs_doc, 2 * self.config.n_hidden, self.config.random_base, self.keep_prob2, self.config.l2_reg, self.config.n_class)
        return sen_logits, logits











