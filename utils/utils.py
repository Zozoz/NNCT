#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_metrics(pred_prob, data_y):
    p = precision_score(np.argmax(data_y, 1), np.argmax(pred_prob, 1), average=None)
    r = recall_score(np.argmax(data_y, 1), np.argmax(pred_prob, 1), average=None)
    f1 = f1_score(np.argmax(data_y, 1), np.argmax(pred_prob, 1), average=None)
    print 'p:', p, 'avg=', sum(p) / len(data_y[0])
    print 'r:', r, 'avg=', sum(r) / len(data_y[0])
    print 'f1:', f1, 'avg=', sum(f1) / len(data_y[0])


def extract_word2id(sf, df, freq=1):
    fp = open(sf)
    w2c = defaultdict(int)
    for line in fp:
        words = line.decode('utf8').split()
        for word in words:
            w2c[word] += 1
    fp = open(df, 'w')
    w2c = sorted(w2c.items(), key=lambda d: d[1], reverse=True)
    cnt = 0
    for k, v in w2c:
        if v >= freq:
            cnt += 1
            fp.write(k.encode('utf8') + ' ' + str(cnt) + '\n')
    print 'extract word2id done!'


def rule_based_sentiment_analysis(sentence, sentiment_lex):
    if type(sentence) is str:
        sentence = sentence.split()
    polarity = 0
    for word in sentence:
        if word in sentiment_lex:
            polarity += sentiment_lex[word]
    return polarity


def cal_sen_sen_in_doc(doc_file, lex_file, doc_sentiment_file):
    # load sentiment lexicon
    lex_dict = dict()
    for line in open(lex_file):
        k, v = line.split()
        lex_dict[k] = float(v)
    print 'loading sentiment lexicon done!'
    # calculate sentences sentiment in doc
    sf = open(doc_file)
    df = open(doc_sentiment_file, 'w')
    for line in sf:
        sentences = line.split('||')[-1].split('<sssss>')
        polarity = []
        for sentence in sentences:
            polarity.append(str(rule_based_sentiment_analysis(sentence, lex_dict)))
        df.write(' '.join(polarity) + '\n')
    print 'calculating done!'


def add_sen_sen_in_doc(doc_file, lex_file, doc_file_new):
    # load sentiment lexicon
    lex_dict = dict()
    for line in open(lex_file):
        k, v = line.split()
        lex_dict[k] = float(v)
    print 'loading sentiment lexicon done!'
    sf = open(doc_file)
    df = open(doc_file_new, 'w')
    for line in sf:
        line = line.split('||')
        y = line[0]
        sents = line[-1].split('<sssss>')
        df.write(y + '||')
        for sent in sents:
            p = rule_based_sentiment_analysis(sent, lex_dict)
            if p > 0:
                sent = '<POS> ' + sent + ' </POS>' + ' <sssss> '
            elif p < 0:
                sent = '<NEG> ' + sent + ' </NEG>' + ' <sssss> '
            else:
                sent = sent + ' <sssss> '
            df.write(sent)
        df.write('\n')


def cv(files, dest_file, fd=10):
    """
    generate cross validate file
    :param files: list type containing different polarity file name
    :param dest_file: destination file directory path
    :param fd: int type, n-fold
    :return: none
    """
    for file in files:
        lines = open(file).readlines()
        batch_size = len(lines) / fd
        for i in range(fd):
            s = i * batch_size
            e = s + batch_size
            test_f = open(dest_file + '/' + str(i) + '_test.txt', 'a')
            for line in lines[s:e]:
                test_f.write(line)
            test_f.close()
            train_f = open(dest_file + '/' + str(i) + '_train.txt', 'a')
            for line in (lines[:s] + lines[e:]):
                train_f.write(line)
            train_f.close()






