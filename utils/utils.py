#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os


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


def cv_10(files, dest_file):
    for file in files:
        lines = open(file).readlines()
        batch_size = len(lines) / 10
        for i in range(10):
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






