"""
NLP_2ed_ch_5_problems

Code for NLP 2nd Edition Chapter 5 problems

    Uses Penn-Treebank corpus from NLTK.

    5.5 Implement Brill TBL.
        Create a small number of templates.
        Train on POS-tagged corpus.
        Divide corpus into training and test sets,
        80:20 at random by sentence.
    5.6 Implement most-likely-tag baseline.
        Compute most-likely-tags for a POS-tagged corpus.
        Run Rrill TBL starting with all words tagged 'NN'.
        Compute error rate on known and unknown words.
        Create 5 rules to do a better job.
        Show the difference in error rates.

Sig Nin
2018 Sep 1
"""
import nltk
from nltk.corpus import treebank as tb
from nlp_book_nltk import *
from random import random as rand

tb_tagged_bigrams_sents = bi_grams_sents(tb.tagged_sents())

def split_train_test(tagged_sents):
    training = []
    test = []
    for sent in tagged_sents:
        if rand() > 0.8:
            test += [ sent ]
        else:
            training += [ sent ]
    return ( training, test, )

tb_training_sents, tb_test_sents = split_train_test(tb.tagged_sents())

def get_word_tag_counts(tagged_sents):
    wtc = {}
    for sent in tagged_sents:
        for word, tag in sent:
            if word not in wtc:
                wtc[word] = {}
            word_entry = wtc[word]
            if tag not in word_entry:
                word_entry[tag] = 0
            word_entry[tag] += 1
    return wtc

tb_word_tag_counts = get_word_tag_counts(tb.tagged_sents())

