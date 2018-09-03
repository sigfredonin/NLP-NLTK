"""
NLP_2ed_ch_5_problems

Code for NLP 2nd Edition Chapter 5 problems

    Uses Penn-Treebank corpus from NLTK.

    5.5 Implement Brill TBL.
        Create a small number of templates.
        Train on POS-tagged corpus.
    5.6 Implement most-likely-tag baseline:
        Divide corpus into training and test sets,
        80:20 at random by sentence.
        Verify that test set has words not in the
        training set (unknown words).
        Compute the error rate on the test set if
        the training set words are all tagged NN.
        Compute separate error rates for known
        and unknown words.
        Compute most-likely-tags using the training set.
        Compute the error rate for known words in the test set
        if they are tagged with the most likely tag.
        Compute the error rate for unknown words in the test set
        if they are tagged NN.
        Run Rrill TBL starting with all words tagged 'NN',
        using just the template
            "Change a to b iff previous tag is z."
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

# This way is very inefficient, O(n^2).
def get_unknown_words_loopy(training_sents, test_sents):
    unknown = []
    for test_sentence in test_sents:
        for training_sentence in training_sents:
            unknown += [ w for (w , pos) in test_sentence
                         if w not in training_sentence ]
    return set(unknown)

# This way is O(n)
def get_unknown_words(training_sents, test_sents):
    test = set()
    for sentence in test_sents:
        for w, tag in sentence:
            test.add(w)
    training = set()
    for training_sentence in training_sents:
        for w, tag in sentence:
            training.add(w)
    return [ w for w in test if w not in training ]

tb_unknown_words = get_unknown_words(tb_training_sents, tb_test_sents)

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
