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
import operator

# These are the bigrams, per sentence, from the NLTK penntree coupus.
#   Each sentence in a list.
#   Each bigram is a tuple:
#       ('', (word, tag))          - beginning of sentence
#       ((word, tag), (word, tag))  - middle of sentence
#       ((word, tag), '')          - end of sentence
# Example:
#   Sentences: "I am Sam.", "Sam I am."
#   Tagged: [[('I', 'PRP'), ('am', 'VBP'), ('Sam', 'NNP'), ('.', '.')],
#            [('Sam', 'NNP'), ('I', 'PRP'), ('am', 'VBP'), ('.', '.')]]
#   Tagged Bigrams:
#           [('', ('I', 'PRP')), (('I', 'PRP'), ('am', 'VBP')),
#            (('am', 'VBP'), ('Sam', 'NNP')), (('Sam', 'NNP'), ('.', '.')),
#            (('.', '.'), ''), ('', ('Sam', 'NNP')),
#            (('Sam', 'NNP'), ('I', 'PRP')), (('I', 'PRP'), ('am', 'VBP')),
#            (('am', 'VBP'), ('.', '.')), (('.', '.'), '')]
tb_tagged_bigrams_sents = bi_grams_sents(tb.tagged_sents())

def split_train_test(tagged_sents):
    """
    Split a list of tagged sentences into a training set and a test set,
    ration 80:20 training: test, selected randomly.
    """
    training = []
    test = []
    for sent in tagged_sents:
        if rand() > 0.8:
            test += [ sent ]
        else:
            training += [ sent ]
    return ( training, test, )

# This is an 80:20 random split of the pentree tagged sentences.
tb_training_sents, tb_test_sents = split_train_test(tb.tagged_sents())

def get_unknown_words_loopy(training_sents, test_sents):
    """
    Get the unknown words in a test set.
    The unknown words are the ones that appear in the test set
    but not in the training set.
    But ugh. This way is very inefficient, O(n^2).
    """
    unknown = []
    for test_sentence in test_sents:
        for training_sentence in training_sents:
            unknown += [ w for (w , pos) in test_sentence
                         if w not in training_sentence ]
    return set(unknown)

def get_unknown_words(training_sents, test_sents):
    """
    Get the unknown words in a test set.
    The unknown words are the ones that appear in the test set
    but not in the training set.
    This way is better, O(n).
    """
    test = set()
    for sentence in test_sents:
        for w, tag in sentence:
            test.add(w)
    training = set()
    for training_sentence in training_sents:
        for w, tag in sentence:
            training.add(w)
    return [ w for w in test if w not in training ]

# These are the unknown words in the treebank split.
tb_unknown_words = get_unknown_words(tb_training_sents, tb_test_sents)

# The next two methods are for finding the most frequent tag for each word.
def get_word_tag_counts(tagged_sents):
    """
    Get the number of times each word is tagged with a tag.
        Returns a dictionary of words, each entry is a dictionary
        containing the tags, with the count for each.
    Example:
        Sentences: "I went for a run." "Don't run so fast!"
        Tag counts:
            {'I': {'PRP': 1}, 'went': {'VBD': 1}, 'for': {'IN': 1},
             'a': {'DT': 1}, 'run': {'NN': 1, 'VB': 1}, '.': {'.': 1},
             'Do': {'VBP': 1}, "n't": {'RB': 1}, 'so': {'RB': 1},
             'fast': {'RB': 1}, '!': {'.': 1}}
    """
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
tb_training__tag_counts = get_word_tag_counts(tb_training_sents)

def get_most_frequent_tag(tag_counts):
    """
    Get the most frequent tag for each word in a tag counts dictionary.
    Example:
        Input word tag counts:
            {'I': {'PRP': 1}, 'went': {'VBD': 1}, 'for': {'IN': 1},
             'a': {'DT': 1}, 'run': {'NN': 1, 'VB': 1}, '.': {'.': 1},
             'Do': {'VBP': 1}, "n't": {'RB': 1}, 'so': {'RB': 1},
             'fast': {'RB': 1}, '!': {'.': 1}}
        Most frequent word tags:
            {'I': 'PRP', 'went': 'VBD', 'for': 'IN',
             'a': 'DT', 'run': 'NN', '.': '.',
             'Do': 'VBP', "n't": 'RB', 'so': 'RB',
             'fast': 'RB', '!': '.'}
    """
    word_mf_tags = { w : max(tags.items(), key=operator.itemgetter(1))[0]
                     for w, tags in tag_counts.items() }

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

iamsam = "I am Sam."
iamsamTokens = nltk.word_tokenize(iamsam)
iamsamTagged = nltk.pos_tag(iamsamTokens)
samiam = "Sam I am."
samiamTokens = nltk.word_tokenize(samiam)
samiamTagged = nltk.pos_tag(samiamTokens)
samTaggedSents = [ iamsamTagged, samiamTagged ]
samTaggedSentsBigrams = bi_grams_sents(samTaggedSents)
samWordTagCounts = get_word_tag_counts(samTaggedSents)
samWordMFTags = get_most_frequent_tag(samWordTagCounts)

iwentforarun="I went for a run."
iwentforarunTokens = nltk.word_tokenize(iwentforarun)
iwentforarunTagged = nltk.pos_tag(iwentforarunTokens)
dontrunsofast = "Don't run so fast!"
dontrunsofastTokens = nltk.word_tokenize(dontrunsofast)
dontrunsofastTagged = nltk.pos_tag(dontrunsofastTokens)
runTaggedSents = [ iwentforarunTagged, dontrunsofastTagged ]
runTaggedSentsBigrams = bi_grams_sents(runTaggedSents)
runTagCounts = get_word_tag_counts(runTaggedSents)
runWordMFTags = get_most_frequent_tag(runWordTagCounts)
