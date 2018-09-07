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

# ------------------------------------------------------------------------
# Split the corpus 80:20 by sentences into a training set and a test set.
# Find the unknown words, that is, the words in the test set that are
# not in the training set.
# ------------------------------------------------------------------------

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
    for sentence in training_sents:
        for w, tag in sentence:
            training.add(w)
    known_words = []
    unknown_words = []
    for word in test:
        if word in training:
            known_words += [ word ]
        else:
            unknown_words += [ word ]
    return ( known_words, unknown_words, )

# ------------------------------------------------------------------------
# The next two methods are for finding the most frequent tag for each word.
# ------------------------------------------------------------------------

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
    return word_mf_tags

# ------------------------------------------------------------------------
# Calculate errors for known and unknown words...
# - if always tagged NN
# ------------------------------------------------------------------------

def count_errors_if_tagged_NN(tagged_test_sents, unknown_words):
    """
    Count the errors in the test set if all words tagged as nouns 'NN'.

    Inputs: both are lists of sentences, each word token in each sentence
            is tagged with the 'correct' tag.
    Output: tuple containing error counts from test set:
            (errors in known words, errors in unknown words)
    """
    errors_known = 0;
    errors_unknown = 0;
    for sentence in tagged_test_sents:
        for w, tag in sentence:
            if tag != 'NN':
                if w in unknown_words:
                    errors_unknown += 1
                else:
                    errors_known += 1
    return (errors_known, errors_unknown,)

def error_rate_if_tagged_NN(tagged_test_sents, unknown_words):
    """
    Compute the error rate in the test set if all words tagged as nouns 'NN'.
    The error rate is the count errors / count words in the test set
    (instances of words, not unique words).

    Inputs: both are lists of sentences, each word token in each sentence
            is tagged with the 'correct' tag.
    Output: tuple containing error rate from test set:
            (error rate in known words, error rate in unknown words)
    """
    errors_known, errors_unknown = count_errors_if_tagged_NN(
        tagged_test_sents, unknown_words
    )
    tagged_words = [ w for sent in tagged_test_sents for w, tag in sent ]
    count_known_words = len(tagged_words) - len(unknown_words)
    error_rate_known = errors_known / count_known_words
    error_rate_unknown = errors_unknown / len(unknown_words)
    error_rate_test = ( errors_known + errors_unknown ) / len(tagged_words)
    return ( error_rate_known, error_rate_unknown, error_rate_test, )

# ------------------------------------------------------------------------
# Calculate errors for known and unknown words...
# - if tagged with the most frequent tag for the word
# ------------------------------------------------------------------------

def count_errors_if_tagged_most_frequent(
    tagged_test_sents, unknown_words, most_frequent_tags):
    """
    Count the errors in the test set if each word is tagged its most frequent tag.

    Inputs: both are lists of sentences, each word token in each sentence
            is tagged with the 'correct' tag.
    Output: tuple containing error counts from test set:
            (errors in known words, errors in unknown words)
    """
    errors_known = 0;
    errors_unknown = 0;
    for sentence in tagged_test_sents:
        for w, tag in sentence:
            if tag != most_frequent_tags[w]:
                if w in unknown_words:
                    errors_unknown += 1
                else:
                    errors_known += 1
    return (errors_known, errors_unknown,)

def error_rate_if_tagged_most_frequent(
    tagged_test_sents, unknown_words, most_frequent_tags):
    """
    Compute the error rate in the test set if each word is tagged its
    most frequent tags.
    The error rate is the count errors / count words in the test set
    (instances of words, not unique words).

    Inputs: both are lists of sentences, each word token in each sentence
            is tagged with the 'correct' tag.
    Output: tuple containing error rate from test set:
            (error rate in known words, error rate in unknown words)
    """
    errors_known, errors_unknown = count_errors_if_tagged_most_frequent(
        tagged_test_sents, unknown_words, most_frequent_tags
    )
    tagged_words = [ w for sent in tagged_test_sents for w, tag in sent ]
    count_known_words = len(tagged_words) - len(unknown_words)
    error_rate_known = errors_known / count_known_words
    error_rate_unknown = errors_unknown / len(unknown_words)
    error_rate_test = ( errors_known + errors_unknown ) / len(tagged_words)
    return ( error_rate_known, error_rate_unknown, error_rate_test, )

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':
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
    runWordTagCounts = get_word_tag_counts(runTaggedSents)
    runWordMFTags = get_most_frequent_tag(runWordTagCounts)

    # NLTK treebank corpus tests.
    tb_training_sents, tb_test_sents = split_train_test(tb.tagged_sents())
    tb_tagged_words = [ (w, tag, ) for s in tb.tagged_sents()
                                   for w, tag in s]
    tb_training_tagged_words = [ (w, tag, ) for s in tb_training_sents
                                            for w, tag in s]
    tb_test_tagged_words = [ (w, tag, ) for s in tb_test_sents
                                        for w, tag in s]
    tb_vocabulary = set(w for w, tag in tb_tagged_words)
    tb_training_vocabulary = set(w for w, tag in tb_training_tagged_words)
    tb_test_vocabulary = set(w for w, tag in tb_test_tagged_words)
    tb_known_words, tb_unknown_words = \
        get_unknown_words(tb_training_sents, tb_test_sents)
    tb_known_tagged_words = [ w for w, tag in tb_test_tagged_words
                              if w in tb_known_words]
    tb_unknown_tagged_words = [ w for w, tag in tb_test_tagged_words
                                if w in tb_unknown_words]
    tb_word_tag_counts = get_word_tag_counts(tb.tagged_sents())
    tb_word_tags_count = sum(tb_word_tag_counts[w][tag]
                       for w in tb_word_tag_counts
                       for tag in tb_word_tag_counts[w])
    tb_training_tag_counts = get_word_tag_counts(tb_training_sents)
    tb_training_tags_count = sum(tb_training_tag_counts[w][tag]
                       for w in tb_training_tag_counts
                       for tag in tb_training_tag_counts[w])
    tb_test_tag_counts = get_word_tag_counts(tb_test_sents)
    tb_test_tags_count = sum(tb_test_tag_counts[w][tag]
                       for w in tb_test_tag_counts
                       for tag in tb_test_tag_counts[w])
    tb_error_counts_NN = \
        count_errors_if_tagged_NN(tb_test_sents, tb_unknown_words)
    tb_error_rates_NN = error_rate_if_tagged_NN(tb_test_sents, tb_unknown_words)
    tb_most_frequent_tags = get_most_frequent_tag(tb_word_tag_counts)
    tb_error_counts_most_frequent = \
        count_errors_if_tagged_most_frequent(tb_test_sents, tb_unknown_words, \
                                              tb_most_frequent_tags)
    tb_error_rates_most_frequent = \
        error_rate_if_tagged_most_frequent(tb_test_sents, tb_unknown_words, \
                                            tb_most_frequent_tags)

    print("Penn-Treebank corpus")
    print("-- sentence counts: training, test, sum, corpus")
    print(len(tb_training_sents), len(tb_test_sents),
          len(tb_training_sents)+len(tb_test_sents),
          len(tb.tagged_sents()))
    print("-- tagged word counts: training, test, sum, corpus")
    print(len(tb_training_tagged_words), len(tb_test_tagged_words),
          len(tb_training_tagged_words)+len(tb_test_tagged_words),
          len(tb_tagged_words))
    print("-- unique word counts: training, test, unknown, training+unknown, corpus")
    print(len(tb_training_vocabulary), len(tb_test_vocabulary),
          len(tb_unknown_words), len(tb_training_vocabulary)+len(tb_unknown_words),
          len(tb_vocabulary))
    print("-- word tag counts: training, test, sum, corpus")
    print(tb_training_tags_count, tb_test_tags_count,
          tb_training_tags_count+tb_test_tags_count,
          tb_word_tags_count)

    print("-- ... with all words tagged 'NN'")
    print("-- tag error counts: known, unknown, total test")
    print(tb_error_counts_NN[0], 'of',
          len(tb_known_tagged_words), 'known tagged test words')
    print(tb_error_counts_NN[1], 'of',
          len(tb_unknown_tagged_words), 'unknown tagged test words')
    print(tb_error_counts_NN[0]+tb_error_counts_NN[1], 'of',
          len(tb_test_tagged_words), 'total tagged test words')
    print("-- tag error rates: known, unknown, total test")
    print(tb_error_rates_NN[0], tb_error_rates_NN[1], tb_error_rates_NN[2])

    print("-- ... with each word tagged with its most frequent tag")
    print("-- tag error counts: known, unknown, total test")
    print(tb_error_counts_most_frequent[0], 'of',
          len(tb_known_tagged_words), 'known tagged test words')
    print(tb_error_counts_most_frequent[1], 'of',
          len(tb_unknown_tagged_words), 'unknown tagged test words')
    print(tb_error_counts_most_frequent[0]+tb_error_counts_most_frequent[1], 'of',
          len(tb_test_tagged_words), 'total tagged test words')
    print("-- tag error rates: known, unknown, total test")
    print(tb_error_rates_most_frequent[0], tb_error_rates_most_frequent[1],
          tb_error_rates_most_frequent[2])
