# nlp_book_nltk: Define functions to solve NLP book problems with NLTK
# Sig Nin 2018-08-08

from nltk import *

# Compute bigrams from a list of word tokens
def bi_grams(words):
    grams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    return grams

# Compute bigrams from a sentence
# comprising a list of words (including punctuation).
# The first bigram for the sentence is ( '', <first word> ).
# The last bigram for the sentence is ( <last word>, '' ).
def bi_grams_sent(sent):
    grams_first = []
    grams_last = []
    if len(sent) > 0:
        grams_first = [('', sent[0],)]
        grams_last = [(sent[-1], '',)]
    return grams_first + bi_grams(sent) + grams_last

# Compute unsmoothed bigram probabilities from a list of sentences
def bi_grams_sents(sents):
    grams = []
    for sent in sents:
        grams += bi_grams_sent(sent)
    return grams

# Compute trigrams from a list of word tokens
def tri_grams(words):
    grams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
    return grams

# Compute trigrams from a sentence
# comprising a list of words (including punctuation).
# The first trigram for the sentence is ( '', '', <first word> ).
# The second trigram is ( ''. <first word>, <second word> ).
# The next to last trigram is ( <next to last word>, <last word>, '' ).
# The last trigram is ( <last word>, '', '' ).
def tri_grams_sent(sent):
    grams = []
    if len(sent) == 1:
        grams = [('', '', sent[0],), ('', sent[0], '',), (sent[0], '', '',)]
    if len(sent) == 2:
        grams = [('', '', sent[0],), ('', sent[0], sent[1],),
                 (sent[0], sent[1], '',), (sent[1], '', '',)]
    if len(sent) > 2:
        grams  = [('', '', sent[0],), ('', sent[0], sent[1],)]
        grams += tri_grams(sent)
        grams += [(sent[-2], sent[-1], '',), (sent[-1], '', '',)]
    return grams

# Compute unsmoothed trigram probabilities from a list of sentences
def tri_grams_sents(sents):
    grams = []
    for sent in sents:
        grams += tri_grams_sent(sent)
    return grams

# Compute unsmoothed unigram probabilities from a list of words
def unsmoothed_unigrams(words):
    dist_words = FreqDist(words)    # unigram counts
    pgrams = {}
    for gram in set(words):
        count_word = dist_words[gram]
        pgram = count_word / len(words)
        pgrams[gram] = pgram
    return pgrams

# Compute unsmoothed unigram probabilities from a list of sentences
def unsmoothed_unigrams_sents(sents):
    words = []
    for sent in sents:
        words += sent
    return unsmoothed_unigrams(words)

# Compute unsmoothed bigram probabilities from a list of words
def unsmoothed_bigrams(words):
    grams = bi_grams(words)
    dist_words = FreqDist(words)    # unigram counts
    dist_grams = FreqDist(grams)    # bigram counts
    pgrams = {}
    for iGram in range(len(grams)):
        gram = grams[iGram]
        count_gram = dist_grams[gram]
        count_word_0 = dist_words[gram[0]]
        pgram = count_gram / count_word_0
        pgrams[gram] = pgram
    return pgrams

# Compute unsmoothed bigram probabilities from a list of sentences
def unsmoothed_bigrams_sents(sents):
    words = []
    for sent in sents:
        words += ['']    # start of sentence
        words += sent
    grams = []
    for sent in sents:
        grams += bi_grams_sent(sent)
    dist_words = FreqDist(words)    # unigram counts
    dist_grams = FreqDist(grams)    # bigram counts
    pgrams = {}
    for iGram in range(len(grams)):
        gram = grams[iGram]
        count_gram = dist_grams[gram]
        count_word_0 = dist_words[gram[0]]
        try:
            pgram = count_gram / count_word_0
            pgrams[gram] = pgram
        except ZeroDivisionError:
            print("zero word count for:", "'", gram[0], "'")
    return pgrams

# Compute unsmoothed trigram probabilities from a list of words
def unsmoothed_trigrams(words):
    N2grams = bi_grams(words)
    N3grams = tri_grams(words)
    dist_N2 = FreqDist(N2grams)    # bigram counts
    dist_N3 = FreqDist(N3grams)    # trigram counts
    pgrams = {}
    for iGram in range(len(N3grams)):
        N3gram = N3grams[iGram]
        N2gram = N3gram[:2]
        count_N2gram = dist_N2[N2gram]
        count_N3gram = dist_N3[N3gram]
        pgram = count_N3gram / count_N2gram
        pgrams[N3gram] = pgram
    return pgrams

# Compute unsmoothed trigram probabilities from a list of sentences
def unsmoothed_trigrams_sents(sents):
    print("sents: ", sents[:5])
    N2grams = []
    N3grams = []
    count = 0
    for sent in sents:
        if count < 5:
            print("sent: ", sent)
        N2grams += [('', '',)]   # start of sentence
        N2grams += bi_grams_sent(sent)
        N3grams += tri_grams_sent(sent)
        count += 1
    print("bigrams: ", N2grams[:40])
    print("trigrams: ", N3grams[:40])
    dist_N2 = FreqDist(N2grams)    # bigram counts
    dist_N3 = FreqDist(N3grams)    # trigram counts
    pgrams = {}
    for iGram in range(len(N3grams)):
        N3gram = N3grams[iGram]
        N2gram = N3gram[:2]
        count_N2gram = dist_N2[N2gram]
        count_N3gram = dist_N3[N3gram]
        try:
            pgram = count_N3gram / count_N2gram
            pgrams[N3gram] = pgram
        except ZeroDivisionError:
            print("Zero count for bigram: ", N2gram)
    return pgrams

# trigrams beginning with particular bigram, from trigrams
def tri_grams_starting(trigrams, bigram):
    return [ trigram for trigram in trigrams
             if trigram[0] == bigram[0] and trigram[1] == bigram[1] ]

# Sentence start trigrams from trigrams
# Extracts the trigrams that begin ( '', '', )
def tri_grams_sent_starts(trigrams):
    return tri_grams_starting(trigrams, ( '', '', ))

# calculate cummulative probabilities for
# a list of unsmoothed Ngram probabilities
def cummulative_probabilities(utps):
    utcps = []                                      # .. cummulative
    cummulative_probability = 0.0
    for n_gram in utps:
        cummulative_probability += utps[n_gram]
        utcps += [(n_gram, cummulative_probability, )]
    return utcps

# choose an Ngram at random from a list of (Ngram, cummulative probability)
# so that each ngram has its own probability of being chosen
def choose_by_probability(utcps):
    from random import uniform
    cummulative_probability = utcps[-1][1]
    r = uniform(0.0, cummulative_probability)
    print("Random value, r:", r)
    entry = None
    for i in range(len(utcps)):
        entry = utcps[i]
        prob = entry[1];
        if i < 10:
            print("---", entry, prob)
        if r <= prob:
            break
    return entry

# choose an Ngram at random from a list of (Ngram, cummulative probability)
# so that each ngram has its own probability of being chosen.
# Use binary search.
def choose_by_probability_bin_search(utcps):
    from random import uniform
    cummulative_probability = utcps[-1][1]
    r = uniform(0.0, cummulative_probability)
    print("Random value, r:", r, ", Trigram list size:", len(utcps))
    entry = None
    first = 0
    last = len(utcps) - 1
    found = False
    while first < last:     # while interval size > 1
        i = (first + last) // 2
        entry = utcps[i]
        prob = entry[1];
        if i < 20:
            print("---", first, i, last, ":", entry, prob)
        if r < prob:
            last = i        # in this or earlier interval
        else:
            first = i + 1   # in later interval
        
    return utcps[last]

# Test
def test():
    sents = [[], ["One"], ["One", "."], ["One", "two", "."],
             ['This', 'is', 'a', 'short', 'test', '.'] ]
    for sent in sents:
        print("----", sent, "----")
        print("-- bigrams")
        print(bi_grams(sent))
        print(bi_grams_sent(sent))
        print("-- trigrams")
        print(tri_grams(sent))
        print(tri_grams_sent(sent))

# test()

# Problem 4.2 test
def problem_4_2(filename):
    from nltk.corpus import gutenberg
    words = gutenberg.words(filename)
    sents = gutenberg.sents(filename)
    UG_dist = FreqDist(words)
    UG_dist.most_common(n=50)
    bigrams = bi_grams_sents(sents)
    BG_dist = FreqDist(bigrams)
    BG_dist.most_common(n=50)
    emma_trigrams = tri_grams_sents(sents)
    TG_dist = FreqDist(trigrams)
    TG_dist.most_common(n=50)
    uups = unsmoothed_unigrams_sents(sents)
    ubps = unsmoothed_bigrams_sents(sents)
    utps = unsmoothed_trigrams_sents(sents)
    ssts = tri_grams_sent_starts(trigrams)
    sstps = { gram : utps[gram] for gram in ssts }
    print("---- from ", filename, " ----")
    print("Count words:",
          len(words))
    print("Count unique words:",
          len(set(words)))
    print(words[:30])
    print("Count sentences:",
          len(sents))
    print(sents[:5])
    print("Count unsmoothed uigram probabilities:",
          len(uups))
    print("Unsmoothed unigram probabilities for most common unigrams:")
    print([(gram, uups[gram[0]],)
           for gram in UG_dist.most_common(n=50)])
    print("Count unsmoothed bigram probabilities:",
          len(ubps))
    print("Unsmoothed bigram probabilities for most common bigrams:")
    print([(gram, ubps[gram[0]],)
           for gram in BG_dist.most_common(n=50)])
    print("Count unsmoothed trigram probabilities:",
          len(utps))
    print("Unsmoothed trigram probabilities for most common trigrams:")
    print([(gram, utps[gram[0]],)
           for gram in TG_dist.most_common(n=50)])

# Problem 4.3 example
def problem_4_3():
    problem_4_2('austen-emma.txt')
    problem_4_2('carroll-alice.txt')

# problem_4_3()

# Problem 4.4
def problem_4_4_(filename):
    from nltk.corpus import gutenberg
    words = gutenberg.words(filename)
    sents = gutenberg.sents(filename)
    UG_dist = FreqDist(words)
    UG_dist.most_common(n=50)
    bigrams = bi_grams_sents(sents)
    BG_dist = FreqDist(bigrams)
    BG_dist.most_common(n=50)
    trigrams = tri_grams_sents(sents)
    TG_dist = FreqDist(trigrams)
    TG_dist.most_common(n=50)
    uups = unsmoothed_unigrams_sents(sents)
    ubps = unsmoothed_bigrams_sents(sents)
    utps = unsmoothed_trigrams_sents(sents)
    print("---- from ", filename, " ----")
    # calculate cummulative probabilities for start words ...
    # choose words until a sentence ending trigram chosen
    bigram = ('', '',)   # starting bigram: ('', '')
    sentence = ''
    while True:
        ntgs = tri_grams_starting(trigrams, bigram)     # possible next words
        ntps = { gram : utps[gram] for gram in ntgs }   # next word probabilities
        ntcs = cummulative_probabilities(ntps)          # .. cummulative
        cummulative_probability = ntcs[-1][1]
        entry = choose_by_probability_bin_search(ntcs)
        print("Randomly chosen entry:", entry)
        if entry == None:
            print("Error: no next word found, r:", r,
                  " > ", cummulative_probability)
            break
        next_word = entry[0][2]
        print("Chosen next word: ", next_word,
              ", from entry:", (entry[0], ntps[entry[0]],),
              "cumm prob:", entry[1])
        if next_word == '':
            break
        sentence += " " + next_word
        bigram = (entry[0][1], entry[0][2],)    # next bigram
    print(sentence)
    outFileName = filename+".sentences.txt"
    with open(outFileName, "a") as outFile:
        outFile.write(sentence+"\n\n")
        outFile.close()

# Add timestamp to output file
def add_timestamp(outFileName):
    from datetime import datetime
    with open(outFileName, "a") as outFile:
        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        timestampLine = "====" + nowStr + "====\n\n"
        outFile.write(timestampLine)
        outFile.close()

# Problem 4.4 - Generate random sentences from trigrams
def problem_4_4():
    texts = [ 'carroll-alice.txt', 'austen-emma.txt' ]
    for title in texts:
        add_timestamp(title + ".sentences.txt")
        for i in range(10):
            problem_4_4_(title)
