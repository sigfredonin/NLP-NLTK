from nltk import *
from nlp_book_nltk import bi_grams_sents
from nlp_book_nltk import tri_grams_sents
from nlp_book_nltk import unsmoothed_unigrams_sents
from nlp_book_nltk import unsmoothed_bigrams_sents
from nlp_book_nltk import unsmoothed_trigrams_sents
filename = 'austen-emma.txt'
print("---- from ", filename, " ----")
from nltk.corpus import gutenberg
words = gutenberg.words(filename)
sents = gutenberg.sents(filename)
UG_dist = FreqDist(words)
UG_dist.most_common(n=50)
bigrams = bi_grams_sents(sents)
BG_dist = FreqDist(bigrams)
BG_dist_counts = [ BG_dist[gram] for gram in BG_dist ]
BG_dist.most_common(n=50)
trigrams = tri_grams_sents(sents)
TG_dist = FreqDist(trigrams)
TG_dist_counts = [ TG_dist[gram] for gram in TG_dist ]
TG_dist.most_common(n=50)
uups = unsmoothed_unigrams_sents(sents)
ubps = unsmoothed_bigrams_sents(sents)
utps = unsmoothed_trigrams_sents(sents)
