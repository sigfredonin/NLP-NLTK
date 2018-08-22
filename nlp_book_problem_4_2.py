# nlp_book_problem_4_2: Compute unsmoothed unigrams, bigrams, and trigrams.
# Sig Nin 2018-08-08

import nltk
from nltk.book import *
from nlp_book_nltk import *

# From word lists in NLTK book corpora ...
def get_bigrams(words):
    two_grams = bi_grams(words)
    return two_grams

# Solve the problem, incrementally ..
def main():
    text_1_bigrams = bi_grams(text1) # Moby Dick
    print("Count bigrams in Moby Dick")
    print(len(list(text_1_bigrams)))

    text_1_p_bigrams = unsmoothed_bigrams(text1)
    print("Count unsmoothed bigram probabilities for Moby Dick")
    print(len(text_1_p_bigrams))
    print("First 10 unsmoothed bigram probabilities for Moby Dick")
    for gram in text_1_bigrams[:10]:
        print(gram, ":", text_1_p_bigrams[gram])

    text_1_trigrams = tri_grams(text1) # Moby Dick
    print("Count trigrams in Moby Dick")
    print(len(list(text_1_trigrams)))

    text_1_p_trigrams = unsmoothed_trigrams(text1)
    print("Count unsmoothed trigram probabilities for Moby Dick")
    print(len(text_1_p_trigrams))
    print("First 10 unsmoothed trigram probabilities for Moby Dick")
    for gram in text_1_trigrams[:10]:
        print(gram, ":", text_1_p_trigrams[gram])

# Run the program
main()
