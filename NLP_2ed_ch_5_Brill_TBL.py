"""
Brill Transformation-Based Learning Tagger

Algorithm sketched in NLP 2nd Ed figure 5.21.

Sig Nin
5 Sep 2018
"""
class BrillTBL:

    # Each template is [ type, tagFrom, tagTo, tagZ, tagW, description ]
    # where the description is a format string with places for the tags used.
    # The tagger is initialized with the simple template used in the
    # algorithm sketch in NLP 2ed figure 5.21.
    # A template requires corresponding code in the tagger.
    templates = [
        [ 'm1', '', '', '', "Change %s to %s if preceding tag is %s" ]
    ]

    # This is the constructor, which takes a corpus with an initial tagging.
    __init__(self, tagged_sents):
    """
    Initialize a new instance with a tagged corpus.
    The tagged corpus is a list of sentences with the word tokens tagged.
    Example:
        [
         [('I', 'PRP'), ('went', 'VBD'), ('for', 'IN'),
          ('a', 'DT'), ('run', 'NN'), ('.', '.')],
         [('Do', 'VBP'), ("n't", 'RB'), ('run', 'VB'),
          ('so', 'RB'), ('fast', 'RB'), ('!', '.')]
        ]
    """
    self.tagged_sents = tagged_sents
