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
    DEFAULT_TEMPLATE =
        [ 'm1', '', '', '', '', "Change %s to %s if preceding tag is %s" ]

    # This is the constructor, which takes a corpus with an initial tagging
    # and a tagger name.  As an example, the name could identify the
    # corpus and initial tagging:
    #   "Penn-Treebank initialized with nltk.pos_tag()"
    __init__(self, tagged_sents_truth, tagged_sents_initial, name):
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
        self.true_tagged_sents = tagged_sents_truth
        self.initial_tagged_sents = tagged_sents_initial
        self.corpus_name = name
        self.templates = DEFAULT_TEMPLATE
        self.transforms_queue = []

    # Learn the best transforms, instances of the templates that
    # most improve on the initial tagging.
    learn():
        """
        Get the best transforms for this initial tagging.
        Return a list of the best rules, transform instances
        with the tag slots set.
        """
        while True:
            best_transform = get_best_transform()
            best_rule = apply_transform(best_transform, tagged_sents)
            self.transforms_queue += [ best_rule ]
        return self.transforms_queue

    # Get the transform that most improves the current tagging.
    get_best_transform():
        """
        Iterate through the templates, generating the template instance
        that most improves the current tagging.
        """
        best_transform_score = 0
        for template in self.templates:
            (instance, score) = get_best_instance(template)
            if (score > best_tansform_score)
