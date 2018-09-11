"""
Brill Transformation-Based Learning Tagger

Algorithm sketched in NLP 2nd Ed figure 5.21.

Sig Nin
5 Sep 2018
"""

from collections import defaultdict

# Each template is [ type, tagFrom, tagTo, tagZ, tagW, description ]
# where the description is a format string with places for the tags used.
# The tagger is initialized with the simple template used in the
# algorithm sketch in NLP 2ed figure 5.21.
# The NIL template is the starting point for the best transform.
# A template requires corresponding code in the tagger.
DEFAULT_TEMPLATE = [
        'm1', '', '', '', '', "Change %s to %s if preceding tag is %s%s."
    ]
NIL_TEMPLATE = [
        'NIL', '', '', '', '', "Don't change the tag%s%s%s%s."
    ]

# Start of text tag, for positions preceding the beginning of the text.
START_TAG = '<START>'

class BrillTBL:
    """
    Brill TBL tagger.
    Starting with an initial tag and rule templates of the form
        "Change <fromTag> to <toTag> iff <tagZ, tagW in context>"
    learns a series of rules in which the tags are resolved
    that improve the tagging of the training set compared to a
    ground truth tagging.
    The series of rules can be applied to another corpus to tag it.
    """

    # This is the constructor, which takes a corpus with an initial tagging,
    # a reference ground-truth tagging, a list of possible tags - the tagset,
    # and a tagger name.  As an example, the name could identify the
    # corpus and initial tagging:
    #   "Penn-Treebank initialized with nltk.pos_tag()"
    def __init__(self, tagged_words_initial, tagged_words_true, tagset, name):
        """
        Initialize a new instance with a tagged corpus.
        The tagged corpus is a list of tagged word tokens.
        Example:
            [
             ('I', 'PRP'), ('went', 'VBD'), ('for', 'IN'),
             ('a', 'DT'), ('run', 'NN'), ('.', '.'),
             ('Do', 'VBP'), ("n't", 'RB'), ('run', 'VB'),
             ('so', 'RB'), ('fast', 'RB'), ('!', '.')
            ]
        The learning algorithm begins with an initial tagging.  It uses a
        "ground-truth" tagging and a tagset as reference.
        """
        self.tagged_words = tagged_words_initial
        self.tagged_words_true = tagged_words_true
        self.set_tags_from_tagged_words()
        self.tagset = tagset
        self.name = name
        self.templates = [ DEFAULT_TEMPLATE ]
        self.transforms_queue = []

    def set_tags_from_tagged_words(self):
        self.tags_init = [ tag for word, tag in self.tagged_words ]
        self.tags_true = [ tag for word, tag in self.tagged_words_true ]
        self.tags = list(zip(self.tags_init, self.tags_true))
        
    # Display a transform instance
    def printTransform(self, transform):
        typeT, fromTag, toTag, zTag, wTag, desc = transform[0]
        score = transform[1]
        description = desc % (fromTag, toTag, zTag, wTag, )
        print(typeT+':', description, 'score=', score)

    # Learn the best transforms, instances of the templates that
    # most improve on the initial tagging.
    def learn(self):
        """
        Get the best transform sequence for this initial tagging.
        Return a list of the best rules, transform instances
        with the tag slots set.
        The learned rules can be applied with the "tag()" method
        to another corpus.
        """
        minimum = 1
        last_score = 0
        while True:
            best_transform = self.get_best_transform()
            score = best_transform[1]
            change_in_score = score - last_score
            if score < minimum:
                break
            last_score = score
            self.apply_transform(best_transform)
            self.set_tags_from_tagged_words()
            self.transforms_queue += [ best_transform ]
            self.printTransform(best_transform)
            print("----------")
        print("==> ", end='')
        self.printTransform(best_transform)
        return self.transforms_queue

    # Get the transform that most improves the current tagging.
    def get_best_transform(self):
        """
        Iterate through the templates, generating the template instance
        that most improves the current tagging.
        Return the best transform: ( transform instance, score )
        """
        best_transform = ( NIL_TEMPLATE, 0, )
        for template in self.templates:
            instance, score = self.get_best_instance(template)
            if (score > best_transform[1]):
                best_transform = ( instance, score, )
        print("--> ", end='')
        self.printTransform(best_transform)
        return best_transform

    # Test transform at every position across the corpus
    def test_transform_m1_loopy(self, fromTag, toTag,
            count_good_transforms, count_bad_transforms):
        """
        Tests the effect of the transform with the given tags.
        Scan the corpus and at each position test the effect
        of the transform.  Record the effect in the counts,
        good and bad, for the relevant tags z and w.
        """
        for pos in range(len(self.tagged_words)):
            correct_tag = self.tagged_words_true[pos][1]
            current_tag = self.tagged_words[pos][1]
            # Apply the default template criteria
            if pos > 0:
                preceding_tag = self.tagged_words[pos-1][1]
            else:
                preceding_tag = START_TAG
            if correct_tag == toTag and current_tag == fromTag:
                count_good_transforms[(preceding_tag, '',)] += 1
            if correct_tag == fromTag and current_tag == fromTag:
                count_bad_transforms[(preceding_tag, '',)] += 1


    # Get the best instance of a transform.
    def get_best_instance_loopy(self, template):
        """
        The best transform is an instance of a template
            [ type fromTag toTag zTag wTag description ]
        with the fromTag and toTag, and depending on the template type,
        the zTag and wTag filled in with specific tags,
        that produces the best overall improvement in the
        current tagging compared to the "ground truth" tagging.
        Input: a template
        Output: the best instance: ( instance, score )
        """
        best_transform = ( NIL_TEMPLATE, 0, )
        # counts of good and bad results for a transform instance
        # iterate over all tag pairs ...
        for fromTag in self.tagset:
            print(".", end='')
            for toTag in self.tagset:
                # test the default transform, context just preceding tag
                count_good_transforms = { (tag, '') : 0
                                          for tag in self.tagset }
                count_good_transforms[(START_TAG, '')] = 0
                count_bad_transforms  = { (tag, '') : 0
                                          for tag in self.tagset }
                count_bad_transforms[(START_TAG, '')] = 0
                self.test_transform_m1(fromTag, toTag,
                        count_good_transforms, count_bad_transforms)
                # Find the best tags z and w among those encountered,
                # the one that produced the most improvements in the
                # current tagging.
                bestZ = START_TAG
                bestScore = 0
                for zTag in self.tagset: # for now, just iterate over z
                    score = count_good_transforms[(zTag, '',)] \
                           - count_bad_transforms[(zTag, '',)]
                    if score > bestScore:
                        bestScore = score
                        bestZ = zTag
                if bestScore > best_transform[1]:
                    instance = [ DEFAULT_TEMPLATE[0],
                                 fromTag, toTag, bestZ, '',
                                 DEFAULT_TEMPLATE[5]
                               ]
                    best_transform = ( instance, bestScore, )
                    print()
                    self.printTransform(best_transform)
        print()
        return best_transform

    # Test transform at every position across the corpus
    def test_transform_m1(self):
        """
        Tests the effect of the transform across the corpus.
        Scan the corpus and at each position test the effect
        of the transform.  Record the effect in the counts,
        good and bad, for the relevant tags z and w.
        """
        count_good_transforms = defaultdict(int)
        count_bad_transforms  = defaultdict(int)
        preceding_tag = START_TAG
        for current_tag, correct_tag in self.tags:
            for toTag in self.tagset:
                transform_id = (current_tag, toTag, preceding_tag, '',)
                # Apply the default template criteria
                if correct_tag == current_tag:
                    count_bad_transforms[transform_id] += 1
                if correct_tag == toTag:
                    count_good_transforms[transform_id] += 1
            preceding_tag = current_tag
        return count_good_transforms, count_bad_transforms

    # Get best instance of a transform
    def get_best_instance(self, template):
        """
        The best transform is an instance of a template
            [ type fromTag toTag zTag wTag description ]
        with the fromTag and toTag, and depending on the template type,
        the zTag and wTag filled in with specific tags,
        that produces the best overall improvement in the
        current tagging compared to the "ground truth" tagging.
        Input: a template
        Output: the best instance: ( instance, score )
        """
        best_transform = ( NIL_TEMPLATE, 0, )
        count_good_transforms, count_bad_transforms = self.test_transform_m1()
        for fromTag in self.tagset:
            print(".", end='')
            for toTag in self.tagset:
                # Find the best tags z and w among those encountered,
                # the one that produced the most improvements in the
                # current tagging.
                bestZ = START_TAG
                bestScore = 0
                for zTag in self.tagset: # for now, just iterate over z
                    transform_id = ( fromTag, toTag, zTag, '', )
                    if transform_id in count_good_transforms:
                        count_good = count_good_transforms[transform_id]
                    else:
                        count_good = 0 # avoid lengthening dictionary
                    if transform_id in count_bad_transforms:
                        count_bad = count_bad_transforms[transform_id]
                    else:
                        count_bad = 0 # avoid lengthening dictionary
                    score =  count_good - count_bad
                    if score > bestScore:
                        bestScore = score
                        bestZ = zTag
                if bestScore > best_transform[1]:
                    instance = [ DEFAULT_TEMPLATE[0],
                                 fromTag, toTag, bestZ, '',
                                 DEFAULT_TEMPLATE[5]
                               ]
                    best_transform = ( instance, bestScore, )
                    print()
                    self.printTransform(best_transform)
        print()
        return best_transform

        

    # Apply a transform instance to the corpus,
    # updating the current tagging.
    def apply_transform(self, transform):
        """
        Apply the transform with context just the preceding tag.
        """
        type, fromTag, toTag, zTag, wTag, desc = transform[0]
        previous_tag = START_TAG
        count_changes = 0
        for pos in range(len(self.tagged_words)):
            word, current_tag = self.tagged_words[pos]
            if pos > 0:
                previous_tag = self.tagged_words[pos-1][1]
            if current_tag == fromTag and previous_tag == zTag:
                self.tagged_words[pos] = ( word, toTag, )
                count_changes += 1
        print("-=-=-> changed ", count_changes, " tags")

    def get_mistagged_words(self):
        """
        Return a list of the currently mistagged words,
            [ ( word, incorrect tag, true tag, ), ... ]
        """
        mistagged = [ (self.tagged_words[i][0], \
                       self.tagged_words[i][1], \
                       self.tagged_words_true[i][1]) \
                for i in range(len(self.tagged_words)) \
                if self.tagged_words[i][1] \
                   != self.tagged_words_true[i][1] \
	    ]
        return mistagged

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':
    import nltk
    from nltk.corpus import treebank as tb
    from NLP_2ed_ch_5_problems import split_train_test
    from NLP_2ed_ch_5_problems import get_word_tag_counts
    from NLP_2ed_ch_5_problems import get_most_frequent_tag
    from random import random as rand
    from datetime import datetime


    # Run the tests 5 times
    for k in range(5):

        print("+++ TEST ", k+1, "+++")
        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

        # NLTK treebank corpus tests.
        tb_training_sents, tb_test_sents = split_train_test(tb.tagged_sents())
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
        tb_tagged_words = [ (w, tag, ) for s in tb.tagged_sents()
                                       for w, tag in s]
        tb_training_tagged_words = [ (w, tag, ) for s in tb_training_sents
                                                for w, tag in s]
        tb_test_tagged_words = [ (w, tag, ) for s in tb_test_sents
                                            for w, tag in s]
        tb_tagset = set(tag for word, tag in tb_tagged_words)
        tb_training_tagged_words_NN = [ (w, 'NN',)
                                        for w, tag in tb_training_tagged_words]
        tb_test_tagged_words_NN = [ (w, 'NN',)
                                    for w, tag in tb_test_tagged_words]
        tb_most_frequent_tags = get_most_frequent_tag(tb_word_tag_counts)
        tb_training_tagged_words_MF = [ (w, tb_most_frequent_tags[w],)
                                        for w, tag in tb_training_tagged_words]
        tb_test_tagged_words_MF = [ (w, tb_most_frequent_tags[w],)
                                    for w, tag in tb_test_tagged_words]
        tb_training_mistagged_MF = [ (tb_training_tagged_words_MF[i][0], \
                                      tb_training_tagged_words_MF[i][1], \
                                      tb_training_tagged_words[i][1]) \
                for i in range(len(tb_training_tagged_words_MF)) \
                if tb_training_tagged_words_MF[i][1] \
                   != tb_training_tagged_words[i][1] \
	    ]

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

        # Test class BrillTBL
        tagger = BrillTBL(tb_training_tagged_words_MF,
                          tb_training_tagged_words,
                          tb_tagset,
                          "NLTK Penn-Treebank initially tagged with most frequent tag")
        print("Description:   ", tagger.name)
        print("Initial Tagging:", len(tagger.tagged_words),
              tagger.tagged_words[:10])
        print("True Tagging:   ", len(tagger.tagged_words_true),
              tagger.tagged_words_true[:10])
        print("Tag Set:", len(tagger.tagset), tagger.tagset)
        print("---")
        mistagged_words_initial = tagger.get_mistagged_words()
        print("Initial count incorrectly tagged words:",
              len(mistagged_words_initial),
              "of", len(tagger.tagged_words))
        error_rate_initial = (1.0 * len(mistagged_words_initial)) \
            / len(tagger.tagged_words)
        print("Initial error rate:", error_rate_initial)
              
        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

        # Test test_transform_m1()
        """
        count_good_transforms = { (tag, '') : 0 for tag in tagger.tagset }
        count_good_transforms[(START_TAG, '')] = 0
        count_bad_transforms  = { (tag, '') : 0 for tag in tagger.tagset }
        count_bad_transforms[(START_TAG, '')] = 0
        tagger.test_transform_m1_loopy('NN', 'NNP',
                              count_good_transforms, count_bad_transforms)
        """
        count_good_transforms, count_bad_transforms = \
            tagger.test_transform_m1()
        good_z = [ (k,count_good_transforms[k])
                    for k in count_good_transforms
                    if count_good_transforms[k] > 0 ]
        
        bad_z  = [ (k,count_bad_transforms[k])
                    for k in count_bad_transforms
                    if count_bad_transforms[k] > 0 ]
        print('count good z:', len(good_z))
        print('count bad z:', len(bad_z))

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

        # Test learn()
        transform_sequence = tagger.learn()
        print("---- TRANSFORM SEQUENCE ----")
        print("Rules found:", len(transform_sequence))
        for rule in transform_sequence:
            tagger.printTransform(rule)
        mistagged_words_final = tagger.get_mistagged_words()
        print("Final count incorrectly tagged words:",
              len(mistagged_words_final),
              "of", len(tagger.tagged_words))
        error_rate_final = (1.0 * len(mistagged_words_final)) \
              / len(tagger.tagged_words)
        print("Final error rate:", error_rate_final)
        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

