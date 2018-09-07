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
    # The NIL template is the starting point for the best transform.
    # A template requires corresponding code in the tagger.
    DEFAULT_TEMPLATE =
        [ 'm1', '', '', '', '', "Change %s to %s if preceding tag is %s%s." ]
    NIL_TEMPLATE =
        [ 'NIL', '', '', '', '', "Don't change the tag%s%s%s%s." ]

    # Start of text tag, for positions preceding the beginning of the text.
    START_TAG = '<START>'

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
        self.tagset = tagset
        self.name = name
        self.templates = DEFAULT_TEMPLATE
        self.transforms_queue = []

    # Display a transform instance
    def printTransform(transform):
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
        """
        minimum_change = 10
        last_score = 0
        while True:
            best_transform = get_best_transform()
            apply_transform(best_transform, tagged_sents)
            self.transforms_queue += [ best_transform ]
            printTransform(best_transform)
            change_in_score = best_transform[1] - last_score
            if change_in_score < minimum_change:
                break
            last_score = best_transform[1]
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
            instance, score = get_best_instance(template)
            if (score > best_transform[1]):
                best_transform = ( instance, score, )
        return best_transform

    # Test transform at every position across the corpus
    def test_transform(template, fromTag, toTag,
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
                preceding_tag = self.tagged_words[pos-1][]
            else:
                preceding_tag = START_TAG
            if correct_tag == toTag and current_tag == fromTag:
                count_good_transforms[(preceding_tag, '',)] += 1
            if correct_tag == fromTag and current_tag == fromTag:
                count_bad_transforms[(preceding_tag, '',)] += 1


    # Get the best instance of a transform.
    def get_best_instance(template):
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
        count_good_transforms = { tags : 0 for tag in self.tagset }
        count_bad_transforms  = { tags : 0 for tag in self.tagset }
        for fromTag in self.tagset:
            for toTag in self.tagset:
                test_transform(template, fromTag, toTag,
                        count_good_transforms, count_bad_transforms)
                # Find the best tags z and w among those encountered,
                # the one that produced the most improvements in the
                # current tagging.
                bestZ = START_TAG
                bestScore = 0
                for zTag in self.tagset: # for now, just iterate over z
                    score = count_good_transforms[(zTag, '',)]
                           - count_bad_transforms[(zTag, '',)]
                    if score > bestScore
                        bestScore = score
                        bestZ = zTag
                if bestScore > best_transform[1]:
                    instance = [ DEFAULT_TEMPLATE[0],
                                 fromTag, toTag, bestZ, '',
                                 DEFAULT_TEMPLATE[5]
                               ]
                    best_transform = ( instance, bestScore, )
        return best_transform

        # Apply a transform instance to the corpus,
        # updating the current tagging.
    def apply_transform(self, transform):
        """
        Apply the transform with context just the preceding tag.
        """
        type, fromTag, toTag, zTag, wTag, desc = transform[0]
        previous_tag = START_TAG
        for pos in range(len(self.tagged_words)):
            current_tag = self.tagged_words[pos][1]
            if pos > 0
                previous_tag = self.tagged_words[pos-1][1]
            if current_tag == tagFrom and previous_tag == zTag:
                self.tagged_words[pos][1] = toTag
