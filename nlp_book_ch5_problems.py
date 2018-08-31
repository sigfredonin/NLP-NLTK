"""
NLP problems 5.1 and 5.2

Tag sentences with Penn-Treebank using NLTK.pos_tag

Sig Nin
2018 Aug 30
"""

import nltk

def tag_sentence_PT(sentence):
    s_tokens = nltk.word_tokenize(sentence)
    s_pos = nltk.pos_tag(s_tokens)
    s_tagged = ""
    for tagged in [ nltk.tag.tuple2str(tuple) for tuple in s_pos ]:
        s_tagged += tagged + " "
    return s_tagged

p5_1_sentences = [
    "I need a flight from Atlanta.",
    "Does this flight serve dinner?",
    "I have a friend living in Denver.",
    "Can you list the nonstop afternoon flights?"
    ]

def problem_5(sentences, heading):
    ss_tagged = [ tag_sentence_PT(s) for s in sentences ]
    print(heading)
    for tagged in ss_tagged:
        print(tagged)
    return ss_tagged

p5_2_sentences = [
    "It is a nice night.",
    "This crap game is over a garage in Fifty-second Street.",
    "...Nobody ever takes the newspapers she sells ...",
    """
    He is a tall, skinny guy with a long, sad, mean-looking kisser,
    and a mournful voice.
    """,
    """
    ...I am sitting in Mindy's restaurant putting on the gefillte fish,
    which is a dish I am very fond of.
    """,
    """
    When a guy and a doll get to taking peeks back and forth at each other,
    why there you are indeed.
    """
    ]

def ch_5_problems():
    problem_5(p5_1_sentences, "Problem 5.1 NLTK tags")
    problem_5(p5_2_sentences, "Problem 5.2 NLTK tags")

