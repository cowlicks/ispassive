"""
Adapted from https://github.com/j-c-h-e-n-g/nltk-passive-voice 

Who apparently adapted from http://narorumo.googlecode.com

Tags a sentence with a way-overkill four-level tagger trained from the Brown
Corpus, and then looks at its verbs. If somewhere in the sentence, there's a
to-be verb and then later on a non-gerund, we'll flag the sentence as probably
passive voice.

"""
import os
from pickle import dump, load

import nltk
from nltk.corpus import brown
from itertools import dropwhile


class Tagger:
    def __init__(self):
        if os.path.exists("tagger.pkl"):
            with open('tagger.pkl', 'rb') as data:
                tagger = load(data)
            self.tagger = tagger
        else:
            tagger = create_tagger()
            self.tagger = tagger
            self.save()

    def save(self):
        with open('tagger.pkl', 'wb') as output:
            dump(self.tagger, output, -1)

    def tag(self, sent):
        return self.tagger.tag(sent)

    def tag_sentence(self, sent):
        """Take a sentence as a string and return a list of (word, tag) tuples."""
        tokens = nltk.word_tokenize(sent)
        return self.tag(tokens)

    def is_passive(self, sent):
        return is_passive(self, sent)

def passivep(tags):
    """Takes a list of tags, returns true if we think this is a passive
    sentence.

    Particularly, if we see a "BE" verb followed by some other, non-BE
    verb, except for a gerund, we deem the sentence to be passive.
    """
    
    after_to_be = list(dropwhile(lambda tag: not tag.startswith("BE"), tags))
    nongerund = lambda tag: tag.startswith("V") and not tag.startswith("VBG")

    filtered = filter(nongerund, after_to_be)
    out = any(filtered)

    return out

def create_tagger():
    """Train a tagger from the Brown Corpus. This should not be called very
    often; only in the event that the tagger pickle wasn't found."""
    train_sents = brown.tagged_sents()

    # These regexes were lifted from the NLTK book tagger chapter.
    t0 = nltk.RegexpTagger(
        [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'), # cardinal numbers
         (r'(The|the|A|a|An|an)$', 'AT'), # articles
         (r'.*able$', 'JJ'),              # adjectives
         (r'.*ness$', 'NN'),              # nouns formed from adjectives
         (r'.*ly$', 'RB'),                # adverbs
         (r'.*s$', 'NNS'),                # plural nouns
         (r'.*ing$', 'VBG'),              # gerunds
         (r'.*ed$', 'VBD'),               # past tense verbs
         (r'.*', 'NN')                    # nouns (default)
        ])
    t1 = nltk.UnigramTagger(train_sents, backoff=t0)
    t2 = nltk.BigramTagger(train_sents, backoff=t1)
    t3 = nltk.TrigramTagger(train_sents, backoff=t2)
    return t3

def is_passive(tagger, sent):
    tagged = tagger.tag_sentence(sent)
    tags = map(lambda tup: tup[1], tagged)
    return bool(passivep(tags))

if __name__ == '__main__':
    t = Tagger()
    assert t.is_passive('Mistakes were made.')
    assert not t.is_passive('I made mistakes.')
    # Noteable fail case. Fix me. I think it is because the 'to be' verb is
    # omitted.
    #assert t.is_passive('guy shot by police')
