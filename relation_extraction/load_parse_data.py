import os
import sys
import collections
import math
import random
import numpy as np
import tensorflow as tf

from six.moves import xrange  # pylint: disable=redefined-builtin
from lxml import etree

from structures.sentence_structure import Sentence, Token, Dependency
from structures.instances import Instance


def build_dataset(words, occur_count = None):
    """Process raw inputs into a dataset."""
    num_total_words = len(set(words))
    discard_count = 0
    if occur_count is not None:
        for c in collections.Counter(words):
            if collections.Counter(words)[c] < occur_count:
                discard_count +=1

    num_words = num_total_words - discard_count

    count = []
    count.extend(collections.Counter(words).most_common(num_words))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    for word in words:
        if word in dictionary:
            index = dictionary[word]
            data.append(index)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def feature_construction(dep_type_vocabulary, word_vocabulary, candidate_sentences):

    data, count, dictionary, reversed_dictionary = build_dataset(word_vocabulary, 10)
    dep_data, dep_count, dep_dictionary, dep_reversed_dictionary = build_dataset(dep_type_vocabulary)


    common_words_file = open('./static_data/common_words.txt','rU')
    lines = common_words_file.readlines()
    common_words_file.close()

    common_words = set()
    for l in lines:
        common_words.add(l.split()[0])

    word_dictionary = {}
    array_place = 0
    for c in count:
        if c[0] not in common_words:
            word_dictionary[c[0]] = array_place
            array_place += 1


    for cs in candidate_sentences:
        cs.build_features(word_dictionary, dep_dictionary)

    return candidate_sentences, word_dictionary, dep_dictionary

def load_xml(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    candidate_sentences = []
    sentences = list(root.iter('sentence'))

    all_word_vocabulary = []
    all_dep_type_vocabulary = []


    for sentence in sentences:
        candidate_sentence = Sentence(sentence.get('id'))
        tokens = list(sentence.iter('token'))
        elements = set()
        if sentence.find('interaction') is not None:
            interactions = sentence.find('interaction').text.split('|')
            for i in interactions:
                i_elements = set(i.split('-'))
                elements = elements.union(i_elements)


        for token in tokens:
            normalized_ner = None
            if token.find('NormalizedNER') is not None:
                normalized_ner = token.find('NormalizedNER').text

            candidate_token = Token(token.get('id'), token.find('word').text, token.find('lemma').text, token.find('CharacterOffsetBegin').text,
                                    token.find('CharacterOffsetEnd').text, token.find('POS').text, token.find('NER').text, normalized_ner)
            candidate_sentence.add_token(candidate_token)

        dependencies = list(sentence.iter('dependencies'))
        basic_dependencies = dependencies[0]
        deps = list(basic_dependencies.iter('dep'))
        for d in deps:
            candidate_dep = Dependency(d.get('type'), candidate_sentence.get_token(d.find('governor').get('idx')), candidate_sentence.get_token(d.find('dependent').get('idx')))
            candidate_sentence.add_dependency(candidate_dep)

        candidate_sentence.generate_entity_pairs('HUMAN_GENE','VIRAL_GENE')
        entity_pairs = candidate_sentence.get_entity_pairs()
        candidate_sentence.build_dependency_matrix()

        for pair in entity_pairs:
            if candidate_sentence.tokens[pair[0]].get_normalized_ner() in elements and candidate_sentence.tokens[pair[1]].get_normalized_ner() in elements:
                candidate_instance = Instance(candidate_sentence, pair[0], pair[1], 1)
                candidate_sentences.append(candidate_instance)
                all_word_vocabulary += candidate_instance.get_word_path()
                all_dep_type_vocabulary.append(''.join(candidate_instance.get_type_dependency_path()))
            else:
                candidate_instance = Instance(candidate_sentence, pair[0], pair[1], 0)
                candidate_sentences.append(candidate_instance)

    return all_dep_type_vocabulary, all_word_vocabulary, candidate_sentences