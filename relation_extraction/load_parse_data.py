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


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def feature_construction(dep_type_vocabulary, word_vocabulary, candidate_sentences):
    dep_type_vocabulary_size = int(len(set(dep_type_vocabulary)))
    word_vocabulary_size = int(len(set(word_vocabulary)))

    data, count, dictionary, reversed_dictionary = build_dataset(word_vocabulary, word_vocabulary_size)
    dep_data, dep_count, dep_dictionary, dep_reversed_dictionary = build_dataset(dep_type_vocabulary, dep_type_vocabulary_size)

    common_words_file = open('./static_data/common_words.txt','rU')
    lines = common_words_file.readlines()
    common_words_file.close()

    common_words = set()
    for l in lines:
        common_words.add(l.split()[0])

    feature_words = set()
    feature_pos_array = {}
    array_place = 0
    for c in count:
        if c[0] not in common_words and int(c[1]) >= 10:
            feature_words.add(c[0])
            feature_pos_array[c[0]] = array_place
            array_place += 1

    for c in candidate_sentences:
        c.build_features(feature_words,feature_pos_array, dep_dictionary)

    return candidate_sentences

def load_xml(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    candidate_sentences = []
    sentences = list(root.iter('sentence'))

    word_vocabulary = []
    dep_type_vocabulary = []


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
                candidate_instance = Instance(candidate_sentence, pair[0], pair[1], 'Positive')
                candidate_sentences.append(candidate_instance)
                word_vocabulary += candidate_instance.get_word_path()
                dep_type_vocabulary.append(''.join(candidate_instance.get_type_dependency_path()))
            else:
                candidate_instance = Instance(candidate_sentence, pair[0], pair[1], 'Negative')
                candidate_sentences.append(candidate_instance)

    return dep_type_vocabulary, word_vocabulary, candidate_sentences