import os
import sys
import collections
import itertools
import cPickle as pickle

import math
import random
import numpy as np


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


def build_instances_training(candidate_sentences, distant_interactions,reverse_distant_interactions, entity_1_list = None, entity_2_list = None, symmetric = False):
    ''' Builds instances for training'''
    #initialize vocabularies for different features
    path_word_vocabulary = []
    words_between_entities_vocabulary = []
    dep_type_vocabulary = []
    dep_type_word_elements_vocabulary = []
    candidate_instances = []
    for candidate_sentence in candidate_sentences:
        entity_pairs = candidate_sentence.get_entity_pairs()

        for pair in entity_pairs:
            entity_1_token = candidate_sentence.get_token(pair[0])
            entity_2_token = candidate_sentence.get_token(pair[1])
            entity_1 = entity_1_token.get_normalized_ner().split('|')
            entity_2 = entity_2_token.get_normalized_ner().split('|')

            if entity_1_list is not None:
                if len(set(entity_1).intersection(entity_1_list)) == 0:
                    continue

                # check if entity_2 overlaps with entity_1_list if so continue
                if len(set(entity_2).intersection(entity_1_list)) > 0:
                    continue

            if entity_2_list is not None:
                if len(set(entity_2).intersection(entity_2_list)) == 0:
                    continue

                # check if entity_1 overlaps with entity_2_list if so continue
                if len(set(entity_1).intersection(entity_2_list)) > 0:
                    continue

            entity_combos = set(itertools.product(entity_1,entity_2))

            forward_train_instance = Instance(candidate_sentence, pair[0], pair[1], 0)
            reverse_train_instance = Instance(candidate_sentence, pair[1], pair[0], 0)

            if symmetric is False:
                # check if check returned true because of reverse
                if len(entity_combos.intersection(distant_interactions)) > 0:
                    path_word_vocabulary += forward_train_instance.get_dep_word_path()
                    words_between_entities_vocabulary += forward_train_instance.get_between_words()
                    dep_type_word_elements_vocabulary += forward_train_instance.get_dep_type_word_elements()
                    dep_type_vocabulary.append(' '.join(forward_train_instance.get_type_dependency_path()))
                    forward_train_instance.set_label(1)
                    candidate_instances.append(forward_train_instance)
                elif len(entity_combos.intersection(reverse_distant_interactions)) > 0:
                    path_word_vocabulary += reverse_train_instance.get_dep_word_path()
                    words_between_entities_vocabulary += reverse_train_instance.get_between_words()
                    dep_type_word_elements_vocabulary += reverse_train_instance.get_dep_type_word_elements()
                    dep_type_vocabulary.append(' '.join(reverse_train_instance.get_type_dependency_path()))
                    reverse_train_instance.set_label(1)
                    candidate_instances.append(reverse_train_instance)
                else:
                    path_word_vocabulary += forward_train_instance.get_dep_word_path()
                    path_word_vocabulary += reverse_train_instance.get_dep_word_path()
                    words_between_entities_vocabulary += forward_train_instance.get_between_words()
                    words_between_entities_vocabulary += reverse_train_instance.get_between_words()
                    dep_type_vocabulary.append(' '.join(forward_train_instance.get_type_dependency_path()))
                    dep_type_vocabulary.append(' '.join(reverse_train_instance.get_type_dependency_path()))
                    dep_type_word_elements_vocabulary += forward_train_instance.get_dep_type_word_elements()
                    dep_type_word_elements_vocabulary += reverse_train_instance.get_dep_type_word_elements()
                    candidate_instances.append(forward_train_instance)
                    candidate_instances.append(reverse_train_instance)


            #if symmetric is true
            else:
                if len(entity_combos.intersection(distant_interactions)) > 0 or \
                                len(entity_combos.intersection(reverse_distant_interactions)) > 0:
                    forward_train_instance.set_label(1)
                    reverse_train_instance.set_label(1)
                else:
                    pass
                dep_type_vocabulary_set = set(dep_type_vocabulary)
                forward_dep_type_path = ' '.join(forward_train_instance.get_type_dependency_path())
                reverse_dep_type_path = ' '.join(reverse_train_instance.get_type_dependency_path())

                if forward_dep_type_path in dep_type_vocabulary_set:
                    dep_type_vocabulary.append(forward_dep_type_path)
                    path_word_vocabulary += forward_train_instance.get_dep_word_path()
                    dep_type_word_elements_vocabulary += forward_train_instance.get_dep_type_word_elements()
                    words_between_entities_vocabulary += forward_train_instance.get_between_words()
                    candidate_instances.append(forward_train_instance)
                elif reverse_dep_type_path in dep_type_vocabulary_set:
                    dep_type_vocabulary.append(reverse_dep_type_path)
                    path_word_vocabulary += reverse_train_instance.get_dep_word_path()
                    dep_type_word_elements_vocabulary += reverse_train_instance.get_dep_type_word_elements()
                    words_between_entities_vocabulary += reverse_train_instance.get_between_words()
                    candidate_instances.append(reverse_train_instance)
                else:
                    dep_type_vocabulary.append(forward_dep_type_path)
                    path_word_vocabulary += forward_train_instance.get_dep_word_path()
                    dep_type_word_elements_vocabulary += forward_train_instance.get_dep_type_word_elements()
                    words_between_entities_vocabulary += forward_train_instance.get_between_words()
                    candidate_instances.append(forward_train_instance)


    data, count, dep_path_word_dictionary, reversed_dictionary = build_dataset(path_word_vocabulary)
    dep_data, dep_count, dep_dictionary, dep_reversed_dictionary = build_dataset(dep_type_vocabulary)
    dep_element_data, dep_element_count, dep_element_dictionary, dep_element_reversed_dictionary = build_dataset(dep_type_word_elements_vocabulary)
    between_data, between_count, between_word_dictionary, between_reversed_dictionary = build_dataset(words_between_entities_vocabulary)

    '''
    dep_path_word_dictionary = {}
    array_place = 0
    for c in count:
        dep_path_word_dictionary[c[0]] = array_place
        array_place += 1

    between_word_dictionary = {}
    array_place = 0
    for c in between_count:
        between_word_dictionary[c[0]] = array_place
        array_place += 1
    '''
    print(dep_dictionary)
    print(dep_path_word_dictionary)
    print(between_word_dictionary)
    print(dep_element_dictionary)

    for ci in candidate_instances:
        ci.build_features(dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary)

    return candidate_instances, dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary

def build_instances_testing(test_sentences, dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary,
                            distant_interactions,reverse_distant_interactions, entity_1_list =  None, entity_2_list = None, symmetric = False):
    test_instances = []
    for test_sentence in test_sentences:
        entity_pairs = test_sentence.get_entity_pairs()

        for pair in entity_pairs:
            entity_1_token = test_sentence.get_token(pair[0])
            entity_2_token = test_sentence.get_token(pair[1])
            entity_1 = entity_1_token.get_normalized_ner().split('|')
            entity_2 = entity_2_token.get_normalized_ner().split('|')
            if entity_1_list is not None:
                if len(set(entity_1).intersection(entity_1_list)) == 0:
                    continue

                #check if entity_2 overlaps with entity_1_list if so continue
                if len(set(entity_2).intersection(entity_1_list)) > 0:
                    continue

            if entity_2_list is not None:
                if len(set(entity_2).intersection(entity_2_list)) == 0:
                    continue
                #check if entity_1 overlaps with entity_2_list if so continue
                if len(set(entity_1).intersection(entity_2_list)) > 0:
                   continue

            entity_combos = set(itertools.product(entity_1,entity_2))
            forward_test_instance = Instance(test_sentence, pair[0], pair[1], 0)
            reverse_test_instance = Instance(test_sentence, pair[1], pair[0], 0)


            if symmetric is False:

                # check if check returned true because of reverse
                if len(entity_combos.intersection(distant_interactions)) > 0 :
                    forward_test_instance.set_label(1)
                elif len(entity_combos.intersection(reverse_distant_interactions)) > 0:
                    reverse_test_instance.set_label(1)
                else:
                    pass

                test_instances.append(forward_test_instance)
                test_instances.append(reverse_test_instance)

            #if symmetric is True
            else:
                if len(entity_combos.intersection(distant_interactions)) > 0 or \
                        len(entity_combos.intersection(reverse_distant_interactions)) > 0:
                    forward_test_instance.set_label(1)
                    reverse_test_instance.set_label(1)

                forward_dep_type_path = ' '.join(forward_test_instance.get_type_dependency_path())
                reverse_dep_type_path = ' '.join(reverse_test_instance.get_type_dependency_path())

                if forward_dep_type_path in dep_dictionary:
                    test_instances.append(forward_test_instance)
                elif reverse_dep_type_path in dep_dictionary:
                    test_instances.append(reverse_test_instance)
                else:
                    test_instances.append(forward_test_instance)


    for instance in test_instances:
        instance.build_features(dep_dictionary, dep_path_word_dictionary, dep_element_dictionary,  between_word_dictionary)

    return test_instances

def build_instances_predict(predict_sentences, dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary, entity_1_list = None, entity_2_list = None, symmetric = False):
    predict_instances = []
    for p_sentence in predict_sentences:
        entity_pairs = p_sentence.get_entity_pairs()

        for pair in entity_pairs:
            entity_1_token = p_sentence.get_token(pair[0])
            entity_2_token = p_sentence.get_token(pair[1])
            entity_1 = entity_1_token.get_normalized_ner().split('|')
            entity_2 = entity_2_token.get_normalized_ner().split('|')
            if entity_1_list is not None:
                if len(set(entity_1).intersection(entity_1_list)) == 0:
                    continue

                #check if entity_2 overlaps with entity_1_list if so continue
                if len(set(entity_2).intersection(entity_1_list)) > 0:
                    continue

            if entity_2_list is not None:
                if len(set(entity_2).intersection(entity_2_list)) == 0:
                    continue
                #check if entity_1 overlaps with entity_2_list if so continue
                if len(set(entity_1).intersection(entity_2_list)) > 0:
                   continue

            forward_predict_instance = Instance(p_sentence, pair[0], pair[1], -1)
            reverse_predict_instance = Instance(p_sentence, pair[1], pair[0], -1)
            if symmetric is False:
                predict_instances.append(forward_predict_instance)
                predict_instances.append(reverse_predict_instance)
            #if symmetric is True
            else:
                forward_dep_type_path = ' '.join(forward_predict_instance.get_type_dependency_path())
                reverse_dep_type_path = ' '.join(reverse_predict_instance.get_type_dependency_path())

                if forward_dep_type_path in dep_dictionary:
                    predict_instances.append(forward_predict_instance)
                elif reverse_dep_type_path in dep_dictionary:
                    predict_instances.append(reverse_predict_instance)
                else:
                    predict_instances.append(forward_predict_instance)


    for instance in predict_instances:
        instance.build_features(dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary)

    return predict_instances

def load_xml(xml_file, entity_1, entity_2):
    '''loads xml file for sentences, stanford corenlp parsed format only'''
    tree = etree.parse(xml_file)
    root = tree.getroot()
    candidate_sentences = []
    sentences = list(root.iter('sentence'))


    for sentence in sentences:
        candidate_sentence = Sentence(sentence.get('id')) #get candidate sentence
        tokens = list(sentence.iter('token')) #get tokens for sentence

        for token in tokens:
            normalized_ner = None
            ner = token.find('NER').text
            if token.find('NormalizedNER') is not None:
                normalized_ner = token.find('NormalizedNER').text
            #create token objects for sentences. Use to get word, lemma, POS, etc.
            candidate_token = Token(token.get('id'), token.find('word').text, token.find('lemma').text, token.find('CharacterOffsetBegin').text,
                                    token.find('CharacterOffsetEnd').text, token.find('POS').text, ner, normalized_ner)
            candidate_sentence.add_token(candidate_token)
        #gets dependencies between tokens from stanford dependency parse
        dependencies = list(sentence.iter('dependencies'))
        basic_dependencies = dependencies[0]
        #list of all dependencies in sentence
        deps = list(basic_dependencies.iter('dep'))
        #generates list of all dependencies within a sentence
        for d in deps:
            candidate_dep = Dependency(d.get('type'), candidate_sentence.get_token(d.find('governor').get('idx')), candidate_sentence.get_token(d.find('dependent').get('idx')))
            candidate_sentence.add_dependency(candidate_dep)
        #gets entity pairs of sentence
        candidate_sentence.generate_entity_pairs(entity_1, entity_2)
        if candidate_sentence.get_entity_pairs() is not None:
        #generates dependency matrix
            candidate_sentence.build_dependency_matrix()
            candidate_sentences.append(candidate_sentence)

    return candidate_sentences


def load_distant_kb(distant_kb_file, column_a, column_b,distant_rel_col):
    '''Loads data from knowledge base into tuples'''
    distant_interactions = set()
    reverse_distant_interactions = set()
    #reads in lines from kb file
    file = open(distant_kb_file,'rU')
    lines = file.readlines()
    file.close()
    for l in lines:
        split_line = l.split('\t')
        #column_a is entity_1 column_b is entity 2
        tuple = (split_line[column_a],split_line[column_b])
        if split_line[distant_rel_col].endswith('by') is False:
            distant_interactions.add(tuple)
        else:
            reverse_distant_interactions.add(tuple)

    #returns both forward and backward tuples for relations
    return distant_interactions,reverse_distant_interactions

def load_id_list(id_list,column_a):
    '''loads normalized ids for entities, only called if file given'''
    id_set = set()
    file = open(id_list,'rU')
    lines = file.readlines()
    file.close()

    for l in lines:
        split_line = l.split('\t')
        id_set.add(split_line[column_a])

    return id_set

def load_abstracts_from_directory(directory_folder,entity_1,entity_2):
    abstract_dict = {}
    count = 0
    for path, subdirs, files in os.walk(directory_folder):
        for name in files:
            if name.endswith('.xml'):
                xmlpath = os.path.join(path, name)
                abstract_sentences = load_xml(xmlpath,entity_1,entity_2)
                if len(abstract_sentences) > 0:
                    abstract_dict[count] = abstract_sentences
                    print(count)
                    count += 1
            else:
                continue
    #save dictionary to pickle file so you don't have to read them in every time.
    pickle.dump(abstract_dict, open(directory_folder+'.pkl', "wb"))
    return abstract_dict

def load_abstracts_from_pickle(pickle_file):
    abstract_dict = pickle.load( open(pickle_file, "rb" ) )
    return abstract_dict