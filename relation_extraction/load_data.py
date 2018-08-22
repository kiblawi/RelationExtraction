import os
import sys
import collections
import itertools
import cPickle as pickle

import math
import random
import numpy as np

import tensorflow as tf



from lxml import etree

from structures.sentence_structure import Sentence, Token, Dependency
from structures.instances import Instance


def np_to_tfrecord(features,labels,tfresult_file):

    writer = tf.python_io.TFRecordWriter(tfresult_file)
    #print(features.shape[0])
    for i in range(features.shape[0]):
        x = features[i]
        x= np.array(x,dtype='int8')
        x=x.tobytes()
        #print(x.shape)
        y = labels[i]
        y = np.array(y,dtype='int8')
        y = y.tobytes()

        feature_dict = {}
        feature_dict['x'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))
        feature_dict['y'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[y]))
        #print(feature_dict['x'])


        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        serialized=example.SerializeToString()
        #print(serialized)
        writer.write(serialized)
    writer.close()

    return tfresult_file

def np_to_lstm_tfrecord(dep_path_list_features,dep_word_features,dep_type_path_length,
                                                         dep_word_path_length,labels,tfresult_file):

    writer = tf.python_io.TFRecordWriter(tfresult_file)
    #print(features.shape[0])
    for i in range(labels.shape[0]):
        dep_path_list_feat = dep_path_list_features[i]
        dep_path_list_feat = np.array(dep_path_list_feat,dtype='int32')
        dep_path_list_feat=dep_path_list_feat.tobytes()

        dep_word_feat = dep_word_features[i]
        dep_word_feat = np.array(dep_word_feat, dtype='int32')
        dep_word_feat = dep_word_feat.tobytes()

        dep_path_length = dep_type_path_length[i]
        dep_path_length = np.array(dep_path_length,dtype='int32')
        dep_path_length = dep_path_length.tobytes()

        dep_word_length = dep_word_path_length[i]
        dep_word_length = np.array(dep_word_length,dtype='int32')
        dep_word_length = dep_word_length.tobytes()

        y = labels[i]
        y = np.array(y,dtype='int32')
        y = y.tobytes()

        feature_dict = {}
        feature_dict['dep_path_list'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[dep_path_list_feat]))
        feature_dict['dep_word_feat'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[dep_word_feat]))
        feature_dict['dep_path_length'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[dep_path_length]))
        feature_dict['dep_word_length'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[dep_word_length]))
        feature_dict['y'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[y]))
        #print(feature_dict['x'])


        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        serialized=example.SerializeToString()
        #print(serialized)
        writer.write(serialized)
    writer.close()

    return tfresult_file




def build_dataset(words, occur_count = None):
    """Process raw mentions of features into dictionary and count dictionary"""
    num_total_words = len(set(words))
    discard_count = 0
    if occur_count is not None:
        word_count_dict = collections.Counter(words)
        discard_count = sum(1 for i in word_count_dict.values() if i < occur_count)
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

def feature_pruning(feature_dict,feature_count_tuples,prune_val):
    """Feature pruning if not done earlier - Don't really need this  function"""
    feature_count_dict = dict(feature_count_tuples)
    for key, value in feature_count_dict.iteritems():
        if value < prune_val:
            popped_element = feature_dict.pop(key)

    return feature_dict

def build_instances_training(candidate_sentences, distant_interactions,reverse_distant_interactions,key_order, entity_1_list = None, entity_2_list = None):
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
            entity_1_token = candidate_sentence.get_token(pair[0][0])
            entity_2_token = candidate_sentence.get_token(pair[1][0])
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
            #print(entity_combos)

            forward_train_instance = Instance(candidate_sentence, pair[0], pair[1], [0]*len(key_order))
            #print(forward_train_instance.dependency_elements)
            reverse_train_instance = Instance(candidate_sentence, pair[1], pair[0], [0]*len(key_order))
            #print(reverse_train_instance.dependency_elements)

            for i in range(len(key_order)):
                distant_key = key_order[i]
                if 'SYMMETRIC' in distant_key:
                    if len(entity_combos.intersection(distant_interactions[distant_key]))>0 or len(entity_combos.intersection(reverse_distant_interactions[distant_key]))>0:
                        forward_train_instance.set_label_i(1,i)
                        reverse_train_instance.set_label_i(1,i)
                else:
                    if len(entity_combos.intersection(distant_interactions[distant_key])) > 0:
                        forward_train_instance.set_label_i(1, i)
                    elif len(entity_combos.intersection(reverse_distant_interactions[distant_key]))>0:
                        reverse_train_instance.set_label_i(1, i)

            path_word_vocabulary += forward_train_instance.dependency_words
            path_word_vocabulary += reverse_train_instance.dependency_words
            words_between_entities_vocabulary += forward_train_instance.between_words
            words_between_entities_vocabulary += reverse_train_instance.between_words
            dep_type_word_elements_vocabulary += forward_train_instance.dependency_elements
            dep_type_word_elements_vocabulary += reverse_train_instance.dependency_elements
            dep_type_vocabulary.append(forward_train_instance.dependency_path_string)
            dep_type_vocabulary.append(reverse_train_instance.dependency_path_string)
            candidate_instances.append(forward_train_instance)
            candidate_instances.append(reverse_train_instance)


    data, count, dep_path_word_dictionary, reversed_dictionary = build_dataset(path_word_vocabulary,100)
    dep_data, dep_count, dep_dictionary, dep_reversed_dictionary = build_dataset(dep_type_vocabulary,100)
    dep_element_data, dep_element_count, dep_element_dictionary, dep_element_reversed_dictionary = build_dataset(
        dep_type_word_elements_vocabulary,100)
    between_data, between_count, between_word_dictionary, between_reversed_dictionary = build_dataset(
        words_between_entities_vocabulary,100)


    print(dep_dictionary)
    print(dep_path_word_dictionary)
    print(between_word_dictionary)
    print(dep_element_dictionary)

    for ci in candidate_instances:
        ci.build_features(dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary)

    return candidate_instances, dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary

def build_instances_testing(test_sentences, dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary,
                            distant_interactions,reverse_distant_interactions, key_order, entity_1_list =  None, entity_2_list = None,dep_path_type_dictionary=None):

    test_instances = []
    for test_sentence in test_sentences:
        entity_pairs = test_sentence.get_entity_pairs()

        for pair in entity_pairs:
            entity_1_token = test_sentence.get_token(pair[0][0])
            entity_2_token = test_sentence.get_token(pair[1][0])
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
            forward_test_instance = Instance(test_sentence, pair[0], pair[1], [0] *len(key_order))
            reverse_test_instance = Instance(test_sentence, pair[1], pair[0], [0] *len(key_order))


            for i in range(len(key_order)):
                distant_key = key_order[i]
                if 'SYMMETRIC' in distant_key:
                    if len(entity_combos.intersection(distant_interactions[distant_key]))>0 or len(entity_combos.intersection(reverse_distant_interactions[distant_key]))>0:
                        forward_test_instance.set_label_i(1,i)
                        reverse_test_instance.set_label_i(1,i)

                else:
                    if len(entity_combos.intersection(distant_interactions[distant_key])) > 0:
                        forward_test_instance.set_label_i(1, i)
                    elif len(entity_combos.intersection(reverse_distant_interactions[distant_key]))>0:
                        reverse_test_instance.set_label_i(1, i)

            test_instances.append(forward_test_instance)
            #test_instances.append(reverse_test_instance)

    if dep_path_type_dictionary is None:
        for instance in test_instances:
            instance.build_features(dep_dictionary, dep_path_word_dictionary, dep_element_dictionary,  between_word_dictionary)
    else:
        for instance in test_instances:
            instance.build_lstm_features(dep_path_type_dictionary,dep_path_word_dictionary)


    return test_instances

def build_instances_predict(predict_sentences,dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary,key_order, entity_1_list = None, entity_2_list = None):

    predict_instances = []
    for p_sentence in predict_sentences:

        entity_pairs = p_sentence.get_entity_pairs()

        for pair in entity_pairs:
            entity_1_token = p_sentence.get_token(pair[0][0])
            entity_2_token = p_sentence.get_token(pair[1][0])
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

            forward_predict_instance = Instance(p_sentence, pair[0], pair[1], [-1]*len(key_order))
            reverse_predict_instance = Instance(p_sentence, pair[1], pair[0], [-1]*len(key_order))

            predict_instances.append(forward_predict_instance)


    for instance in predict_instances:
        instance.build_features(dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary)

    return predict_instances

def load_xml(xml_file, entity_1, entity_2):
    '''loads xml file for sentences, stanford corenlp parsed format only'''
    tree = etree.parse(xml_file)
    root = tree.getroot()
    candidate_sentences = []
    sentences = list(root.iter('sentence'))
    pmids = set()

    for sentence in sentences:
        candidate_sentence = Sentence(sentence.find('PMID').text,sentence.get('id')) #get candidate sentence find for pmid because its a tag, get for 'id' because its an attribute
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
        # generates dependency matrix
        candidate_sentence.build_dependency_matrix()
        #gets entity pairs of sentence
        candidate_sentence.generate_entity_pairs(entity_1, entity_2)
        if candidate_sentence.get_entity_pairs() is not None:
            candidate_sentences.append(candidate_sentence)
            pmids.add(candidate_sentence.pmid)

    return candidate_sentences, pmids


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
    print(directory_folder)
    total_abstract_sentences = []
    total_pmids = set()
    for path, subdirs, files in os.walk(directory_folder):
        for name in files:
            if name.endswith('.txt'):
                print(name)
                xmlpath = os.path.join(path, name)
                abstract_sentences,pmids = load_xml(xmlpath,entity_1,entity_2)
                if len(abstract_sentences) > 0:
                    total_abstract_sentences += abstract_sentences
                    total_pmids = total_pmids.union(pmids)

            else:
                continue
    #save dictionary to pickle file so you don't have to read them in every time.
    #pickle.dump(abstract_dict, open(directory_folder+'.pkl', "wb"))
    print(len(total_abstract_sentences))
    return total_pmids,total_abstract_sentences

def load_abstracts_from_pickle(pickle_file):
    abstract_dict = pickle.load( open(pickle_file, "rb" ) )
    return abstract_dict


def load_distant_directories(directional_distant_directory,symmetric_distant_directory,distant_entity_a_col,distant_entity_b_col,distant_rel_col):
    forward_dictionary = {}
    reverse_dictionary = {}
    for filename in os.listdir(directional_distant_directory):
        if filename.endswith('.txt') is False:
            continue
        distant_interactions,reverse_distant_interactions = load_distant_kb(directional_distant_directory+'/'+filename,
                                                                            distant_entity_a_col,distant_entity_b_col,distant_rel_col)
        forward_dictionary[filename] = distant_interactions
        reverse_dictionary[filename] = reverse_distant_interactions

    for filename in os.listdir(symmetric_distant_directory):
        if filename.endswith('.txt') is False:
            continue
        distant_interactions,reverse_distant_interactions = load_distant_kb(symmetric_distant_directory+'/'+filename,
                                                                            distant_entity_a_col,distant_entity_b_col,distant_rel_col)
        forward_dictionary['SYMMETRIC'+filename] = distant_interactions
        reverse_dictionary['SYMMETRIC'+filename] = reverse_distant_interactions

    return forward_dictionary, reverse_dictionary


def build_dictionaries_from_directory(directory_folder,entity_a,entity_b, entity_1_list=None,entity_2_list=None,LSTM=False):
    print(directory_folder)
    path_word_vocabulary = []
    words_between_entities_vocabulary = []
    dep_type_vocabulary = []
    dep_type_word_elements_vocabulary = []
    dep_type_list_vocabulary = []

    total_pmids = set()
    for path, subdirs, files in os.walk(directory_folder):
        for name in files:
            if name.endswith('.txt'):
                print(name)
                xmlpath = os.path.join(path, name)
                abstract_sentences, pmids = load_xml(xmlpath, entity_a, entity_b)
                for candidate_sentence in abstract_sentences:
                    entity_pairs = candidate_sentence.get_entity_pairs()

                    for pair in entity_pairs:
                        entity_1_token = candidate_sentence.get_token(pair[0][0])
                        entity_2_token = candidate_sentence.get_token(pair[1][0])
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

                        entity_combos = set(itertools.product(entity_1, entity_2))
                        # print(entity_combos)

                        forward_train_instance = Instance(candidate_sentence, pair[0], pair[1], None)
                        # print(forward_train_instance.dependency_elements)
                        reverse_train_instance = Instance(candidate_sentence, pair[1], pair[0], None)

                        #get vocabs
                        path_word_vocabulary += forward_train_instance.dependency_words
                        path_word_vocabulary += reverse_train_instance.dependency_words
                        words_between_entities_vocabulary += forward_train_instance.between_words
                        words_between_entities_vocabulary += reverse_train_instance.between_words
                        dep_type_word_elements_vocabulary += forward_train_instance.dependency_elements
                        dep_type_word_elements_vocabulary += reverse_train_instance.dependency_elements
                        dep_type_list_vocabulary += forward_train_instance.dependency_path_list
                        dep_type_list_vocabulary += reverse_train_instance.dependency_path_list
                        dep_type_vocabulary.append(forward_train_instance.dependency_path_string)
                        dep_type_vocabulary.append(reverse_train_instance.dependency_path_string)


            else:
                continue

    data, count, dep_path_word_dictionary, reversed_dictionary = build_dataset(path_word_vocabulary,100)
    dep_data, dep_count, dep_dictionary, dep_reversed_dictionary = build_dataset(dep_type_vocabulary,100)
    dep_element_data, dep_element_count, dep_element_dictionary, dep_element_reversed_dictionary = build_dataset(
        dep_type_word_elements_vocabulary,100)
    between_data, between_count, between_word_dictionary, between_reversed_dictionary = build_dataset(
        words_between_entities_vocabulary,100)
    dep_type_list_data, dep_type_list_count, dep_type_list_dictionary, dep_type_list_reversed_dictionary = build_dataset(
        dep_type_list_vocabulary, 0)

    if LSTM is False:
        return dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary

    else:
        return dep_type_list_dictionary,dep_path_word_dictionary


def build_instances_from_directory(directory_folder, entity_a, entity_b, dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary,
                                   distant_interactions, reverse_distant_interactions, key_order):
    total_dataset= []
    if os.path.isdir(directory_folder+'_tf_record') == False:
        os.mkdir(directory_folder+'_tf_record')
    for path, subdirs, files in os.walk(directory_folder):
        for name in files:
            if name.endswith('.txt'):
                #print(name)
                xmlpath = os.path.join(path, name)
                test_sentences, pmids = load_xml(xmlpath, entity_a, entity_b)
                candidate_instances = build_instances_testing(test_sentences, dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary,
                            distant_interactions,reverse_distant_interactions, key_order, entity_1_list =  None, entity_2_list = None)

                X = []
                y = []
                for ci in candidate_instances:
                    X.append(ci.features)
                    y.append(ci.label)
                features = np.array(X)
                labels = np.array(y)


                tfrecord_filename = name.replace('.txt','.tfrecord')

                total_dataset.append(np_to_tfrecord(features,labels,directory_folder +'_tf_record/'+ tfrecord_filename))

    return total_dataset

def build_LSTM_instances_from_directory(directory_folder, entity_a, entity_b, dep_type_list_dictionary, dep_path_word_dictionary,
                                        distant_interactions, reverse_distant_interactions, key_order):
    total_dataset= []
    if os.path.isdir(directory_folder+'_lstm_tf_record') == False:
        os.mkdir(directory_folder+'_lstm_tf_record')
    for path, subdirs, files in os.walk(directory_folder):
        for name in files:
            if name.endswith('.txt'):
                #print(name)
                xmlpath = os.path.join(path, name)
                test_sentences, pmids = load_xml(xmlpath, entity_a, entity_b)
                candidate_instances = build_instances_testing(test_sentences, None, dep_path_word_dictionary, None, None,
                                                              distant_interactions, reverse_distant_interactions, key_order, entity_1_list =  None, entity_2_list = None,dep_path_type_dictionary=dep_type_list_dictionary)

                dep_path_list_features = []
                dep_word_features = []
                dep_type_path_length = []
                dep_word_path_length = []
                labels = []
                instance_sentences = set()
                for t in candidate_instances:
                    # instance_sentences.add(' '.join(t.sentence.sentence_words))
                    dep_path_list_features.append(t.features[0:20])
                    dep_word_features.append(t.features[20:40])
                    dep_type_path_length.append(t.features[40])
                    dep_word_path_length.append(t.features[41])
                    labels.append(t.label)


                tfrecord_filename = name.replace('.txt','.tfrecord')

                total_dataset.append(np_to_lstm_tfrecord(dep_path_list_features,dep_word_features,dep_type_path_length,
                                                         dep_word_path_length,labels,directory_folder +'_tf_record/'+ tfrecord_filename))

    return total_dataset

def build_test_instances_from_directory(directory_folder, entity_a, entity_b, dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary,
                                   distant_interactions, reverse_distant_interactions, key_order):

    total_features = []
    total_labels = []
    total_instances = []
    for path, subdirs, files in os.walk(directory_folder):
        for name in files:
            if name.endswith('.txt'):
                #print(name)
                xmlpath = os.path.join(path, name)
                test_sentences, pmids = load_xml(xmlpath, entity_a, entity_b)
                candidate_instances = build_instances_testing(test_sentences, dep_dictionary, dep_path_word_dictionary, dep_element_dictionary, between_word_dictionary,
                            distant_interactions,reverse_distant_interactions, key_order, entity_1_list =  None, entity_2_list = None)

                for ci in candidate_instances:
                    total_instances.append(ci)
                    total_features.append(ci.features)
                    total_labels.append(ci.label)

    return total_instances,total_features,total_labels