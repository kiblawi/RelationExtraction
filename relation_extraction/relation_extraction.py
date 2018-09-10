import sys
import os
import load_data
import cross_validation as cv
import numpy as np
import itertools
import collections
import shutil
import pickle
import time

from machine_learning_models import tf_feed_forward as nn
from machine_learning_models import tf_lstm as lstm

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.externals import joblib
from sklearn import metrics


def write_output(filename, predicts, labels, instances, key_order):
    """
    Prints text file of output
    :param filename: output file
    :param predicts: probability values of instances
    :param labels:  labels of instances
    :param instances: instances so you can get instance data
    :param key_order: key order of labels
    :return:
    """
    for k in range(len(key_order)):
        key = key_order[k]
        labels_list = []
        file = open(filename + '_' + key, 'w')
        file.write('PMID\tE1\tE2\tClASS_LABEL\tPROBABILITY\n')
        for q in range(predicts[:, k].size):
            instance_label = labels[q, k]
            instance_start = instances[q].sentence.get_token(instances[q].start[0]).normalized_ner
            instance_end = instances[q].sentence.get_token(instances[q].end[0]).normalized_ner
            labels_list.append(instance_label)
            file.write(
                str(instances[q].sentence.pmid) + '\t' + str(instance_start) + '\t' + str(instance_end) + '\t' + str(
                    instance_label) + '\t' + str(predicts[q, k]) + '\n')

        file.close()

    return


def predict(model_file, abstract_folder, entity_a, entity_b):
    """
    WORK IN PROGRESS!
    :param model_file:
    :param abstract_folder:
    :param entity_a:
    :param entity_b:
    :return:
    """

    predict_pmids, predict_sentences = load_data.load_abstracts_from_directory(abstract_folder,entity_a,entity_b)

    dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary, key_order = pickle.load(open(model_file + 'a.pickle','rb'))


    predict_instances = load_data.build_instances_predict(predict_sentences,dep_dictionary,
                                                          dep_word_dictionary, dep_element_dictionary,
                                                          between_word_dictionary,key_order)

    predict_features = []
    outfile = open(model_file+'hsv1-instances.txt','w')
    for predict_index in range(len(predict_instances)):
        pi = predict_instances[predict_index]
        if predict_index%1 == 0:
            sentence_format = pi.sentence.sentence_words
            sentence_format[pi.entity_pair[0]] = '***' + sentence_format[pi.entity_pair[0]] + '***'
            sentence_format[pi.entity_pair[1]] = '***'  + sentence_format[pi.entity_pair[1]] + '***'
            outfile.write(pi.sentence.pmid + '\t' + pi.sentence.start_entity_text + '\t' + pi.sentence.end_entity_text + '\t' + ' '.join(sentence_format))
        predict_features.append(pi.features)

    outfile.close()
    predicted_prob = nn.neural_network_predict(predict_features, model_file + '/')

    return predict_instances,predicted_prob,key_order


def test_lstm(model_out, abstract_folder, directional_distant_directory, symmetric_distant_directory,
              distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b):
    """
    Takes a directory of abstracts, labels the instances based on distant supervision and runs it through
    trained LSTM model and outputs spreadsheet of probabilities of instances.
    :param model_out: directory of trained model
    :param abstract_folder: folder of abstracts to test
    :param directional_distant_directory: distant dataset for directional relations
    :param symmetric_distant_directory: distant dataset for symmetric relations
    :param distant_entity_a_col: column in distant dataset for entity a
    :param distant_entity_b_col: column in distant datset for entity b
    :param distant_rel_col: column in datset for relation
    :param entity_a: entity a in format id_type
    :param entity_b: entity b in format id_type
    :return: True if successful
    """

    # loads distant datsets
    distant_interactions, reverse_distant_interactions = load_data.load_distant_directories(
        directional_distant_directory,
        symmetric_distant_directory,
        distant_entity_a_col,
        distant_entity_b_col, distant_rel_col)

    # sorts keys
    key_order = sorted(distant_interactions)

    # check if there is already dictionaries
    if os.path.isfile(model_out + 'a.pickle'):
        dep_path_list_dictionary, dep_word_dictionary, key_order = pickle.load(open(model_out + 'a.pickle', 'rb'))
    # load in sentences and try to get dictionaries built
    else:
        print('error')

    # builds test_instances,features, and labels for LSTM model
    test_instances, test_features, test_labels = load_data.build_LSTM_test_instances_from_directory(abstract_folder, entity_a, entity_b,
                                                                     dep_path_list_dictionary, dep_word_dictionary,
                                                                     distant_interactions,
                                                                     reverse_distant_interactions, key_order)

    print('test_instances')
    print(len(test_instances))
    #print(test_features)
    # create np arrays
    test_labels = np.array(test_labels, dtype='float32')
    test_features = np.array(test_features, dtype='float32')

    # tests instances for LSTM model
    instance_predicts, predict_labels = lstm.lstm_test(test_features, test_labels, model_out + '/')
    print(instance_predicts.shape)
    np.testing.assert_array_equal(test_labels, predict_labels)

    # writes csv to output for test instances
    write_output(model_out + '_test_predictions', instance_predicts, predict_labels, test_instances, key_order)

    return True

def test_feed_forward(model_out, abstract_folder, directional_distant_directory, symmetric_distant_directory,
                      distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b):
    """
    Takes a directory of abstracts, labels the instances based on distant supervision and runs it through
    trained feed forward neural network model and outputs spreadsheet of probabilities of instances.
    :param model_out: directory of trained model
    :param abstract_folder: folder of abstracts to test
    :param directional_distant_directory: distant dataset for directional relations
    :param symmetric_distant_directory: distant dataset for symmetric relations
    :param distant_entity_a_col: column in distant dataset for entity a
    :param distant_entity_b_col: column in distant datset for entity b
    :param distant_rel_col: column in datset for relation
    :param entity_a: entity a in format id_type
    :param entity_b: entity b in format id_type
    :return: true if successful
    """

    # loads distant directories for labelling instances
    distant_interactions, reverse_distant_interactions = load_data.load_distant_directories(
        directional_distant_directory,
        symmetric_distant_directory,
        distant_entity_a_col,
        distant_entity_b_col, distant_rel_col)

    # get order of elements for consistency
    key_order = sorted(distant_interactions)

    # check if there is already dictionaries
    if os.path.isfile(model_out + 'a.pickle'):
        dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary, key_order = pickle.load(
            open(model_out + 'a.pickle', 'rb'))
    # load in sentences and try to get dictionaries built
    else:
        print('error')

    # total number of features
    num_features = len(dep_dictionary) + len(dep_word_dictionary) + len(dep_element_dictionary) + len(between_word_dictionary)
    print(num_features)

    # builds test instances and features for feed forward neural network
    test_instances,test_features,test_labels  = load_data.build_test_instances_from_directory(abstract_folder, entity_a, entity_b,
                                                                       dep_dictionary, dep_word_dictionary,
                                                                       dep_element_dictionary, between_word_dictionary,
                                                                       distant_interactions,
                                                                       reverse_distant_interactions, key_order)
    print('test_instances')
    print(len(test_instances))

    test_labels = np.array(test_labels,dtype='float32')
    test_features = np.array(test_features,dtype='float32')

    # creates instances for neural network
    instance_predicts, predict_labels = nn.neural_network_test(test_features, test_labels, model_out + '/')
    print(instance_predicts.shape)
    np.testing.assert_array_equal(test_labels,predict_labels)

    write_output(model_out + '_test_predictions', instance_predicts, predict_labels, test_instances, key_order)

    return True

def distant_train_lstm(model_out, abstract_folder, directional_distant_directory, symmetric_distant_directory,
                       distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b, testing_abstracts):

    """
    Distantly trains LSTM model with abstract folder full of .xml files
    :param model_out: path to save trained model
    :param abstract_folder: folder of abstracts in .xml format
    :param directional_distant_directory: directory of distant datasets with directional relation
    :param symmetric_distant_directory:  directory of distant datasets with symmetric relation
    :param distant_entity_a_col: column in distant dataset for entity a
    :param distant_entity_b_col: column in distant dataset for entity b
    :param distant_rel_col: column in datset for relation
    :param entity_a: entity a in format id_type
    :param entity_b: entity b in format id_type
    :param testing_abstracts: directory of test abstracts if you want test f1 score to appear during training
    :return: path of trained model
    """

    # load distant datsets for labelling
    distant_interactions, reverse_distant_interactions = load_data.load_distant_directories(directional_distant_directory,
                                                                                            symmetric_distant_directory,
                                                                                            distant_entity_a_col,
                                                                                            distant_entity_b_col,distant_rel_col)
    # sort key order
    key_order = sorted(distant_interactions)

    # check if there is already dictionaries
    if os.path.isfile(model_out + 'a.pickle'):
        dep_type_list_dictionary, dep_word_dictionary, key_order = pickle.load(open(model_out + 'a.pickle', 'rb'))
        word2vec_embeddings = None
        if os.path.isfile(os.path.dirname(os.path.realpath(__file__)) +'/machine_learning_models/PubMed-w2v.bin'):
            print('embeddings exist')
            word2vec_words, word2vec_vectors, dep_word_dictionary = lstm.load_bin_vec(os.path.dirname(os.path.realpath(__file__)) +'/machine_learning_models/PubMed-w2v.bin')
            word2vec_embeddings = np.array(word2vec_vectors)
            print('finished fetching embeddings')
    # load in sentences and try to get dictionaries built
    else:
        dep_type_list_dictionary, dep_word_dictionary, word2vec_embeddings = load_data.build_dictionaries_from_directory(abstract_folder, entity_a, entity_b,LSTM=True)

        pickle.dump([dep_type_list_dictionary, dep_word_dictionary, key_order], open(model_out + 'a.pickle', 'wb'))


    # builds or loads tfrecord datsets from .xml files
    total_dataset_files = []
    if os.path.isdir(abstract_folder):
        for path, subdirs, files in os.walk(abstract_folder):
            for name in files:
                if name.endswith('.tfrecord'):
                    total_dataset_files.append(abstract_folder + '/' + name)

        if len(total_dataset_files) == 0:
            total_dataset_files = load_data.build_LSTM_instances_from_directory(abstract_folder, entity_a, entity_b, dep_type_list_dictionary, dep_word_dictionary,
                                   distant_interactions, reverse_distant_interactions, key_order)

    # builds or loads test tfrecord files if they exist
    total_test_files = None
    if os.path.isdir(testing_abstracts):
        total_test_files = []
        for path, subdirs, files in os.walk(testing_abstracts):
            for name in files:
                if name.endswith('.tfrecord'):
                    total_test_files.append(testing_abstracts + '/' + name)
        if len(total_test_files) == 0:
            total_test_files = load_data.build_LSTM_instances_from_directory(testing_abstracts, entity_a, entity_b,
                                                                           dep_type_list_dictionary, dep_word_dictionary,
                                                                           distant_interactions,
                                                                           reverse_distant_interactions, key_order)

    num_dep_types = len(dep_type_list_dictionary)
    num_path_words = len(dep_word_dictionary)

    # trains LSTM model
    trained_model_path = lstm.lstm_train(total_dataset_files, num_dep_types,num_path_words, model_out + '/', key_order,total_test_files,word2vec_embeddings)


    return trained_model_path


def distant_train_feed_forward(model_out, abstract_folder, directional_distant_directory, symmetric_distant_directory,
                               distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b, testing_abstracts):
    """
    Distantly trains Feed forward model with abstract folder full of .xml files
    :param model_out: path to save trained model
    :param abstract_folder: folder of abstracts in .xml format
    :param directional_distant_directory: directory of distant datasets with directional relation
    :param symmetric_distant_directory:  directory of distant datasets with symmetric relation
    :param distant_entity_a_col: column in distant dataset for entity a
    :param distant_entity_b_col: column in distant dataset for entity b
    :param distant_rel_col: column in datset for relation
    :param entity_a: entity a in format id_type
    :param entity_b: entity b in format id_type
    :param testing_abstracts: directory of test abstracts if you want test f1 score to appear during training (optional)
    :return: path of trained model
    """

    # get distant_relations from external knowledge base file
    print(directional_distant_directory)
    print(symmetric_distant_directory)
    print(distant_entity_a_col)
    print(distant_entity_b_col)
    print(distant_rel_col)
    distant_interactions, reverse_distant_interactions = load_data.load_distant_directories(directional_distant_directory,
                                                                                            symmetric_distant_directory,
                                                                                            distant_entity_a_col,
                                                                                            distant_entity_b_col,distant_rel_col)

    # sort key orders
    key_order = sorted(distant_interactions)

    # check if there is already dictionaries
    if os.path.isfile(model_out + 'a.pickle'):
        dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary, key_order = pickle.load(open(model_out + 'a.pickle', 'rb'))
    # load in sentences and try to get dictionaries built
    else:
        print('building dictionaries')
        dep_dictionary, \
        dep_word_dictionary, \
        dep_element_dictionary, \
        between_word_dictionary = load_data.build_dictionaries_from_directory(abstract_folder, entity_a, entity_b)

        pickle.dump([dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary, key_order],
                    open(model_out + 'a.pickle', 'wb'))

    # get number of features
    num_features = len(dep_dictionary) + len(dep_word_dictionary) + len(dep_element_dictionary)+ len(between_word_dictionary)
    print(num_features)

    # builds or loads tfrecord files for training
    total_dataset_files = []
    if os.path.isdir(abstract_folder):
        for path, subdirs, files in os.walk(abstract_folder):
            for name in files:
                if name.endswith('.tfrecord'):
                    total_dataset_files.append(abstract_folder + '/' + name)
        print(total_dataset_files)
        if len(total_dataset_files) == 0:
            total_dataset_files = load_data.build_instances_from_directory(abstract_folder, entity_a, entity_b, dep_dictionary, dep_word_dictionary,
                                   dep_element_dictionary, between_word_dictionary,
                                   distant_interactions, reverse_distant_interactions, key_order)

    # sets hidden array for hidden layers
    hidden_array = []

    # builds or loads test tfrecord files if they exist
    total_test_files = None
    if os.path.isdir(testing_abstracts):
        total_test_files = []
        for path, subdirs, files in os.walk(testing_abstracts):
            for name in files:
                if name.endswith('.tfrecord'):
                    total_test_files.append(testing_abstracts + '/' + name)
        if len(total_test_files) == 0:
            total_test_files = load_data.build_instances_from_directory(testing_abstracts, entity_a, entity_b,
                                                                           dep_dictionary, dep_word_dictionary,
                                                                           dep_element_dictionary,
                                                                           between_word_dictionary,
                                                                           distant_interactions,
                                                                           reverse_distant_interactions, key_order)

    # trains feef forward neural network model
    trained_model_path = nn.neural_network_train(total_dataset_files, hidden_array, model_out + '/', num_features, key_order, total_test_files)


    return trained_model_path


def main():
    """
    Main method, mode determines whether program runs training, testing, or prediction
    :return:
    """

    mode = sys.argv[1]  # what option
    if "TRAIN_FEED_FORWARD" in mode.upper(): # train feed forward network
        model_out = sys.argv[2]  # location of where model should be saved after training
        abstract_folder = sys.argv[3]  # xml file of sentences from Stanford Parser
        directional_distant_directory = sys.argv[4]  # distant supervision knowledge base to use
        symmetric_distant_directory = sys.argv[5]
        distant_entity_a_col = int(sys.argv[6])  # entity 1 column
        distant_entity_b_col = int(sys.argv[7])  # entity 2 column
        print(distant_entity_b_col)
        distant_rel_col = int(sys.argv[8])  # relation column
        print(distant_rel_col)
        entity_a = sys.argv[9].upper()  # entity_a
        print(entity_a)
        entity_b = sys.argv[10].upper()  # entity_b
        testing_abstracts =sys.argv[11] # optional put None if you don't want to get f1 score of test set

        # distanty train feed forward neural network
        trained_model_path = distant_train_feed_forward(model_out, abstract_folder, directional_distant_directory,
                                                        symmetric_distant_directory,
                                                        distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a,
                                                        entity_b, testing_abstracts)



        print(trained_model_path)
        
    elif "TRAIN_LSTM" in mode.upper(): # train LSTM network
        model_out = sys.argv[2]  # location of where model should be saved after training
        abstract_folder = sys.argv[3]  # xml file of sentences from Stanford Parser
        directional_distant_directory = sys.argv[4]  # distant supervision knowledge base to use
        symmetric_distant_directory = sys.argv[5]
        distant_entity_a_col = int(sys.argv[6])  # entity 1 column
        distant_entity_b_col = int(sys.argv[7])  # entity 2 column
        print(distant_entity_b_col)
        distant_rel_col = int(sys.argv[8])  # relation column
        print(distant_rel_col)
        entity_a = sys.argv[9].upper()  # entity_a
        print(entity_a)
        entity_b = sys.argv[10].upper()  # entity_b
        testing_abstracts =sys.argv[11] # optional put None if you don't want to get f1 score of test set

        #distantly train LSTM network
        trained_model_path = distant_train_lstm(model_out, abstract_folder, directional_distant_directory,
                                                symmetric_distant_directory,
                                                distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a,
                                                entity_b, testing_abstracts)



        print(trained_model_path)

    elif "TEST" in mode.upper(): # tests folder of .xml files
        model_out = sys.argv[2]  # location of where model should be saved after training
        abstract_folder = sys.argv[3]  # xml file of sentences from Stanford Parser
        directional_distant_directory = sys.argv[4]  # distant supervision knowledge base to use
        symmetric_distant_directory = sys.argv[5]
        distant_entity_a_col = int(sys.argv[6])  # entity 1 column
        distant_entity_b_col = int(sys.argv[7])  # entity 2 column
        print(distant_entity_b_col)
        distant_rel_col = int(sys.argv[8])  # relation column
        print(distant_rel_col)
        entity_a = sys.argv[9].upper()  # entity_a
        print(entity_a)
        entity_b = sys.argv[10].upper()  # entity_b
        LSTM = sys.argv[11] # boolean to determine if you want to test LSTM or feed forward network
        LSTM = LSTM == 'True'


        if LSTM is False:
            trained_model_path = test_feed_forward(model_out, abstract_folder, directional_distant_directory,
                                                   symmetric_distant_directory,
                                                   distant_entity_a_col, distant_entity_b_col, distant_rel_col,
                                                   entity_a,
                                                   entity_b)
        else:
            trained_model_path = test_lstm(model_out, abstract_folder, directional_distant_directory,
                                           symmetric_distant_directory,
                                           distant_entity_a_col, distant_entity_b_col, distant_rel_col,
                                           entity_a,
                                           entity_b)
        print('finished testfile')

    elif "PREDICT" in mode.upper(): # WORKING ON IT!
        model_file = sys.argv[2]
        sentence_file = sys.argv[3]
        entity_a = sys.argv[4].upper()
        entity_b = sys.argv[5].upper()
        out_pairs_file = sys.argv[6]

        prediction_instances, predict_probs,key_order = predict(model_file, sentence_file, entity_a, entity_b)

        for key_index in range(len(key_order)):
            key = key_order[key_index]
            outfile = open(out_pairs_file + '_' + key, 'w')
            outfile.write('PMID\tENTITY_1\tENTITY_2\tCLASS_LABEL\tPROBABILITY\tSENTENCE\n')
            for i in range(len(prediction_instances)):
                pi = prediction_instances[i]
                outfile.write(str(pi.sentence.pmid) + '\t'
                              + str(pi.sentence.start_entity_id) + '\t'
                              + str(pi.sentence.end_entity_id) + '\t'
                              + str(pi.label[key_index]) + '\t'
                              + str(predict_probs[i,key_index])+'\t'
                              + str(pi.sentence.start_entity_text) + '\t'
                              + ' '.join(pi.sentence.sentence_words) + '\n')

            outfile.close()

    else:
        print("usage error")


if __name__ == "__main__":
    main()


