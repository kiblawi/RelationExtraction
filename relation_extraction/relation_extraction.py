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



from machine_learning_models import tf_sess_neural_network as snn

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.externals import joblib
from sklearn import metrics


def predict_sentences(model_file, abstract_folder, entity_a, entity_b):

    predict_pmids, predict_sentences = load_data.load_abstracts_from_directory(abstract_folder,entity_a,entity_b)

    dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary, key_order = pickle.load(open(model_file + 'a.pickle','rb'))


    predict_instances = load_data.build_instances_predict(predict_sentences,dep_dictionary,
                                                          dep_word_dictionary, dep_element_dictionary,
                                                          between_word_dictionary,key_order)




    predict_features = []
    outfile = open('/Users/kiblawi/Workspace/Data/predicted_results/hsv1-instances.txt','w')
    for predict_index in range(len(predict_instances)):
        if predict_index%2 == 0:
            pi = predict_instances[predict_index]
            sentence_format = pi.sentence.sentence_words
            sentence_format[pi.entity_pair[0]] = '***' + sentence_format[pi.entity_pair[0]] + '***'
            sentence_format[pi.entity_pair[1]] = '***'  + sentence_format[pi.entity_pair[1]] + '***'
            outfile.write(pi.sentence.pmid + '\t' + pi.sentence.start_entity_text + '\t' + pi.sentence.end_entity_text + '\t' + ' '.join(sentence_format))
        predict_features.append(pi.features)

    outfile.close()
    predicted_prob = snn.neural_network_predict(predict_features,model_file + '/')

    return predict_instances,predicted_prob,key_order


def parallel_train(model_out, abstract_folder, directional_distant_directory, symmetric_distant_directory,
                   distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b, batch_id):

    #get distant_relations from external knowledge base file
    distant_interactions, reverse_distant_interactions = load_data.load_distant_directories(directional_distant_directory,
                                                                                            symmetric_distant_directory,
                                                                                            distant_entity_a_col,
                                                                                            distant_entity_b_col,
                                                                                            distant_rel_col)

    key_order = sorted(distant_interactions)
    #get pmids,sentences
    training_pmids,training_sentences = load_data.load_abstracts_from_directory(abstract_folder,entity_a,entity_b)

    #hidden layer structure
    hidden_array = [256]

    #k-cross val
    instance_predicts,single_instances = cv.parallel_k_fold_cross_validation(batch_id, 10, training_pmids,
                                                                             training_sentences,
                                                                             distant_interactions,
                                                                             reverse_distant_interactions,
                                                                             hidden_array,
                                                                             key_order)

    cv.write_cv_output(model_out + '_' +str(batch_id)+'_predictions',instance_predicts,single_instances,key_order)


    return batch_id

def distant_test_large_data(model_out, abstract_folder, directional_distant_directory, symmetric_distant_directory,
                  distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b):
    print(directional_distant_directory)
    print(symmetric_distant_directory)
    print(distant_entity_a_col)
    print(distant_entity_b_col)
    print(distant_rel_col)
    distant_interactions, reverse_distant_interactions = load_data.load_distant_directories(
        directional_distant_directory,
        symmetric_distant_directory,
        distant_entity_a_col,
        distant_entity_b_col, distant_rel_col)
    # print(distant_interactions)

    key_order = sorted(distant_interactions)

    # check if there is already dictionaries
    if os.path.isfile(model_out + 'a.pickle'):
        dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary, key_order = pickle.load(
            open(model_out + 'a.pickle', 'rb'))
    # load in sentences and try to get dictionaries built
    else:
        dep_dictionary, \
        dep_word_dictionary, \
        dep_element_dictionary, \
        between_word_dictionary = load_data.build_dictionaries_from_directory(abstract_folder, entity_a, entity_b)

        pickle.dump([dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary, key_order],
                    open(model_out + 'a.pickle', 'wb'))

    num_features = len(dep_dictionary) + len(dep_word_dictionary) + len(dep_element_dictionary) + len(
        between_word_dictionary)
    print(num_features)

    total_dataset_files = []
    for path, subdirs, files in os.walk(abstract_folder):
        for name in files:
            if name.endswith('.tfrecord'):
                print(name)
                total_dataset_files.append(abstract_folder + '/' + name)
    print(total_dataset_files)
    if len(total_dataset_files) == 0:
        total_dataset_files = load_data.build_instances_from_directory(abstract_folder, entity_a, entity_b,
                                                                       dep_dictionary, dep_word_dictionary,
                                                                       dep_element_dictionary, between_word_dictionary,
                                                                       distant_interactions,
                                                                       reverse_distant_interactions, key_order)

    trained_model_path = snn.neural_network_test_tfrecord(total_dataset_files, model_out + '/')


def distant_train_large_data(model_out, abstract_folder, directional_distant_directory, symmetric_distant_directory,
                  distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b):

    #get distant_relations from external knowledge base file
    print(directional_distant_directory)
    print(symmetric_distant_directory)
    print(distant_entity_a_col)
    print(distant_entity_b_col)
    print(distant_rel_col)
    distant_interactions, reverse_distant_interactions = load_data.load_distant_directories(directional_distant_directory,
                                                                                            symmetric_distant_directory,
                                                                                            distant_entity_a_col,
                                                                                            distant_entity_b_col,distant_rel_col)
    #print(distant_interactions)

    key_order = sorted(distant_interactions)

    #check if there is already dictionaries
    if os.path.isfile(model_out + 'a.pickle'):
        dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary, key_order = pickle.load(open(model_out + 'a.pickle', 'rb'))
    #load in sentences and try to get dictionaries built
    else:
        dep_dictionary, \
        dep_word_dictionary, \
        dep_element_dictionary, \
        between_word_dictionary = load_data.build_dictionaries_from_directory(abstract_folder, entity_a, entity_b)

        pickle.dump([dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary, key_order],
                    open(model_out + 'a.pickle', 'wb'))

    num_features = len(dep_dictionary) + len(dep_word_dictionary) + len(dep_element_dictionary)+ len(between_word_dictionary)
    print(num_features)

    total_dataset_files = []
    for path, subdirs, files in os.walk(abstract_folder):
        for name in files:
            if name.endswith('.tfrecord'):
                print(name)
                total_dataset_files.append(abstract_folder + '/' + name)
    print(total_dataset_files)
    if len(total_dataset_files) == 0:
        total_dataset_files = load_data.build_instances_from_directory(abstract_folder, entity_a, entity_b, dep_dictionary, dep_word_dictionary,
                                   dep_element_dictionary, between_word_dictionary,
                                   distant_interactions, reverse_distant_interactions, key_order)

    hidden_array = [256]
    print(total_dataset_files)
    trained_model_path = snn.neural_network_train_tfrecord(total_dataset_files, hidden_array, model_out + '/', num_features, key_order)


    return trained_model_path


def distant_train(model_out, abstract_folder, directional_distant_directory, symmetric_distant_directory,
                  distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a, entity_b):

    #get distant_relations from external knowledge base file
    print(directional_distant_directory)
    print(symmetric_distant_directory)
    print(distant_entity_a_col)
    print(distant_entity_b_col)
    print(distant_rel_col)
    distant_interactions, reverse_distant_interactions = load_data.load_distant_directories(directional_distant_directory,
                                                                                            symmetric_distant_directory,
                                                                                            distant_entity_a_col,
                                                                                            distant_entity_b_col,distant_rel_col)
    #print(distant_interactions)

    key_order = sorted(distant_interactions)
    #get pmids,sentences,
    training_pmids,training_sentences = load_data.load_abstracts_from_directory(abstract_folder,entity_a,entity_b)


    #hidden layer structure
    hidden_array = [256]


    #training full model

    training_instances, \
    dep_dictionary, \
    dep_word_dictionary, \
    dep_element_dictionary, \
    between_word_dictionary = load_data.build_instances_training(training_sentences,
                                                   distant_interactions,
                                                   reverse_distant_interactions, key_order)


    X = []
    y = []
    instance_sentences = set()
    count = 0
    for t in training_instances:
        instance_sentences.add(t.sentence.get_sentence_string())
        X.append(t.features)
        if 1 in t.label:
            count +=1
        y.append(t.label)



    X_train = np.array(X)
    y_train = np.array(y)

    if os.path.exists(model_out):
        shutil.rmtree(model_out)


    trained_model_path = snn.neural_network_train(X_train,
                                              y_train,
                                              None,
                                              None,
                                              hidden_array,
                                              model_out + '/', key_order)


    print('Number of Sentences')
    print(len(instance_sentences))
    print('Number of Instances')
    print(len(training_instances))
    print('Number of dependency paths ')
    print(len(dep_dictionary))
    print('Number of dependency words')
    print(len(dep_word_dictionary))
    print('Number of between words')
    print(len(between_word_dictionary))
    print('Number of elements')
    print(len(dep_element_dictionary))
    print('length of feature space')
    print(len(dep_dictionary) + len(dep_word_dictionary) + len(dep_element_dictionary) + len(between_word_dictionary))
    pickle.dump([dep_dictionary, dep_word_dictionary, dep_element_dictionary, between_word_dictionary,key_order], open(model_out + 'a.pickle','wb'))
    print("trained model")


    return trained_model_path


def main():
    ''' Main method, mode determines whether program runs training, testing, or prediction'''
    mode = sys.argv[1]  # what option
    if "DISTANT_TRAIN" in mode.upper():
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

        #symmetric = sys.argv[10].upper() in ['TRUE', 'Y', 'YES']  # is the relation symmetrical (i.e. binds)

        if 'LARGE' in mode.upper():
            trained_model_path = distant_train_large_data(model_out, abstract_folder, directional_distant_directory,
                                               symmetric_distant_directory,
                                               distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a,
                                               entity_b)
        else:
            trained_model_path = distant_train(model_out, abstract_folder, directional_distant_directory,symmetric_distant_directory,
                      distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a,entity_b)


        print(trained_model_path)

    elif "PARALLEL_TRAIN" in mode.upper():
        model_out = sys.argv[2]  # location of where model should be saved after training
        abstract_folder = sys.argv[3]  # xml file of sentences from Stanford Parser
        directional_distant_directory = sys.argv[4]  # distant supervision knowledge base to use
        symmetric_distant_directory = sys.argv[5]
        distant_entity_a_col = int(sys.argv[6])  # entity 1 column
        distant_entity_b_col = int(sys.argv[7])  # entity 2 column
        distant_rel_col = int(sys.argv[8])  # relation column
        entity_a = sys.argv[9].upper()  # entity_a
        entity_b = sys.argv[10].upper()  # entity_b
        batch_id = int(sys.argv[11]) #batch to run

        #symmetric = sys.argv[10].upper() in ['TRUE', 'Y', 'YES']  # is the relation symmetrical (i.e. binds)

        trained_model_batch = parallel_train(model_out, abstract_folder, directional_distant_directory,symmetric_distant_directory,
                      distant_entity_a_col, distant_entity_b_col, distant_rel_col, entity_a,entity_b,batch_id)

        print('finished training: ' + str(trained_model_batch))

    elif "TEST" in mode.upper():
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

        # symmetric = sys.argv[10].upper() in ['TRUE', 'Y', 'YES']  # is the relation symmetrical (i.e. binds)


        trained_model_path = distant_test_large_data(model_out, abstract_folder, directional_distant_directory,
                                                          symmetric_distant_directory,
                                                          distant_entity_a_col, distant_entity_b_col, distant_rel_col,
                                                          entity_a,
                                                          entity_b)


    elif "PREDICT" in mode.upper():
        model_file = sys.argv[2]
        sentence_file = sys.argv[3]
        entity_a = sys.argv[4].upper()
        entity_b = sys.argv[5].upper()
        out_pairs_file = sys.argv[6]

        prediction_instances, predict_probs,key_order = predict_sentences(model_file, sentence_file, entity_a, entity_b)
        #print(total_group_instance_results)


        for key_index in range(len(key_order)):
            key = key_order[key_index]
            outfile = open(out_pairs_file + '_' + key, 'w')
            outfile.write('PMID\tENTITY_1\tENTITY_1_SPECIES\tENTITY_2\tENTITY_2_SPECIES\tPROBABILITY\tENTITY_1_NAME\tENTITY_2_NAME\n')
            for i in range(len(prediction_instances)):
                pi = prediction_instances[i]
                outfile.write(str(pi.sentence.pmid) + '\t'
                              + str(pi.sentence.start_entity_id) + '\t'
                              + str(pi.sentence.start_entity_species) + '\t'
                              + str(pi.sentence.end_entity_id) + '\t'
                              + str(pi.sentence.end_entity_species) + '\t'
                              + str(predict_probs[i,key_index])+'\t'
                              + str(pi.sentence.start_entity_text) + '\t'
                              + str(pi.sentence.end_entity_text) + '\n')

            outfile.close()

    else:
        print("usage error")


if __name__ == "__main__":
    main()
