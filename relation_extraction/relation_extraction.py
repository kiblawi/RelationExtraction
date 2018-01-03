import sys
import operator

import load_data


import random
import itertools

import structures
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics


def predict_sentences(model_file,sentence_file,entity_1,entity_1_file,entity_1_col,
                       entity_2,entity_2_file,entity_2_col,symmetric):

    if entity_1_file.upper() != "NONE":
        entity_1_ids = load_data.load_id_list(entity_1_file,entity_1_col)
    else:
        entity_1_ids = None
    if entity_2_file.upper() != "NONE":
        entity_2_ids = load_data.load_id_list(entity_2_file,entity_2_col)
    else:
        entity_2_ids = None

    predict_candidate_sentences = load_data.load_xml(sentence_file, entity_1, entity_2)


    model, dep_dictionary, dep_word_dictionary,  between_word_dictionary = joblib.load(model_file)
    predict_instances = load_data.build_instances_predict(predict_candidate_sentences,  dep_dictionary, dep_word_dictionary,
                                                          between_word_dictionary, entity_1_ids, entity_2_ids, symmetric)




    X = []
    instance_sentences = set()
    for p in predict_instances:
        X.append(p.features)
        instance_sentences.add(p.get_sentence())

    X_predict = np.array(X)

    predicted_labels = model.predict(X_predict)
    print('Number of Sentences')
    print(len(instance_sentences))
    print('Number of Instances')
    print(len(predict_instances))
    return predict_instances, predicted_labels


def distant_train(model_out,sentence_file,distant_file,distant_e1_col,distant_e2_col,distant_rel_col,entity_1,entity_1_file,entity_1_col,
                      entity_2,entity_2_file,entity_2_col,symmetric):

    if entity_1_file.upper() != "NONE":
        entity_1_ids = load_data.load_id_list(entity_1_file,entity_1_col)
    else:
        entity_1_ids = None
    if entity_2_file.upper() != "NONE":
        entity_2_ids = load_data.load_id_list(entity_2_file,entity_2_col)
    else:
        entity_2_ids = None

    distant_interactions,reverse_distant_interactions = load_data.load_distant_kb(distant_file,distant_e1_col,distant_e2_col,distant_rel_col)
    print(len(distant_interactions))
    print(len(reverse_distant_interactions))
    training_sentences = load_data.load_xml(sentence_file,entity_1,entity_2)

    training_instances, dep_dictionary, dep_word_dictionary,  between_word_dictionary = load_data.build_instances_training(
        training_sentences, distant_interactions,reverse_distant_interactions, entity_1_ids, entity_2_ids, symmetric )


    X = []
    y = []
    instance_sentences = set()
    for t in training_instances:
        instance_sentences.add(t.get_sentence())
        X.append(t.features)
        y.append(t.label)

    X_train = np.array(X)
    y_train = np.ravel(y)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    print('Number of Sentences')
    print(len(instance_sentences))
    print('Number of Instances')
    print(len(training_instances))
    print('Number of Positive Instances')
    print(y.count(1))
    print(model.get_params)
    print('Number of dependency paths ')
    print(len(dep_dictionary))
    print('Number of dependency words')
    print(len(dep_word_dictionary))
    print('Number of between words')
    print(len(between_word_dictionary))
    joblib.dump((model,dep_dictionary,dep_word_dictionary,between_word_dictionary),model_out)

    print("trained model")

    '''
    sorted_dep_dictionary = sorted(dep_dictionary.items(), key=operator.itemgetter(1))
    dep_dictionary_keys = []
    for s in sorted_dep_dictionary:
        dep_dictionary_keys.append('Dep_path: ' + s[0])
    sorted_word_dep_dictionary = sorted(dep_word_dictionary.items(), key=operator.itemgetter(1))
    word_dep_keys = []
    for s in sorted_word_dep_dictionary:
        word_dep_keys.append('Word in Dependency Path: ' + s[0])
    sorted_between_word_dictionary = sorted(between_word_dictionary.items(), key=operator.itemgetter(1))
    between_word_keys = []
    for s in sorted_between_word_dictionary:
        between_word_keys.append('Word Between Entities: ' + s[0])
    
    feature_values = dep_dictionary_keys + word_dep_keys + between_word_keys
    print(feature_values)
    print(len(feature_values))
    print(model.coef_.size)
    feature_dict = {}
    for i in range(model.coef_.size):
        feature_dict[feature_values[i]] = abs(model.coef_.item(i))

    sorted_feature_dict = sorted(feature_dict.items(), key=operator.itemgetter(1))
    for s in sorted_feature_dict:
        print(s[0] + '\t' + str(s[1]))

    '''

def main():
    mode = sys.argv[1]
    if mode.upper() == "DISTANT_TRAIN":
        model_out = sys.argv[2]
        sentence_file = sys.argv[3]
        distant_file = sys.argv[4]
        distant_e1_col = int(sys.argv[5])
        distant_e2_col = int(sys.argv[6])
        distant_rel_col = int(sys.argv[7])
        entity_1 = sys.argv[8].upper()
        entity_1_file = sys.argv[9]
        entity_1_col = int(sys.argv[10])
        entity_2 = sys.argv[11].upper()
        entity_2_file = sys.argv[12]
        entity_2_col = int(sys.argv[13])
        symmetric = sys.argv[14].upper() in ['TRUE','Y','YES']

        distant_train(model_out,sentence_file,distant_file,distant_e1_col,distant_e2_col,distant_rel_col,entity_1,entity_1_file,entity_1_col,
                      entity_2,entity_2_file,entity_2_col,symmetric)

    elif mode.upper() == "TEST":
        model_file = sys.argv[2]
        sentence_file = sys.argv[3]
        entity_1 = sys.argv[4].upper()
        entity_1_file = sys.argv[5]
        entity_1_col = int(sys.argv[6])
        entity_2 = sys.argv[7].upper()
        entity_2_file = sys.argv[8]
        entity_2_col = int(sys.argv[9])
        symmetric = sys.argv[10].upper() in ['TRUE','Y','YES']

        print('testing function not developed yet')

    elif mode.upper() == "PREDICT":
        model_file = sys.argv[2]
        sentence_file = sys.argv[3]
        entity_1 = sys.argv[4].upper()
        entity_1_file = sys.argv[5]
        entity_1_col = int(sys.argv[6])
        entity_2 = sys.argv[7].upper()
        entity_2_file = sys.argv[8]
        entity_2_col = int(sys.argv[9])
        symmetric = sys.argv[10].upper() in ['TRUE','Y','YES']

        predicted_instances, predicted_labels = predict_sentences(model_file,sentence_file,entity_1,entity_1_file,entity_1_col,
                       entity_2,entity_2_file,entity_2_col,symmetric)

        '''
        #trying to assemble list of relations
        outfile = open('/Users/kiblawi/Workspace/Data/predicted_interactions.txt','w')
        outfile2 = open('/Users/kiblawi/Workspace/Data/predicted_interactions2.txt','w')
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == 1:
                pi = predicted_instances[i]
                sp = []
                ep = []
                start_point = pi.get_sentence().get_token(pi.start)
                end_point = pi.get_sentence().get_token(pi.end)
                outfile2.write(start_point.get_normalized_ner() + '\t' + end_point.get_normalized_ner() + '\n')
                for e in pi.get_sentence().entities:
                    for l in pi.get_sentence().entities[e]:
                        if pi.start in l:
                            sp = l
                        elif pi.end in l:
                            ep = l
                outfile.write(' '.join(pi.get_sentence().get_token(a).get_word() for a in sp).encode('utf-8') + '\t' + ' '.join(
                    pi.get_sentence().get_token(b).get_word() for b in ep).encode('utf-8') + '\n')
        outfile.close()
        '''
        outfile = open('/Users/kiblawi/Workspace/Data/predicted_sentences.txt','w')
        for i in range(len(predicted_labels)):
            pi = predicted_instances[i]
            sp = []
            ep = []
            for e in pi.get_sentence().entities:
                for l in pi.get_sentence().entities[e]:
                    if pi.start in l:
                        sp = l
                    elif pi.end in l:
                        ep = l
            outfile.write('Instance: ' + str(i) + '\n')
            outfile.write('Label: ' + str(predicted_labels[i]) + '\n')
            outfile.write(
                ' '.join('Human_gene:' + pi.get_sentence().get_token(a).get_word() for a in sp).encode('utf-8') + '\t' + 'Viral_gene:' + ' '.join(
                    pi.get_sentence().get_token(b).get_word() for b in ep).encode('utf-8') + '\n')
            outfile.write('Human_gene_index: ' + str(pi.start) + '\t' + 'Viral_gene_index: ' + str(pi.end) + '\n')
            outfile.write(pi.get_sentence().get_sentence_string().encode('utf-8') + '\n')
            outfile.write('Accuracy: \n\n')
        outfile.close()


    else:
        print("usage error")


if __name__=="__main__":
    main()
