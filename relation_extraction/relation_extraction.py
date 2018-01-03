import sys

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

    model, dep_dictionary, dep_word_dictionary, between_word_dictionary = joblib.load(model_file)
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


def distant_train(model_out,sentence_file,distant_file,distant_e1_col,distant_e2_col,entity_1,entity_1_file,entity_1_col,
                      entity_2,entity_2_file,entity_2_col,symmetric):

    if entity_1_file.upper() != "NONE":
        entity_1_ids = load_data.load_id_list(entity_1_file,entity_1_col)
    else:
        entity_1_ids = None
    if entity_2_file.upper() != "NONE":
        entity_2_ids = load_data.load_id_list(entity_2_file,entity_2_col)
    else:
        entity_2_ids = None

    distant_interactions = load_data.load_distant_kb(distant_file,distant_e1_col,distant_e2_col)

    training_sentences = load_data.load_xml(sentence_file,entity_1,entity_2)

    training_instances, dep_dictionary, dep_word_dictionary, between_word_dictionary = load_data.build_instances_training(
        training_sentences, distant_interactions, entity_1_ids, entity_2_ids, symmetric )


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

def main():
    mode = sys.argv[1]
    if mode.upper() == "DISTANT_TRAIN":
        model_out = sys.argv[2]
        sentence_file = sys.argv[3]
        distant_file = sys.argv[4]
        distant_e1_col = int(sys.argv[5])
        distant_e2_col = int(sys.argv[6])
        entity_1 = sys.argv[7].upper()
        entity_1_file = sys.argv[8]
        entity_1_col = int(sys.argv[9])
        entity_2 = sys.argv[10].upper()
        entity_2_file = sys.argv[11]
        entity_2_col = int(sys.argv[12])
        symmetric = sys.argv[13].upper() in ['TRUE','Y','YES']

        distant_train(model_out,sentence_file,distant_file,distant_e1_col,distant_e2_col,entity_1,entity_1_file,entity_1_col,
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
        '''

    else:
        print("usage error")



    '''



    ten_fold_length = len(candidate_sentences)/10
    total_chunks = [candidate_sentences[i:i + ten_fold_length] for i in xrange(0, len(candidate_sentences), ten_fold_length)]
    total_chunks.pop(-1)
    for i in range(1):
        chunks = total_chunks[:]
        test_sentences = chunks.pop(8)
        training_sentences = list(itertools.chain.from_iterable(chunks))
        training_instances, word_dictionary, dep_dictionary, between_word_dictionary = load_data.build_instances_training(training_sentences, distant_interactions, None, hiv_genes, True)
        test_instances = load_data.build_instances_testing(test_sentences, word_dictionary, dep_dictionary, between_word_dictionary, distant_interactions, None, hiv_genes, True)
        hsv_instances = load_data.build_instances_predict(hsv_sentences, word_dictionary, dep_dictionary, between_word_dictionary, None, hsv_genes, True)
        print('training linear regression')

        X = []
        y = []
        positive_instances = 0
        print(len(training_instances))
        print('hsvi: ' + str(len(hsv_instances)))
        instance_total += len(training_instances)
        start_set = set()
        for t in training_instances:

            
            print("training instance")
            print(t.label)
            print(t.features)
            print(t.get_type_dependency_path())
            print(t.get_reverse_type_dependency_path())
            print(t.get_dep_word_path())
            print(t.get_between_words())
            print(t.sentence.get_entities())
          

            X.append(t.features)
            y.append(t.label)
            if t.label == 1:
                positive_instances += 1
            
            else:
                print(t.get_sentence().get_sentence_string())
                startpoint = t.get_sentence().get_token(t.start)
                endpoint = t.get_sentence().get_token(t.end)
                print(t.get_sentence().entities)
                print(startpoint.get_word())
                start_set.add(startpoint.get_word())
                print(startpoint.get_normalized_ner())
                print(endpoint.get_word())
                print(endpoint.get_normalized_ner())
            

        print(positive_instances)
        positive_total += positive_instances

        X_train = np.array(X)
        y_train = np.ravel(y)

        model = LogisticRegression()
        model.fit(X_train,y_train)

        X = []
        y = []

        for t in test_instances:
            X.append(t.features)
            y.append(t.label)

        feature_total += len(t.features)
        test_total += len(test_instances)
        X_test = np.array(X)
        y_test = np.array(y)

        predicted = model.predict(X_test)



        average_precision = metrics.average_precision_score(y_test, predicted)
        f1 = metrics.f1_score(y_test,predicted)
        precision,recall,thresholds = metrics.precision_recall_curve(y_test,predicted)
        plt.interactive(False)
        fig = plt.figure()
        plt.step(recall, precision, color = 'b', alpha = 0.2, where = 'post')
        plt.fill_between(recall,precision, step = 'post',alpha = 0.2, color = 'b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
            average_precision))
        fig.savefig('../../Data/plot.png')
        print(f1)
        total += f1

        X = []
        print(len(test_instances))
        print(len(hsv_instances))
        count = 0
        for t in hsv_instances:
           
            print("hsv instance")
            print(t.get_sentence().get_sentence_string())
            startpoint = t.get_sentence().get_token(t.start)
            endpoint = t.get_sentence().get_token(t.end)
            print(t.get_sentence().entities)
            print(startpoint.get_word())
            start_set.add(startpoint.get_word())
            print(startpoint.get_normalized_ner())
            print(endpoint.get_word())
            print(endpoint.get_normalized_ner())
          
            X.append(t.features)

        X_hsv = np.array(X)

        predicted_hsv = model.predict(X_hsv)

        outfile = open('../../Data/hsv_int.txt','w')
        tfile = open('../../Data/hsv_table.txt','w')
        sfile = open('../../Data/hsv_sent.txt','w')
        print(predicted_hsv)
        for p in range(len(predicted_hsv)):
            if predicted_hsv[p] == 1:
                count+=1
                sp = []
                ep = []
                t = hsv_instances[p]
                print(t.get_sentence().sentence_id)
                sfile.write(t.get_sentence().get_sentence_string().encode('utf-8') + '\n')

                print(t.get_sentence().get_sentence_string())
                startpoint = t.get_sentence().get_token(t.start)
                endpoint = t.get_sentence().get_token(t.end)
                for e in t.get_sentence().entities:
                    for l in t.get_sentence().entities[e]:
                        if t.start in l:
                            sp = l
                        elif t.end in l:
                            ep = l
                sfile.write(' '.join(t.get_sentence().get_token(a).get_word() for a in sp).encode('utf-8') + '\t----\t' + ' '.join(
                    t.get_sentence().get_token(b).get_word() for b in ep).encode('utf-8') + '\n')
                print(' '.join(t.get_sentence().get_token(a).get_word() for a in sp) + '\t----\t' + ' '.join(t.get_sentence().get_token(b).get_word() for b in ep))
                outfile.write(' '.join(t.get_sentence().get_token(a).get_word() for a in sp).encode('utf-8') + '\t----\t' + ' '.join(t.get_sentence().get_token(b).get_word() for b in ep).encode('utf-8')+ '\n')
                tfile.write(' '.join(t.get_sentence().get_token(b).get_word() for b in ep).encode('utf-8')+'\tHSV\n')

                print('\n')

    outfile.close()
    tfile.close()
    sfile.close()
        
    print(total/10)
    print(feature_total/10)
    print(instance_total/10)
    print(positive_total/10)
    print(test_total/10)
    print(dep_dictionary)

    print(word_dictionary)
    print(between_word_dictionary)
    print(start_set)
    print(hiv_genes)
    print(count)
    '''

if __name__=="__main__":
    main()
