import sys

import load_data

import random
import itertools

import structures
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics


def main():
    random.seed(102)
    hiv_genes = load_data.load_id_list(sys.argv[3], 2)
    hsv_genes = load_data.load_id_list(sys.argv[5],2)


    candidate_sentences = load_data.load_xml(sys.argv[1], 'HUMAN_GENE', 'VIRAL_GENE', None, hiv_genes)
    distant_interactions = load_data.load_distant_kb(sys.argv[2],4,0)
    hsv_sentences = load_data.load_xml(sys.argv[4],'HUMAN_GENE','VIRAL_GENE',None,hsv_genes)

    print(len(candidate_sentences))

    ten_fold_length = len(candidate_sentences)/10
    total_chunks = [candidate_sentences[i:i + ten_fold_length] for i in xrange(0, len(candidate_sentences), ten_fold_length)]
    total_chunks.pop(-1)

    total = 0
    feature_total = 0
    instance_total = 0
    positive_total = 0
    test_total = 0


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
        instance_total += len(training_instances)
        start_set = set()
        for t in training_instances:

            '''
            print("training instance")
            print(t.label)
            print(t.features)
            print(t.get_type_dependency_path())
            print(t.get_reverse_type_dependency_path())
            print(t.get_dep_word_path())
            print(t.get_between_words())
            print(t.sentence.get_entities())
            '''

            X.append(t.features)
            y.append(t.label)
            if t.label == 1:
                positive_instances += 1
            '''
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
            '''

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
        for t in hsv_instances:
            '''
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
            '''
            X.append(t.features)

        X_hsv = np.array(X)

        predicted_hsv = model.predict(X_hsv)

        outfile = open('../../Data/hsv_int.txt','w')
        tfile = open('../../Data/hsv_table.txt','w')
        sfile = open('../../Data/hsv_sent.txt','w')
        print(predicted_hsv)
        for p in range(len(predicted_hsv)):
            if predicted_hsv[p] == 1:
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
                sfile.write(' '.join(t.get_sentence().get_token(a).get_word() for a in sp) + '\t----\t' + ' '.join(
                    t.get_sentence().get_token(b).get_word() for b in ep) + '\n')
                print(' '.join(t.get_sentence().get_token(a).get_word() for a in sp) + '\t----\t' + ' '.join(t.get_sentence().get_token(b).get_word() for b in ep))
                outfile.write(' '.join(t.get_sentence().get_token(a).get_word() for a in sp) + '\t----\t' + ' '.join(t.get_sentence().get_token(b).get_word() for b in ep)+ '\n')
                tfile.write(' '.join(t.get_sentence().get_token(b).get_word() for b in ep)+'\tHSV\n')

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

if __name__=="__main__":
    main()
