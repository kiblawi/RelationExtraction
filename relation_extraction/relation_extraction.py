import sys

import load_data

import random
import itertools

import structures
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics


def main():
    random.seed(102)
    candidate_sentences = load_data.load_xml(sys.argv[1], 'HUMAN_GENE', 'VIRAL_GENE')
    distant_interactions = load_data.load_distant_kb(sys.argv[2],4,0)
    hiv_genes = load_data.load_id_list(sys.argv[3],2)

    ten_fold_length = len(candidate_sentences)/10
    random.shuffle(candidate_sentences)
    total_chunks = [candidate_sentences[i:i + ten_fold_length] for i in xrange(0, len(candidate_sentences), ten_fold_length)]
    total_chunks.pop(-1)

    total = 0
    feature_total = 0
    instance_total = 0
    positive_total = 0
    test_total = 0


    for i in range(10):
        chunks = total_chunks[:]
        test_sentences = chunks.pop(i)
        training_sentences = list(itertools.chain.from_iterable(chunks))
        training_instances, word_dictionary, dep_dictionary, between_word_dictionary = load_data.build_instances_training(training_sentences, distant_interactions, None, hiv_genes, True)
        test_instances = load_data.build_instances_testing(test_sentences, word_dictionary, dep_dictionary, between_word_dictionary, distant_interactions, None, hiv_genes, True)

        print('training linear regression')

        X = []
        y = []
        positive_instances = 0
        print(len(training_instances))
        instance_total += len(training_instances)
        for t in training_instances:
            #print("training instance")
            #print(t.label)
            #print(t.features)
            #print(t.get_type_dependency_path())
            #print(t.get_reverse_type_dependency_path())
            #print(t.get_dep_word_path())
            #print(t.get_between_words())
            #print(t.sentence.get_entities())
            X.append(t.features)
            y.append(t.label)
            if t.label == 1:
                positive_instances += 1
        print(positive_instances)
        positive_total += positive_instances

        X_train = np.array(X)
        y_train = np.ravel(y)

        model = LogisticRegression()
        model.fit(X_train,y_train)

        X = []
        y = []
        print(len(test_instances))
        for t in test_instances:
            #print("test instance")
            #print(t.label)
            #print(t.features)
            #print(t.get_type_dependency_path())
            #print(t.get_reverse_type_dependency_path())
            #print(t.get_dep_word_path())
            #print(t.get_between_words())
            #print(t.sentence.get_entities())
            X.append(t.features)
            y.append(t.label)

        feature_total += len(t.features)
        test_total += len(test_instances)
        X_test = np.array(X)
        y_test = np.array(y)

        predicted = model.predict(X_test)
        f1 =  metrics.f1_score(y_test,predicted)
        print(f1)
        total += f1

    print(total/10)
    print(feature_total/10)
    print(instance_total/10)
    print(positive_total/10)
    print(test_total/10)
    print(dep_dictionary)

    print(word_dictionary)
    print(between_word_dictionary)

if __name__=="__main__":
    main()
