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
    print(len(total_chunks))
    total = 0
    for i in range(10):
        chunks = total_chunks[:]
        print(len(chunks))
        test_sentences = chunks.pop(i)
        print(len(test_sentences))
        training_sentences = list(itertools.chain.from_iterable(chunks))
        print(len(training_sentences))
        training_instances, word_dictionary, dep_dictionary, between_word_dictionary = load_data.build_instances_training(training_sentences, distant_interactions, None, hiv_genes, True)
        print(len(dep_dictionary))
        test_instances = load_data.build_instances_testing(test_sentences, word_dictionary, dep_dictionary, between_word_dictionary, distant_interactions, None, hiv_genes, True)

        print('training linear regression')

        X = []
        y = []
        positive_instances = 0
        for t in training_instances:
            X.append(t.features)
            y.append(t.label)
            if t.label == 1:
                positive_instances += 1


        X_train = np.array(X)
        y_train = np.ravel(y)

        model = LogisticRegression()
        model.fit(X_train,y_train)

        X = []
        y = []
        for t in test_instances:
            X.append(t.features)
            y.append(t.label)

        X_test = np.array(X)
        y_test = np.array(y)

        predicted = model.predict(X_test)
        f1 =  metrics.f1_score(y_test,predicted)
        print(f1)
        total += f1

    print(total/10)


if __name__=="__main__":
    main()
