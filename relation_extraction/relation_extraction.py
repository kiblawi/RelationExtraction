import sys

import load_data
import structures
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
def main():
    candidate_sentences = load_data.load_xml(sys.argv[1], 'HUMAN_GENE', 'VIRAL_GENE')
    print(len(candidate_sentences))
    distant_interactions = load_data.load_distant_kb(sys.argv[2],4,0)

    training_instances, word_dictionary, dep_dictionary = load_data.build_instances(candidate_sentences, distant_interactions)

    print(len(training_instances))
    print(word_dictionary)
    print(len(word_dictionary))
    print(dep_dictionary)
    print(len(dep_dictionary))

    print('training linear regression')

    X = []
    y = []
    for t in training_instances:
        X.append(t.features)
        y.append(t.label)
    X = np.array(X)
    print(y)
    y = np.ravel(y)
    print(y)
    print(len(y))


    scores = cross_val_score(LogisticRegression(), X, y, scoring='f1', cv=10)
    print scores
    print scores.mean()



if __name__=="__main__":
    main()
