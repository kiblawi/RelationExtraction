import sys

import load_parse_data
import structures
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
def main():
    dep_type_vocabulary, word_vocabulary, candidate_sentences = load_parse_data.load_xml(sys.argv[1])
    training_instances, word_dictionary, dep_dictionary = load_parse_data.feature_construction(dep_type_vocabulary, word_vocabulary, candidate_sentences)
    print('training linear regression')

    X = []
    y = []
    for t in training_instances:
        X.append(t.features)
        y.append(t.label)
    X = np.array(X)

    y = np.ravel(y)
    model = LogisticRegression()
    model = model.fit(X,y)
    print(model.score(X,y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)
    predicted = model2.predict(X_test)
    print predicted
    probs = model2.predict_proba(X_test)
    print probs
    print metrics.accuracy_score(y_test, predicted)
    print metrics.roc_auc_score(y_test, probs[:, 1])
    print metrics.confusion_matrix(y_test, predicted)
    print metrics.classification_report(y_test, predicted)

    scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
    print scores
    print scores.mean()



if __name__=="__main__":
    main()
