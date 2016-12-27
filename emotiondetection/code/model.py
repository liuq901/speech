from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

def _get_raw_model(name):
    if name == 'logistic_regression':
        return LogisticRegression(class_weight = 'balanced')
    elif name == 'naive_bayes':
        return GaussianNB()
    elif name == 'decision_tree':
        return DecisionTreeClassifier(class_weight = 'balanced')
    elif name == 'SVM':
        return SVC(class_weight = 'balanced')
    else:
        raise ValueError('No such model')

def get_model(name):
    model = _get_raw_model(name)
    return OneVsRestClassifier(model)
