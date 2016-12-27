import os

for model_name in ('logistic_regression', 'naive_bayes', 'decision_tree', 'SVM'):
    print 'Model: ' + model_name
    os.system('./code/score.py result/' + model_name + '.txt data/lld/labels/test.txt')
