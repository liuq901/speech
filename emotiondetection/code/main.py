import data
import model
import random
import numpy as np

random.seed(19930131)
np.random.seed(19930131)

train = data.Data('train')
test = data.Data('test')

mean = train.feature.mean(axis = 0)
std = train.feature.std(axis = 0)
train.feature = (train.feature - mean) / std
test.feature = (test.feature - mean) / std
print 'Finish normalization'

for model_name in ('logistic_regression', 'naive_bayes', 'decision_tree', 'SVM'):
    print 'Training model: ' + model_name
    classifier = model.get_model(model_name)
    classifier.fit(train.feature, train.label)
    predict = classifier.predict(test.feature)
    result = open('result/' + model_name + '.txt', 'w')
    for i in xrange(len(predict)):
        result.write(test.name[i] + ' ' + predict[i] + '\n')
    result.close()
