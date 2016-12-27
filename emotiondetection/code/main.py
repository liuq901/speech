import data
import model
from sklearn.metrics import precision_recall_fscore_support as score

def training(model_name):
    name, label = data.get_info('train')
    feature = data.get_feature('train', name)
    classifier = model.get_model(model_name)
    classifier.fit(feature, label)
    return classifier

for model_name in ('logistic_regression', 'naive_bayes', 'decision_tree', 'SVM'):
    classifier = training(model_name)
    testing(classifier)
