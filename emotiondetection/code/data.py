import numpy as np

def _get_feature(dataset, names):
    data_path = 'data/lld/lld/' + dataset + '/'
    res = []
    for name in names:
        tmp = []
        data_file = open(data_path + name + '.csv', 'r')
        for line in data_file:
            tmp.append(float(line))
        res.append(tmp)
    return np.array(res)

def _get_info(dataset):
    name_file = open('data/lld/labels/' + dataset + '.txt', 'r')
    name = []
    label = []
    for line in name_file:
        line = line.split()
        name.append(line[0])
        label.append(line[1])
    return name, label

class Data:
    def __init__(self, dataset):
        self.name, self.label = _get_info(dataset)
        self.feature = _get_feature(dataset, self.name)
        print 'Finish reading dataset: ' + dataset
