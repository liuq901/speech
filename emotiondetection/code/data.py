import numpy as np

def get_feature(dataset, names):
    data_path = 'data/lld/lld/' + dataset + '/'
    res = []
    for name in names:
        tmp = []
        data_file = open(data_path + name + '.csv', 'r')
        for line in data_file:
            tmp.append(float(line))
        res.append(tmp)
    return np.array(res)

def get_info(dataset):
    name_file = open('data/lld/labels/' + dataset + '.txt', 'r')
    name = []
    label = []
    vocab = {'A' : 0, 'E' : 1, 'N' : 2, 'P' : 3, 'R' : 4}
    for line in name_file:
        line = line.split()
        name.append(line[0])
        label.append(vocab[line[1]])
    return name, label
