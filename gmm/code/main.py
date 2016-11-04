import numpy as np
import matplotlib.pyplot as plt

def read_data(file_name, with_label = True):
    data_file = open('data/' + file_name, 'r')
    data = []
    if with_label:
        label = []
    for line in data_file:
        line = line.split()
        data.append((float(line[0]), float(line[1])))
        if with_label:
            label.append(float(line[2]))
    data_file.close()
    if with_label:
        return np.array(data), np.array(label)
    else:
        return np.array(data)

def gaussian(x, c, mu, sigma):
    tmp = np.exp(-0.5 * (x - mu).T.dot(np.linalg.inv(sigma)).dot(x - mu))
    return c * tmp / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))

def random(m):
    if m == 0:
        return -1, 0
    elif m == 1:
        return 1, 0
    elif m == 2:
        return 0, -1
    else:
        return 0, 1

def train(data, mixture_number, c, mu, sigma):
    data_size = len(data)
    gamma = np.ndarray((mixture_number, data_size))
    for m in xrange(mixture_number):
        for n in xrange(data_size):
            gamma[m][n] = gaussian(data[n], c[m], mu[m], sigma[m])
    gamma /= np.sum(gamma, axis = 0)
    gamma_sum = np.sum(gamma, axis = 1)
    mu = (gamma.dot(data).T / gamma_sum).T
    sigma = np.zeros((mixture_number, 2, 2))
    for m in xrange(mixture_number):
        for n in xrange(data_size):
            tmp = data[n] - mu[m]
            sigma[m] += gamma[m][n] * np.outer(tmp, tmp.T) / gamma_sum[m]
    c = gamma_sum / np.sum(gamma_sum)
    return c, mu, sigma

def argmax(data, c, mu, sigma, mixture_number, label_cnt):
    predict_label = []
    for x in data:
        best = -1e100
        label = -1
        for i in xrange(label_cnt):
            tot = 0
            for m in xrange(mixture_number):
                tot += gaussian(x, c[i][m], mu[i][m], sigma[i][m])
            if tot > best:
                best = tot
                label = i
        assert label != -1
        predict_label.append(label + 1)
    return np.array(predict_label)

def plot(data, color):
    data = data.T
    plt.plot(data[0], data[1], color + 'o')

def evaluate(data, mixture_number, c, mu, sigma):
    res = 0
    for x in data:
        tot = 0
        for i in xrange(mixture_number):
            tot += gaussian(x, c[i], mu[i], sigma[i])
        res += np.log(tot)
    return res

mixture_number = 4
label_cnt = 2
iterations = 10
color = ['r', 'b']

train_data, train_label = read_data('train.txt')
dev_data, dev_label = read_data('dev.txt')
for i in xrange(label_cnt):
    plot(train_data[train_label == i + 1], color[i])
plt.savefig('result/train.jpg')

c = [None] * label_cnt
mu = [None] * label_cnt
sigma = [None] * label_cnt
for i in xrange(label_cnt):
    c[i] = np.full(mixture_number, 1.0 / mixture_number)
    mu[i] = np.ndarray((mixture_number, 2))
    sigma[i] = np.ndarray((mixture_number, 2, 2))
    for m in xrange(mixture_number):
        mu[i][m] = np.array(random(m))
        sigma[i][m] = np.eye(2)

likelihood = [0.0] * iterations
predict = [None] * iterations
for _ in xrange(iterations):
    for i in xrange(label_cnt):
        c[i], mu[i], sigma[i] = train(train_data[train_label == i + 1], mixture_number, c[i], mu[i], sigma[i])
        likelihood[_] += evaluate(train_data[train_label == i + 1], mixture_number, c[i], mu[i], sigma[i])
    predict_label = argmax(dev_data, c, mu, sigma, mixture_number, label_cnt)
    predict[_] = sum(predict_label == dev_label)
plt.clf()
plt.plot(np.arange(1, iterations + 1), likelihood, 'bo-')
plt.savefig('result/likelihood.jpg')
plt.clf()
plt.plot(np.arange(1, iterations + 1), predict, 'ro-')
plt.savefig('result/predict.jpg')

test_data = read_data('test.txt', False)
predict_label = argmax(test_data, c, mu, sigma, mixture_number, label_cnt)
plt.clf()
for i in xrange(label_cnt):
    plot(test_data[predict_label == i + 1], color[i])
plt.savefig('result/test.jpg')
res_file = open('result/result.txt', 'w')
for i in xrange(len(test_data)):
    res_file.write(str(test_data[i][0]) + ' ' + str(test_data[i][1]) + ' ' + str(predict_label[i]) + '\n')
res_file.close()
