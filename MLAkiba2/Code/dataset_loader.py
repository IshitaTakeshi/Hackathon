from __future__ import division

import sys
import csv
import json

import numpy as np
from sklearn import datasets
from matplotlib import pyplot, cm


digits = datasets.load_digits(2)
images = digits.images.astype('f')
labels = digits.target


def export_as_json(filename, images=images, labels=labels):
    d = {'images':images, 'labels':labels.tolist()}
    json.dump(d, open(filename, 'wb'))


def load_from_json(filename):
    d = json.load(open(filename, 'rb'))
    return np.array(d['images']), np.array(d['labels'])


def export_as_csv(filename, images=images, labels=labels):
    writer = csv.writer(open(filename, 'wb'))
    writer.writerow(labels)
    writer.writerows(images)


def load_from_csv(filename):
    reader = csv.reader(open(filename, 'rb'))
    labels = reader.next()
    labels = map(float, labels)

    images = []
    for row in reader:
        image = map(float, row)
        images.append(image)
    return np.array(images), np.array(labels)


def show_digits(images=images, (n_rows, n_columns)=(5, 5)):
    n_images = n_columns * n_rows
    images = images[:n_images]

    figure = pyplot.figure()
    for i, image in enumerate(images):
        figure.add_subplot(n_columns, n_rows, i)  
        pyplot.imshow(image, cmap=pyplot.cm.gray_r, interpolation='nearest')


def classify(digits, labels):
    positive = []
    negative = []
    for digit, label in zip(digits, labels):
        digit = digit.reshape(8, 8)
        if(label == 1):
            positive.append(digit)
        elif(label == -1):
            negative.append(digit)
    return positive, negative


def print_accuracy(results, answers):
    n_corrects = 0
    for r, a in zip(results, answers):
        if(r == a):
            n_corrects += 1
    print("Accuracy: {}".format(n_corrects/len(results)))

    
def to_bin_labels(labels, focused_labels):
    for i in range(len(labels)):
        if(labels[i] == focused_labels[0]):
            labels[i] = 1
        elif(labels[i] == focused_labels[1]):
            labels[i] = -1
    return labels


if(__name__ == '__main__'):
    n_trains = int(len(images)*0.8)
    train_json, test_json = sys.argv[1:3]
    #train_csv, test_csv = sys.argv[1:3]

    X, y = load_from_json(train_json)
    test_digits, answers = load_from_json(test_json)
    print("X:\n{}".format(X))
    print("y:\n{}".format(y))
    print("test_digits:\n{}".format(test_digits))
    print("answers:\n{}".format(answers))
    #X, y = load_from_csv(train_csv)
    #test_digits, answers = load_from_csv(test_csv)
