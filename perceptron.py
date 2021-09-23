# ----------------------------------------------------------------
#
#
# Taylor Thomas
# Implementation of TF-IDF aided by:
# https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
# --------------------------------------------------------------
#from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
import random
from random import seed
import pandas as pd
from random import randrange
from csv import reader
from sklearn.feature_extraction.text import TfidfVectorizer

seed(0)
def load_synthetic_data(dimensions):
    """Creating synthetic data here."""
    num_rows = 115
    num_cols = dimensions
    dataset = list()
    for rows in range(num_rows):
        row = []
        for cols in range(num_cols):
            row.append(round(random.random() * 2 - 1, 1))
            row.append(random.randint(0,1))
        dataset.append(row)
    return dataset


def load_bag_of_words():
    documentA = 'crawler of the attic man'
    documentB = 'beyond the jumper of dog likes'
    documentC = 'jumped the dog'
    documentD = 'super duper the fire in the attic was not great'

    # bag_of_Words_A = documentA.split(' ')
    # bag_of_Words_B = documentB.split(' ')
    # bag_of_Words_C = documentC.split(' ')
    # bag_of_Words_D = documentC.split(' ')
    # uniqueWords = set(bag_of_Words_A).union(set(bag_of_Words_B)).union(set(bag_of_Words_C)).union(set(bag_of_Words_D))
    #
    # num_of_words_A = dict.fromkeys(uniqueWords,0)
    # for word in bag_of_Words_A:
    #     num_of_words_A[word] += 1
    #
    # num_of_words_B = dict.fromkeys(uniqueWords, 0)
    # for word in bag_of_Words_B:
    #     num_of_words_B[word] += 1
    #
    # num_of_words_C = dict.fromkeys(uniqueWords, 0)
    # for word in bag_of_Words_C:
    #     num_of_words_C[word] += 1
    #
    # num_of_words_D = dict.fromkeys(uniqueWords, 0)
    # for word in bag_of_Words_D:
    #     num_of_words_D[word] += 1
    #
    # # from nltk.corpus import stopwords
    # # stopwords.words('english')
    #
    # tfA = computeTF(num_of_words_A, bag_of_Words_A)
    # tfB = computeTF(num_of_words_B, bag_of_Words_B)
    # tfC = computeTF(num_of_words_C, bag_of_Words_C)
    # tfD = computeTF(num_of_words_D, bag_of_Words_D)
    #
    # idfs = computeIDF([num_of_words_A, num_of_words_B, num_of_words_C, num_of_words_D])
    #
    # tfidfA = computeTFIDF(tfA, idfs)
    # tfidfB = computeTFIDF(tfB, idfs)
    # tfidfC = computeTFIDF(tfC, idfs)
    # tfidfD = computeTFIDF(tfD, idfs)
    # df = pd.DataFrame([tfidfA, tfidfB, tfidfC, tfidfD])
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([documentA, documentB, documentC])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    dataset = df.values.tolist()
    for row in dataset:
        row.append(random.randint(0,1))
    return dataset



def computeTF(wordDict, bagOfWords):
    """ Compute Term Frequency"""
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


def computeIDF(documents):
    """ log of num documents that contain word """
    import math
    N = len(documents)

    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

# Make a prediction with weights
def predict(row, weights):
    #print(row)
    #print(weights)
    activation = weights[0]  # bias
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


# testing predictions
dataset = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0], [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0], [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1], [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]

weights = [-.1, 0.20653640140000007, -0.23418117710000003]  # weights

for row in dataset:
    prediction = predict(row, weights)
    print("Expected=%d, Predicted=%d" % (row[-1], prediction))


# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        print('>epoch=%d, lrate=%.5f, error=%.3f' % (epoch, l_rate, sum_error))
        print(weights)
    return weights


# l_rate = .1
# n_epoch = 5
# weights = train_weights(dataset, l_rate, n_epoch)
# print(weights)

# Perceptron Algoirthm on the Sonar DataSet



# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    class_report = [0,0,0]
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
        # calculating precision, recall, f1 score using scikit-learn
        class_report[0] += classification_report(actual, predicted, labels=[1], output_dict=1, digits=4)['1']['precision']
        class_report[1] += classification_report(actual, predicted, labels=[1], output_dict=1, digits=4)['1']['recall']
        class_report[2] += classification_report(actual, predicted, labels=[1], output_dict=1, digits=4)['1']['f1-score']
        # use scikit to find confusion matrix and compute metrics
        conf_matrix = (actual, predicted)

    return (scores, class_report[0] , class_report[1], class_report[2], conf_matrix)


# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
    predictions = list()
    weights = train_weights(train, l_rate, n_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return (predictions)


# Test the Perceptron algorithm on the sonar dataset
seed(0)
is_synthetic = 0
user_input = input("Synthetic data? (y/n)")
if user_input == "y" or user_input == "yes":
    is_synthetic = 1
if is_synthetic == 0:
    # load and prepare data
    user_input_2 = input("TF-IDF? (y/n)")
    if user_input_2 == "y" or user_input_2 == "yes":
        dataset = load_bag_of_words()
    else:
        filename = 'sonar.all-data.csv'
        filename = 'small_dataset.csv'
        dataset = load_csv(filename)
        for i in range(len(dataset[0]) - 1):
            str_column_to_float(dataset, i)

else:
    num_dimensions = int(input("How many dimensions?"))
    dataset = load_synthetic_data(num_dimensions)
 # convert string class to integers
str_column_to_int(dataset, len(dataset[0]) - 1)
# evaluate algorithm
n_folds = 3
l_rate = .01
n_epoch = 500
print(dataset)
output = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
print('Scores: %s' % output[0])
print('Precision: %.3f%% | Recall: %.3f%% | F1-Score: %.3f%% ' % (output[1]/float(len(output[0])),
      output[2]/float(len(output[0])), output[3]/float(len(output[0]))) )  # printing other metrics
print('Mean Accuracy: %.3f%%' % (sum(output[0]) / float(len(output[0]))))
print("Confusion Matrix:", output[4])

