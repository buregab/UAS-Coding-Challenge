import os
import numpy as np
from sklearn.svm import SVC

training_dir = 'digits/trainingDigits/'
testing_dir =  'digits/testDigits/'

# get the training data as arrays and get the training labels
training_data = []
training_labels = []
for file_name in os.listdir(training_dir):
    f = open(training_dir + file_name, 'r')
    text = f.read()
    row = [x for x in text if x != '\n']
    training_data.append(row)
    label = file_name.split('_')[0]
    training_labels.append(label)

# get the testing data as arrays and get the testing labels
testing_data = []
testing_labels = []
for file_name in os.listdir(testing_dir):
    f = open(testing_dir + file_name, 'r')
    text = f.read()
    row = [x for x in text if x != '\n']
    testing_data.append(row)
    label = file_name.split('_')[0]
    testing_labels.append(label)

# fit and predict using an SMV model
clf = SVC()
clf.fit(training_data, training_labels)
predictions = clf.predict(testing_data)
score = clf.score(testing_data, testing_labels)

print(predictions)
print(testing_labels)
print(score)
# when running this I got a score of 0.973572938689
