import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
digits = datasets.load_digits()
# print(digits.data)
# print(digits.target)

clf = svm.SVC(gamma=0.001, C=100)

# train_X, test_X, train_y, test_y = train_test_split(digits.data, digits.target, test_size=1/3, random_state=42)
# clf.fit(train_X,train_y)

# def accuracyPercent()

# lbls_pred =clf.predict(test_X)

# print("Aantal goed voorgespelde waardes: ", accuracy)
X_,y = digits.data[:-10], digits.target[:-10]
clf.fit(X_,y)
n_classes = len(np.unique(X_))
# print(type(X_))
# print(X_)
# print(n_classes)
def random_13de_traning(dig):
    data_length = len(dig.data)
    training_index = []
    testing_index = []
    X = np.ndarray((0, 64))
    y = np.ndarray((0,))
    testing_data = np.ndarray((0, 64))
    testing_lbl = np.ndarray((0,))
    
    while len(training_index) <= (data_length* 2/3 ):
        rnd_nmbr= random.randrange(data_length-1)
        if training_index.count(rnd_nmbr) == 0:
            training_index.append(rnd_nmbr)
            X = np.vstack((X, dig.data[rnd_nmbr]))
            y = np.append(y, dig.target[rnd_nmbr])

    training_index.sort()
    for number in range(1,data_length):
        if number not in training_index:
            testing_index.append(number) 
    test_data = []
    test_lbl = []
    for i in testing_index:
        testing_data = np.vstack((testing_data, dig.data[i]))
        testing_lbl = np.append(testing_lbl, dig.target[i])
    # test_data = test_data.reshape(-1,1) 
    
    return X,y,testing_data, testing_lbl

rnd_train_data, rnd_train_lbls, testing_data, testing_lbl = random_13de_traning(digits)
print("data: ",testing_data)
clf.fit(rnd_train_data,rnd_train_lbls)
def calcPer(data, lbls):
    count  = 0
    for i in range(0,len(data)-1):
        predict  = clf.predict(data[i].reshape(1,-1))
        if predict == lbls[i]:
            count += 1
    return count / len(data) * 100

print("De berekende accuracy is het volgende: ",round(calcPer(testing_data, testing_lbl)) , "%")


