import auxiliary_functions as utils
import numpy as np

G = utils.G

##
## Naïve Bayes Model
##
## Sequential Holdout 80/20 split

data,labels,individuals = utils.load_dataset()

data_train = data[:,:,0:int(2045*0.8)]   #1636 of instances (80%) to train model
data_test = data[:,:,int(2045*0.8):2045] #409 of instances (20%) to evaluate model

labels_train = labels[0:int(2045*0.8)]
labels_test = labels[int(2045*0.8):2045]

M = utils.learn_model(data_train,labels_train)
probs = utils.classify_instances(data_test, M)

A = utils.Accuracy(probs, labels_test)
print(A)


##
## Linear Gaussian Model
##
## Sequential Holdout 80/20 split

data,labels,individuals = utils.load_dataset()

data_train = data[:,:,0:int(2045*0.8)]   #1636 of instances (80%) to train model
data_test = data[:,:,int(2045*0.8):2045] #409 of instances (20%) to evaluate model

labels_train = labels[0:int(2045*0.8)]
labels_test = labels[int(2045*0.8):2045]

M = utils.learn_model(data_train,labels_train,G)
probs = utils.classify_instances(data_test, M)


A = utils.Accuracy(probs, labels_test)
print(A)


##
## Naïve Bayes Model
##
## Random Holdout 80/20 split

data,labels,individuals = utils.load_dataset()

shuffledIndex = utils.RandomVect(2045,0,2044)

data_train = data[:,:,shuffledIndex[0:int(2045*0.8)]]   #1636 of instances (80%) to train model
data_test = data[:,:,shuffledIndex[int(2045*0.8):2045]] #409 of instances (20%) to evaluate model

labels_train = labels[shuffledIndex[0:int(2045*0.8)]]
labels_test = labels[shuffledIndex[int(2045*0.8):2045]]

M = utils.learn_model(data_train,labels_train)
probs = utils.classify_instances(data_test, M)


A = utils.Accuracy(probs, labels_test)
print(A)


##
## Linear Gaussian Model
##
## Random Holdout 80/20 split

data,labels,individuals = utils.load_dataset()

shuffledIndex = utils.RandomVect(2045,0,2044)

data_train = data[:,:,shuffledIndex[0:int(2045*0.8)]]   #1636 of instances (80%) to train model
data_test = data[:,:,shuffledIndex[int(2045*0.8):2045]] #409 of instances (20%) to evaluate model

labels_train = labels[shuffledIndex[0:int(2045*0.8)]]
labels_test = labels[shuffledIndex[int(2045*0.8):2045]]

M = utils.learn_model(data_train,labels_train, G)

probs = utils.classify_instances(data_test, M)

A = utils.Accuracy(probs, labels_test)
print(A)



##
## Naïve Bayes Model
##
## Stratified Crossvalidation 4Fold

data,labels,individuals = utils.load_dataset()
[M, A, measures] = utils.GetPerformance(data, np.concatenate(labels), 4)

print("Accuracies       = " + str(measures))
print("Best Accuracy    = " + str(A))
print("Average Accuracy = " + str(np.average(measures)))


##
## Linear Gaussian Model
##
## Stratified Crossvalidation 4Fold

data,labels,individuals = utils.load_dataset()
[M, A, measures] = utils.GetPerformance(data, np.concatenate(labels), 4, G)

print("Accuracies       = " + str(measures))
print("Best Accuracy    = " + str(A))
print("Average Accuracy = " + str(np.average(measures)))