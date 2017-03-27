
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import numpy as np
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from pylab import *
from numpy import *
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_mnist(dataset, digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
        limit = 20000
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
        limit = 5000
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    images, labels = images.reshape(len(images), -1), labels.reshape(len(labels), )
    images = images / 255.0
    images, labels = images[:limit], labels[:limit]

    return images, labels

images, labels = load_mnist('training')
images_Testing, labels_Testing = load_mnist('testing')


##### Hidden nodes and layers ###############

#train_scores, test_scores = validation_curve(MLPClassifier(random_state=1),
#images, labels, "hidden_layer_sizes", [(50,), (100,),(200,),(300,),(500,)])
#print(train_scores)
#print (test_scores)
#highest score is with 500 hidden nodes in 1 layer

#train_scores_mean = np.mean(train_scores, axis=1)
#test_scores_mean = np.mean(test_scores, axis=1)
#print(train_scores_mean)

#MLP with 1 hidden layer of size 500
mlp = MLPClassifier(hidden_layer_sizes=(500,), random_state=0)
mlp.fit(images, labels)
print(mlp.score(images_Testing, labels_Testing))
#score is 0.9644

#train_scores, test_scores = validation_curve(MLPClassifier(random_state=1),
#images, labels, "hidden_layer_sizes", [(50,50), (100,100),(200,200),(300,300),(500,500)])
#print(train_scores)
#print (test_scores)
#highest score is with 2 layers, 500 nodes each

#MLP with 2 hidden layer of size 500
mlp = MLPClassifier(hidden_layer_sizes=(500,500), random_state=0)
mlp.fit(images, labels)
print(mlp.score(images_Testing, labels_Testing))
#score is 0.9688

##### Activation functions ###############

#MLP with 2 hidden layer each of size 500 with activation=identity
mlp = MLPClassifier(hidden_layer_sizes=(500,500), random_state=0,activation='identity')
mlp.fit(images, labels)
print(mlp.score(images_Testing, labels_Testing))
#score is 0.877

#MLP with 2 hidden layer each of size 500 with activation=logistic
mlp = MLPClassifier(hidden_layer_sizes=(500,500), random_state=0,activation='logistic')
mlp.fit(images, labels)
print(mlp.score(images_Testing, labels_Testing))
#score is 0.9612

#MLP with 2 hidden layer each of size 500 with activation=tanh
mlp = MLPClassifier(hidden_layer_sizes=(500,500), random_state=0,activation='tanh')
mlp.fit(images, labels)
print(mlp.score(images_Testing, labels_Testing))
#score is 0.961

##### learning rates ###############

#MLP with 2 hidden layer each of size 500 with learning rate=invscaling
#mlp = MLPClassifier(hidden_layer_sizes=(500,500), random_state=0, learning_rate ='invscaling', solver='sgd')
#mlp.fit(images, labels)
#print(mlp.score(images_Testing, labels_Testing))
#score is 0.655

#MLP with 2 hidden layer each of size 500 with learning rate=adaptive
mlp = MLPClassifier(hidden_layer_sizes=(500,500), random_state=0, learning_rate ='adaptive', solver='sgd')
mlp.fit(images, labels)
print(mlp.score(images_Testing, labels_Testing))
#score is 0.9514

#MLP with 2 hidden layer each of size 500 with learning rate=constant
#mlp = MLPClassifier(hidden_layer_sizes=(500,500), random_state=0, learning_rate ='constant', solver='sgd')
#mlp.fit(images, labels)
#print(mlp.score(images_Testing, labels_Testing))
#score is 0.9514

#train_scores, test_scores = validation_curve(MLPClassifier(hidden_layer_sizes=(500,500), random_state=1, 
#    learning_rate ='adaptive', solver='sgd'),
#images, labels, "learning_rate_init", [0.05,0.1,0.15,0.2, 0.25])
#print(train_scores)
#print (test_scores) 
#highest score is with initial learning rate of 0.25

#train_scores_mean = np.mean(train_scores, axis=1)
#test_scores_mean = np.mean(test_scores, axis=1)
#print(train_scores_mean)
#print(test_scores_mean)

#MLP with 2 hidden layer each of size 500 with learning rate=adaptive and initial value for learning rate = 0.25
mlp = MLPClassifier(hidden_layer_sizes=(500,500), random_state=0, learning_rate ='adaptive', learning_rate_init=0.25, solver='sgd')
mlp.fit(images, labels)
print(mlp.score(images_Testing, labels_Testing))
#score is 0.9688

##### Momentum ###############

#train_scores, test_scores = validation_curve(MLPClassifier(hidden_layer_sizes=(500,500), random_state=0, 
#    learning_rate ='adaptive', learning_rate_init=0.25, solver='sgd'),
#images, labels, "momentum", [0.1,0.5,0.8,0.95, 1])
#print(train_scores)
#print (test_scores)
#highest score is with Momentum 0.8

#train_scores_mean = np.mean(train_scores, axis=1)
#test_scores_mean = np.mean(test_scores, axis=1)
#print(train_scores_mean)
#print(test_scores_mean)

#MLP with 2 hidden layer each of size 500 with momentum of 0.8
mlp = MLPClassifier(hidden_layer_sizes=(500,500), random_state=0, 
    learning_rate ='adaptive', learning_rate_init=0.25, solver='sgd', momentum=0.8)
mlp.fit(images, labels)
print(mlp.score(images_Testing, labels_Testing))
#score is 0.9664


##### Batch Size ###############

#MLP with 2 hidden layer each of size 500 with learning rate=adaptive and initial value for learning rate = 0.25
#and batch size was eqaul to 20000 which is the size of the whole dataset
#mlp = MLPClassifier(hidden_layer_sizes=(500,500), random_state=0, 
#    learning_rate ='adaptive', learning_rate_init=0.25, solver='sgd', 
#    batch_size = 20000)
#mlp.fit(images, labels)
#print(mlp.score(images_Testing, labels_Testing))
#score is 0.476

#MLP with 2 hidden layer each of size 500 with learning rate=adaptive and initial value for learning rate = 0.25
#and batch size was eqaul to 1 which is sequential
#mlp = MLPClassifier(hidden_layer_sizes=(500,500), random_state=0, 
#    learning_rate ='adaptive', learning_rate_init=0.25, solver='sgd', 
#    batch_size = 1)
#mlp.fit(images, labels)
#print(mlp.score(images_Testing, labels_Testing))
#sTook more than 6 hours to run, didn't reach score.

#train_scores, test_scores = validation_curve(MLPClassifier(hidden_layer_sizes=(500,500), random_state=0, 
#    learning_rate ='adaptive', learning_rate_init=0.25, solver='sgd'),
#images, labels, "batch_size", [100,250,300,500, 700])
#print(train_scores)
#print (test_scores)
#highest score is with batch size 250

#MLP with 2 hidden layer each of size 500 with learning rate=adaptive and initial value for learning rate = 0.1 
#and batch size was eqaul to 250 => minibatch applied
#mlp = MLPClassifier(hidden_layer_sizes=(500,500), random_state=0, 
#    learning_rate ='adaptive', learning_rate_init=0.25, solver='sgd', batch_size=250)
#mlp.fit(images, labels)
#print(mlp.score(images_Testing, labels_Testing))
#score is 0.967

##### Early stopping ###########
#MLP with 2 hidden layer each of size 500 with learning rate=adaptive and initial value for learning rate = 0.1 
#and early stopping applied.
mlp = MLPClassifier(hidden_layer_sizes=(500,500), random_state=0, 
    learning_rate ='adaptive', learning_rate_init=0.25, solver='sgd', early_stopping=True)
mlp.fit(images, labels)
print(mlp.score(images_Testing, labels_Testing))
#score is 0.9636

##### revisiting ###############

#MLP with 3 hidden layer each of size 20
#mlp = MLPClassifier(hidden_layer_sizes=(20,20,20), random_state=0)
#mlp.fit(images, labels)
#print(mlp.score(images_Testing, labels_Testing))
#score is 0.9276

#MLP with 4 hidden layer each of size 20
#mlp = MLPClassifier(hidden_layer_sizes=(20,20,20,20), random_state=0)
#mlp.fit(images, labels)
#print(mlp.score(images_Testing, labels_Testing))
#score is 0.924

#MLP with 2 hidden layer each of size 20 with learning rate=adaptive and initial value for learning rate = 0.1 and early stopping applied
#mlp = MLPClassifier(hidden_layer_sizes=(20,20), random_state=0, learning_rate ='adaptive', learning_rate_init=0.1, solver='sgd', 
#    early_stopping =True)
#mlp.fit(images, labels)
#print(mlp.score(images_Testing, labels_Testing))
#score is 0.9362

#MLP with 2 hidden layer each of size 20 with learning rate=adaptive and initial value for learning rate = 0.1 and early stopping applied
#and momentum 0.95
#mlp = MLPClassifier(hidden_layer_sizes=(20,20), random_state=0, learning_rate ='adaptive', learning_rate_init=0.1, solver='sgd', 
#    early_stopping =True, momentum=0.95)
#mlp.fit(images, labels)
#print(mlp.score(images_Testing, labels_Testing))
#score is 0.9336

#MLP with 2 hidden layer each of size 20 with learning rate=adaptive and initial value for learning rate = 0.1 and early stopping applied
#and batch size was eqaul to 20000 which is the size of the whole dataset
#mlp = MLPClassifier(hidden_layer_sizes=(20,20), random_state=0, learning_rate ='adaptive', learning_rate_init=0.1, solver='sgd', 
#    early_stopping =True, batch_size = 20000)
#mlp.fit(images, labels)
#print(mlp.score(images_Testing, labels_Testing))
#score is 0.754

#MLP with 2 hidden layer each of size 20 with learning rate=adaptive and initial value for learning rate = 0.1 and early stopping applied
#and batch size was eqaul to 1 => sequential
#mlp = MLPClassifier(hidden_layer_sizes=(20,20), random_state=0, learning_rate ='adaptive', learning_rate_init=0.1, solver='sgd', 
#    early_stopping =True, batch_size = 1)
#mlp.fit(images, labels)
#print(mlp.score(images_Testing, labels_Testing))
#score is 0.1142

#MLP with 2 hidden layer each of size 20 with learning rate=adaptive and initial value for learning rate = 0.1 and early stopping applied
#and batch size was eqaul to 100 => minibatch applied
#mlp = MLPClassifier(hidden_layer_sizes=(20,20), random_state=0, learning_rate ='adaptive', learning_rate_init=0.1, solver='sgd', 
#    early_stopping =True, batch_size = 100)
#mlp.fit(images, labels)
#print(mlp.score(images_Testing, labels_Testing))
#score is 0.9336

#MLP with 2 hidden layer each of size 20 with learning rate=adaptive and initial value for learning rate = 0.1 and early stopping applied
#and batch size was eqaul to 300 => minibatch applied
#mlp = MLPClassifier(hidden_layer_sizes=(20,20), random_state=0, learning_rate ='adaptive', learning_rate_init=0.1, solver='sgd', 
#    early_stopping =True, batch_size = 300)
#mlp.fit(images, labels)
#print(mlp.score(images_Testing, labels_Testing))
#score is 0.9292

#MLP with 2 hidden layer each of size 20 with learning rate=adaptive and initial value for learning rate = 0.1 and early stopping applied
#and batch size was eqaul to 400 => minibatch applied
#mlp = MLPClassifier(hidden_layer_sizes=(20,20), random_state=0, learning_rate ='adaptive', learning_rate_init=0.1, solver='sgd', 
#    early_stopping =True, batch_size = 400)
#mlp.fit(images, labels)
#print(mlp.score(images_Testing, labels_Testing))
#score is 0.9306

########### SVC ####################


#svc with linear kernel 
svc = SVC(C=1, kernel = 'linear', random_state=0)
svc.fit(images,labels)
print(svc.score(images_Testing, labels_Testing))
#score is 0.9016

#svc with polynomial kernel 
#svc = SVC(C=1, kernel = 'poly', random_state=0)
#svc.fit(images,labels)
#print(svc.score(images_Testing, labels_Testing))
#score is 0.1918

#svc with radial base kernel 
svc = SVC(C=1, kernel = 'rbf', random_state=0)
svc.fit(images,labels)
print(svc.score(images_Testing, labels_Testing))
#score is 0.908

#svc with sigmoid kernel 
#svc = SVC(C=1, kernel = 'sigmoid', random_state=0)
#svc.fit(images,labels)
#print(svc.score(images_Testing, labels_Testing))
#score is 0.8934

#train_scores, test_scores = validation_curve(SVC(kernel='linear', random_state = 0),
#images, labels, "C", [0.1,0.5,1.5,2, 3])

#print(train_scores)
#print (test_scores)
#highest score is with C = 0.1

#train_scores_mean = np.mean(train_scores, axis=1)
#test_scores_mean = np.mean(test_scores, axis=1)
#print(train_scores_mean)
#print(test_scores_mean)

#svc with linear kernel and C=0.1
svc = SVC(C=0.1, kernel = 'linear', random_state=0)
svc.fit(images,labels)
print(svc.score(images_Testing, labels_Testing))
#score is 0.9178

#svc with rbf kernel and C=0.1
#svc = SVC(C=0.1, kernel = 'rbf', random_state=0)
#svc.fit(images,labels)
#print(svc.score(images_Testing, labels_Testing))
#score is 0.863

###Testing default values for svc#######
svc = SVC(C=3.5,coef0=0.0,
    degree=6, gamma='auto', kernel='rbf',
    random_state=0)
svc.fit(images,labels)
print(svc.score(images_Testing, labels_Testing))
#score is 0.9224 ===> highest score

#### aplying PCA on the best combination for the SVC ##########
dataset = np.concatenate((images,images_Testing),axis=0)
pca = PCA(n_components =30)
images_proj = pca.fit_transform(dataset)
print('images dimensionality before PCA:', dataset.shape)
print('images dimensionality after PCA:', images_proj.shape)

images_reshaped = images_proj.reshape(len(images_proj), -1)
print('images reshaped',images_reshaped.shape)

new_images = images_reshaped[:20000]
new_testing_images = images_reshaped[-5000:]

print('training split',new_images.shape)
print('test split',new_testing_images.shape)

#applying svc on data resulting from PCA
#svc with linear kernel and C=0.1
svc = SVC(C=3.5,coef0=0.0,
    degree=6, gamma='auto', kernel='rbf',
    random_state=0)
svc.fit(new_images,labels)
#print(svc.score(new_testing_images, labels_Testing))
# 120 components ==> score is 0.9468
# 150 components ==> score is 0.9418
# 100 components ==> score is 0.9522
# 80 components ==> score is 0.9558
# 60 components ==> score is 0.9602
# 40 components ==> score is 0.9608
# 35 components ==> score is 0.962
# 30 components ==> score is 0.9622
# 25 components ==> score is 0.9616

#30 components, C=2 ===> score is 0.9668
#30 components, C=3 ===> score is 0.968
#30 components, C=3.5 ===> score is 0.9682