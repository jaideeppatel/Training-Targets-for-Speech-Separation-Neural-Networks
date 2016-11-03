import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import mode
import os
from random import randint

os.chdir('C:/Users/jaide/PycharmProjects/AML-PA2')
# Code to read the training input file
inputdata = pd.read_csv('train.csv')
org_traindata = pd.read_csv('train.csv')


# Code to read the test input file
testdata = pd.read_csv('test.csv')
testrecords=testdata.shape[0]
samplelist = []
# testdata['predclass'] = 'NA'
# testdata.is_copy = False

def preparedata(inputdata,testdata,org_traindata):
    # Prepare train data
    classcol = inputdata['bruises?-bruises']
    del inputdata['bruises?-bruises']
    del inputdata['bruises?-no']
    weighttrain = 1/inputdata.shape[0]
    inputdata['weight'] = weighttrain
    inputdata['class'] = classcol

    # Prepare test data
    classcol = testdata['bruises?-bruises']
    del testdata['bruises?-bruises']
    del testdata['bruises?-no']
    testdata['class'] = classcol
    testdata['predclass'] = 'NA'
    testdata.is_copy = False

    # Prepare test data for Boosting
    classcol = org_traindata['bruises?-bruises']
    del org_traindata['bruises?-bruises']
    del org_traindata['bruises?-no']
    org_traindata['class'] = classcol
    org_traindata['predclass'] = 'NA'
    org_traindata.is_copy = False


def createBags(inputdata):
    trainbag = pd.DataFrame()
    j = 0
    records=int(inputdata.shape[0] * 2 / 100)
    index = records-1
    for j in range(0,records):
        rownum = randint(0,index)
        trainbag = trainbag.append(inputdata.iloc[rownum])
        samplelist.append(rownum)
    return trainbag


# apply decision tree builder in the train data
depth = [1]
noofbags = 1
allpredictions=pd.DataFrame()
# allpredictions['class'] = testdata['class']
accuracylist = []

preparedata(inputdata,testdata,org_traindata)
testrecords=org_traindata.shape[0]
newcolnames = inputdata.columns


for dep in depth:
    repeat=0
    for repeat in range(0,noofbags):
        # Create Bag data for the current iteration...
        train = createBags(inputdata)
        print("samplelist is",samplelist)
        filename = 'trainbag'+str(repeat)+'.csv'
        train.to_csv(filename)
        BuildDesiciontreeBoost(dep)
        allpredictions[repeat] = org_traindata['predclass']
        org_traindata['predclass'] = 'NA'
        org_traindata.is_copy = False

        # Capture the Prediction result
    # print(allpredictions)
    allpredictions.to_csv('allprediction.csv')
    sum_class = np.sum(allpredictions,axis=1)
    predicted_class = []
    x = 0
    for x in range(0,testrecords):
        if sum_class[x]<=noofbags/2:
            predicted_class.append(0)
        else:
            predicted_class.append(1)

    # print(predicted_class)
    org_traindata['predclass'] = predicted_class
    org_traindata.to_csv('testresult.csv')
    # Function Call to find accuracy
    treeaccuracy = FindAccuracy(org_traindata)
    error = 1 - treeaccuracy
    accuracylist.append(treeaccuracy)
    print(treeaccuracy)

    # Function Call to build confucation matrix
    confusion_matrix = ConfusionMatrix(org_traindata)
    print("Confusion Matrix for Depth ",dep," is: ")
    print(confusion_matrix)
    UpdateWeight(inputdata)

def UpdateWeight(inputdata,error):






