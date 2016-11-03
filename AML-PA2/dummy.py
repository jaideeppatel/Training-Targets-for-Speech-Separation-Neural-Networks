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
    records=int(inputdata.shape[0] * 75 / 100)
    index = records-1
    for j in range(0,records):
        rownum = randint(0,index)
        trainbag = trainbag.append(inputdata.iloc[rownum])
        samplelist.append(rownum)
    return trainbag


# Main Function - For Bagging
def BuildDesiciontree(inputdepth):
    # Function Call to Construct the Tree Structure
    tree_table=BuildTree(inputdepth)

    # Build a Temp Search Tree
    searchtree = tree_table

    #Function Call to perform the Prediction
    for i in range(0,testdata.shape[0]):
        testnode = testdata.iloc[i]
        GetPredition(searchtree,testnode,i)

    print('Prediction train data is:',org_traindata['predclass'])


# Main Function - For Boosting
def BuildDesiciontreeBoost(inputdepth):
    # Function Call to Construct the Tree Structure
    tree_table=BuildTree(inputdepth)

    # Build a Temp Search Tree
    searchtree = tree_table

    #Function Call to perform the Prediction
    for i in range(0,org_traindata.shape[0]):
        testnode = org_traindata.iloc[i]
        GetPredition(searchtree,testnode,i)

# Function to Calculate the Accuracy
def FindAccuracy(org_traindata):
    count=0
    retval = []
    corincorpred = []
    for i in range(0,org_traindata.shape[0]):
        if org_traindata.loc[i,'predclass']==org_traindata.loc[i,'class']:
            corincorpred.append(1)
            count=count+1
        else:
            corincorpred.append(0)
    accuracy = (count/org_traindata.shape[0])
    retval.append(accuracy)
    retval.append(corincorpred)
    # print("Accuracy is: ",accuracy)
    return retval

# Function to Calculate the Confusion Matrix
def ConfusionMatrix(org_traindata):
    # print("Tree Data Read",org_traindata)
    con_mat = pd.DataFrame(columns=['Actual_Class','Predicted_Class=0','Predicted_Class=1'])
    data00 = org_traindata.loc[(org_traindata['class']==0) & (org_traindata['predclass']==0)].shape[0]
    data01 = org_traindata.loc[(org_traindata['class']==0) & (org_traindata['predclass']==1)].shape[0]
    data10 = org_traindata.loc[(org_traindata['class']==1) & (org_traindata['predclass']==0)].shape[0]
    data11 = org_traindata.loc[(org_traindata['class']==1) & (org_traindata['predclass']==1)].shape[0]
    class0 = pd.DataFrame([['Actual:0',int(data00),int(data01)]],columns=['Actual_Class','Predicted_Class=0','Predicted_Class=1'])
    class1 = pd.DataFrame([['Actual:1',int(data10),int(data11)]],columns=['Actual_Class','Predicted_Class=0','Predicted_Class=1'])
    con_mat = con_mat.append(class0,ignore_index=True)
    con_mat = con_mat.append(class1,ignore_index=True)
    return  con_mat


# Function to build the decision tree for the given train data set (Internally the function calls the Information gain function above...)
def BuildTree(inputdepth):

    # Function to build the tree structure
    depth=0
    nodecount=1
    nodenumber=0
    treedepth=inputdepth
    inc=0
    dflist = [train]
    depthlist=[depth]
    tree_table = pd.DataFrame(columns=['nodenumber','depth','attr','leftvalue','rightvalue','left','right','isleaf','class'])
    new_data = pd.DataFrame(dflist[0])

    while len(dflist)>0 and depthlist[0]<=treedepth:

        inc=inc+1
        l_leaf=0
        r_leaf=0


        parentclasses = new_data['class'].unique()
        if depthlist[0]<treedepth:
            if len(parentclasses)>1:

                depth=depthlist[0]+1
                maxdf = GetInformationGain(new_data) # Calling the information gain function
                class_value = mode(new_data['class'])[0][0]
                for items in maxdf.values:
                    splitcol = items[0]
                    splitvalue = items[1]

                left_node=new_data.loc[new_data[splitcol]==splitvalue] # ----
                right_node=new_data.loc[new_data[splitcol]!=splitvalue] # ----

                rightnode_values = right_node[splitcol].unique()

                # check for child leaf nodes
                if left_node.shape[0]!=0:
                    dflist.append(left_node)
                    depthlist.append(depth)
                    # print("Left node row size:",left_node.shape[0])
                    left_class = left_node['class'].unique()
                    if len(left_class)==1:
                        # print("Left Child is Leaf")
                        l_leaf = 1

                if right_node.shape[0]!=0:
                    dflist.append(right_node)
                    depthlist.append(depth)
                    # print("Right node row size:",right_node.shape[0])
                    right_class = right_node['class'].unique()
                    if len(right_class)==1:
                        # print("Right Child is Leaf")
                        r_leaf = 1

                # Self Append
                nodenumber=nodenumber+1
                nodecount=nodecount+2
                # print(nodecount,nodenumber)
                thisrow = pd.DataFrame([[int(nodenumber),depthlist[0],splitcol,splitvalue,rightnode_values,nodecount-1,nodecount,'N',class_value]], columns=['nodenumber','depth','attr','leftvalue','rightvalue','left','right','isleaf','class'])
                tree_table = tree_table.append(thisrow,ignore_index=True)

                dflist.pop(0)
                depthlist.pop(0)
                if len(dflist)>0:
                    new_data=pd.DataFrame(dflist[0])
            else:
                # If node is a leaf below gets executed
                nodenumber=nodenumber+1
                classvalue = mode(new_data['class'])[0][0]
                thisrow = pd.DataFrame([[int(nodenumber),depthlist[0],'','','','','','Y',classvalue]], columns=['nodenumber','depth','attr','leftvalue','rightvalue','left','right','isleaf','class'])
                tree_table = tree_table.append(thisrow,ignore_index=True)
                dflist.pop(0)
                depthlist.pop(0)
                if len(dflist)>0:
                    new_data=pd.DataFrame(dflist[0])
        else:
            # If node is a leaf below gets executed
            nodenumber=nodenumber+1
            classvalue = mode(new_data['class'])[0][0]
            thisrow = pd.DataFrame([[int(nodenumber),depthlist[0],'','','','','','Y',classvalue]], columns=['nodenumber','depth','attr','leftvalue','rightvalue','left','right','isleaf','class'])
            tree_table = tree_table.append(thisrow,ignore_index=True)
            dflist.pop(0)
            depthlist.pop(0)
            if len(dflist)>0:
                new_data=pd.DataFrame(dflist[0])

    dflist=None
    depthlist=None
    print("The Decision Tree is:")
    print(tree_table) # This returns the final decision tree in a table format
    return tree_table

# Function to get the split criteria using information gain
def GetInformationGain(mydata):
    i=1
    j=1
    k=1
    minset = 0.00001
    rows = mydata.shape[0]
    cols = mydata.shape[1]
    probclass = mydata['class'].value_counts()
    parententropy = 0
    categorylistentropy = pd.DataFrame()

    for items in probclass:
        entropy = - items/rows * math.log(items/rows,2)
        parententropy = parententropy + entropy

    for i in range(1,cols-2):
        # For Every attribute in the dataset
        curcol = newcolnames[i]
        uniquevalues = mydata[curcol].unique()
        uniquelength = np.size(uniquevalues)

        if uniquelength>1:
            # For every categorical value in the attribute
            for vals in uniquevalues:
                dataset_equal =mydata.loc[mydata[curcol]==vals] # ---
                dataset_nonequal =mydata.loc[mydata[curcol]!=vals] # ---
                class_equal_unique = dataset_equal['class'].unique()
                class_nonequal_unique = dataset_nonequal['class'].unique()
                hs_left = 0
                hs_right = 0
                cat_equalrows = dataset_equal.shape[0]
                cat_nonequalrows = dataset_nonequal.shape[0]


                # For every class label in the split result - for left cat
                for classvals in class_equal_unique:
                    dsleft = dataset_equal.loc[dataset_equal['class']==classvals] # ---
                    dsleft_rows = dsleft.shape[0]
                    classentropy = - dsleft_rows/cat_equalrows * math.log(dsleft_rows/cat_equalrows+minset,2)
                    hs_left = hs_left + classentropy

                # Calculate HS left here
                entropy_left = hs_left * cat_equalrows/rows

                # For every class label in the split result - for right cat
                for classvals in class_nonequal_unique:
                    dsright = dataset_nonequal.loc[dataset_nonequal['class']==classvals] #---
                    dsright_rows = dsright.shape[0]
                    classentropy = - dsright_rows/cat_nonequalrows * math.log(dsright_rows/cat_nonequalrows+minset,2)
                    hs_right = hs_right + classentropy

                # Calculate HS right here
                entropy_right = hs_right * cat_nonequalrows/rows

                # calculate gain here ... attribute,cateory,value
                branchentropy = parententropy - entropy_left - entropy_right
                thisentropy = pd.DataFrame([[curcol,vals,branchentropy]] , columns=['atrr','value','igain']) # ---
                # series1 = pd.Series([curcol,str(vals),branchentropy])
                # series2 = pd.Series([curcol,str(vals),branchentropy])

                categorylistentropy = categorylistentropy.append(thisentropy,ignore_index=True)
    # print("List entropy",categorylistentropy)
    maximum = categorylistentropy['igain'].idxmax()
    maxdf = categorylistentropy[maximum:maximum+1]
    return maxdf

def GetPredition(searchtree,testnode,i):
    nextnode=1.0
    childnode=None
    found = 0
    prediction = True
    while (found==0 and prediction==True):

        ind = nextnode

        searchnode = searchtree.loc[searchtree['nodenumber']==ind].squeeze()
        if searchnode['isleaf']=='Y':
            org_traindata.loc[i,'predclass'] = searchnode['class']
            found=1

        else:
            attribute = searchnode['attr']
            arr=searchnode['rightvalue'].tolist()
            ll = testnode[attribute] # ----
            flag=0
            for x in range(0,len(arr)):
                if ll==arr[x]:
                    flag=1
            if searchnode['leftvalue'] == testnode[attribute]:
                nextnode = int(searchnode['left'])

            elif (ll in arr):
                nextnode = int(searchnode['right'])

            else:
                prediction = False
                org_traindata.loc[i,'predclass'] = 'F'

# apply decision tree builder in the train data
depth = [3]
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
    accuracy = treeaccuracy[0]
    errorlist = treeaccuracy[1]
    error = 1 - accuracy
    print(accuracy)

    # Function Call to build confucation matrix
    confusion_matrix = ConfusionMatrix(org_traindata)
    print("Confusion Matrix for Depth ",dep," is: ")
    print(confusion_matrix)
#     UpdateWeight(inputdata)
#
# def UpdateWeight(inputdata,error):
#
#




