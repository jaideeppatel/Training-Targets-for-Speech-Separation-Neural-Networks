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
    allrows = inputdata.shape[0] - 1
    for j in range(0,records):
        rownum = randint(0,allrows)
        trainbag = trainbag.append(inputdata.iloc[rownum])
        # samplelist.append(rownum)
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

    # print('Prediction train data is:',org_traindata['predclass'])


# Main Function - For Boosting
def BuildDesiciontreeBoost(inputdepth):
    # Function Call to Construct the Tree Structure
    tree_table=BuildTree(inputdepth)

    # Build a Temp Search Tree
    searchtree = tree_table

    #Function Call to perform the Prediction on testdata
    for i in range(0,testdata.shape[0]):
        testnode = testdata.iloc[i]
        GetPredition(searchtree,testnode,i)
    
    #Function Call to perform the Prediction on org_traindata
    for i in range(0,org_traindata.shape[0]):
        testnode = org_traindata.iloc[i]
        GetTrainPredition(searchtree,testnode,i)
    
    

# Function to Calculate the Accuracy
def FindAccuracy(testdata):
    count=0
    for i in range(0,testdata.shape[0]):
        if testdata.loc[i,'predclass']==testdata.loc[i,'class']:
            count=count+1
    accuracy = (count/testdata.shape[0])
    # print("Accuracy is: ",accuracy)
    return accuracy
    
# Function to Calculate the Accuracy
def FindWeightParam(org_traindata):
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
    minset = 0.0000001
    rows = mydata.shape[0]
    cols = mydata.shape[1]
    probclass = mydata['class'].value_counts()
    probclassent = mydata['class'].unique()
    parententropy = 0
    categorylistentropy = pd.DataFrame()
    
    # Calculate the parent entropy
    for items in probclassent:
        numer = sum(mydata.loc[mydata['class']==items,'weight'])
        allentropy = sum(mydata['weight'])
        entresult = - (numer/allentropy * math.log(numer/allentropy,2))
        parententropy = parententropy + entresult
    
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
                allent_left = sum(dataset_equal['weight'])
                allent_right = sum(dataset_nonequal['weight'])
                
                # For every class label in the split result - for left cat
                for classvals in class_equal_unique:
                    dsleft = dataset_equal.loc[dataset_equal['class']==classvals] # ---
                    left_ent = sum(dsleft.loc[dsleft['class']==classvals,'weight'])
                    dsleft_rows = dsleft.shape[0]
                    classentropy = - left_ent/allent_left * math.log(left_ent/allent_left+minset,2)
                    hs_left = hs_left + classentropy
                
                # Calculate HS left here
                entropy_left = hs_left * cat_equalrows/rows
                
                # For every class label in the split result - for right cat
                for classvals in class_nonequal_unique:
                    dsright = dataset_nonequal.loc[dataset_nonequal['class']==classvals] #---
                    right_ent = sum(dsright.loc[dsright['class']==classvals,'weight'])
                    dsright_rows = dsright.shape[0]
                    classentropy = - right_ent/allent_right * math.log(right_ent/allent_right+minset,2)
                    hs_right = hs_right + classentropy
                
                # Calculate HS right here
                entropy_right = hs_right * cat_nonequalrows/rows
                
                # calculate gain here ... attribute,cateory,value
                branchentropy = parententropy - entropy_left - entropy_right
                thisentropy = pd.DataFrame([[curcol,vals,branchentropy]] , columns=['atrr','value','igain']) # ---
                
    
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
            testdata.loc[i,'predclass'] = searchnode['class']
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
                testdata.loc[i,'predclass'] = 'F'

def GetTrainPredition(searchtree,testnode,i):
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




def UpdateWeight(error,errorlist):
    # here we update the weights of each training record based on the results obtained by testing ....
    trainrecords = inputdata.shape[0]
    print('Model error is:',error)
    alpha = (1/2) * np.log((1-error)/error)
    pos_e = np.exp(alpha)
    pos_p = np.exp(-alpha)
    
    j=0
    for j in range(0,trainrecords):
        if (errorlist[j]==0):
            inputdata.loc[j,'weight'] = inputdata.loc[j,'weight'] * pos_e
        else:
            inputdata.loc[j,'weight'] = inputdata.loc[j,'weight'] * pos_p
    
    return alpha


# apply decision tree builder in the train data
depth = 3
noofbags = 1
allpredictions=pd.DataFrame()
# allpredictions['class'] = testdata['class']
accuracylist = []
# samplelist = []
preparedata(inputdata,testdata,org_traindata)
testrecords=testdata.shape[0]
newcolnames = inputdata.columns

repeat=0
modelweight = []
for repeat in range(0,noofbags):
    # Create Bag data for the current iteration...
    train = createBags(inputdata)
    
    filename = 'trainbag'+str(repeat)+'.csv'
    train.to_csv(filename)
    BuildDesiciontreeBoost(depth)
    
    # Section to test the testdata file
    allpredictions[repeat] = testdata['predclass']
    testdata['predclass'] = 'NA'
    testdata.is_copy = False
    
    # Find accuracy of the prediction for the org_traindata
    train_treeaccuracy = FindWeightParam(org_traindata)
    train_accuracy = train_treeaccuracy[0]
    errorlist = train_treeaccuracy[1]
    error = 1 - train_accuracy
    
    org_traindata['predclass'] = 'NA'
    org_traindata.is_copy = False
    
    # Function call to update the weights of the train data records
    modelwt = UpdateWeight(error,errorlist)
    modelweight.append(modelwt)
    fname = 'weight'+str(repeat)+'.csv'
    inputdata.to_csv(fname)

# print(allpredictions)
allpredictions.to_csv('allprediction.csv')
# sum_class = np.sum(allpredictions,axis=1)

# Find final prediction using the sign function...
allpredictions = allpredictions.replace(0,-1)
predicted_class = []
x = 0
for x in range(0,testrecords):
    series = np.array(allpredictions.loc[x])
    wtseries = np.array(modelweight)
    value = np.sign(sum(series*wtseries))
    if value<=0:
        predicted_class.append(0)
    else:
        predicted_class.append(1)
testdata['predclass'] = predicted_class
testdata.to_csv('testresult.csv')

# Find accuracy for the prediction on the testdata
# print(predicted_class)
# Function Call to find accuracy
treeaccuracy = FindAccuracy(testdata)
print('Accuracy of the model on test set is:',treeaccuracy)
# Function Call to build confusion matrix
confusion_matrix = ConfusionMatrix(testdata)
print("Confusion Matrix for Depth ",depth," is: ")
print(confusion_matrix)
