import pandas as pd
import numpy as np
import math
from scipy.stats import mode
import os
from random import randint
import sys, os

# def load_data(datapath):
#     pass

def learn_bagged(tdepth, nummodels, datapath):
    # apply decision tree builder in the train data
    depth = tdepth
    noofbags = nummodels
    allpredictions=pd.DataFrame()

    # Create Bag data for the current iteration...
    for repeat in range(0,noofbags):
        # Create Bag data for the current iteration...
        train = createBags(inputdata)
        filename = 'trainbag'+str(repeat)+'.csv'
        train.to_csv(filename)
        BuildDesiciontree(depth,train)
        allpredictions[repeat] = testdata['predclass']
        testdata['predclass'] = 'NA'
        testdata.is_copy = False

    # Capture the Prediction result
    print(allpredictions)
    allpredictions.to_csv('allprediction.csv')
    sum_class = np.sum(allpredictions,axis=1)
    predicted_class = []
    x = 0
    for x in range(0,testrecords):
        if sum_class[x]<=noofbags/2:
            predicted_class.append(0)
        else:
            predicted_class.append(1)

    print(predicted_class)
    testdata['predclass'] = predicted_class
    testdata.to_csv('testresult.csv')
    # Function Call to find accuracy
    treeaccuracy = FindAccuracy(testdata)

    # Function Call to build confucation matrix
    confusion_matrix = ConfusionMatrix(testdata)
    print("Confusion Matrix for Depth ",depth," is: ")
    print(confusion_matrix)

    return treeaccuracy

def createBags(inputdata):
    trainbag = pd.DataFrame()
    j = 0
    records=int(inputdata.shape[0] * 100 / 100)
    index = inputdata.shape[0]-1
    for j in range(0,records):
        rownum = randint(0,index)
        trainbag = trainbag.append(inputdata.iloc[rownum])
    return trainbag


# Main Function
def BuildDesiciontree(inputdepth,train):
    # Function Call to Construct the Tree Structure
    tree_table=BuildTree(inputdepth,train)

    # Build a Temp Search Tree
    searchtree = tree_table

    #Function Call to perform the Prediction
    for i in range(0,testdata.shape[0]):
        testnode = testdata.iloc[i]
        GetPredition(searchtree,testnode,i)

# Function to Calculate the Accuracy
def FindAccuracy(testdata):
    count=0
    for i in range(0,testdata.shape[0]):
        if testdata.loc[i,'predclass']==testdata.loc[i,'class']:
            count=count+1
    accuracy = (count/testdata.shape[0])
    # print("Accuracy is: ",accuracy)
    return accuracy

# Function to Calculate the Confusion Matrix
def ConfusionMatrix(testdata):
    # print("Tree Data Read",testdata)
    con_mat = pd.DataFrame(columns=['Actual_Class','Predicted_Class=0','Predicted_Class=1'])
    data00 = testdata.loc[(testdata['class']==0) & (testdata['predclass']==0)].shape[0]
    data01 = testdata.loc[(testdata['class']==0) & (testdata['predclass']==1)].shape[0]
    data10 = testdata.loc[(testdata['class']==1) & (testdata['predclass']==0)].shape[0]
    data11 = testdata.loc[(testdata['class']==1) & (testdata['predclass']==1)].shape[0]
    class0 = pd.DataFrame([['Actual:0',int(data00),int(data01)]],columns=['Actual_Class','Predicted_Class=0','Predicted_Class=1'])
    class1 = pd.DataFrame([['Actual:1',int(data10),int(data11)]],columns=['Actual_Class','Predicted_Class=0','Predicted_Class=1'])
    con_mat = con_mat.append(class0,ignore_index=True)
    con_mat = con_mat.append(class1,ignore_index=True)
    return  con_mat


# Function to build the decision tree for the given train data set (Internally the function calls the Information gain function above...)
def BuildTree(inputdepth,train):

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

    for i in range(1,cols-1):
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




def learn_boosted(tdepth, numtrees, datapath):
    pass;


if __name__ == "__main__":
    # The arguments to your file will be of the following form:
    # <ensemble_type> <tree_depth> <num_bags/trees> <train_set_path>
    # Ex. bag 3 10 "train/agaricus-lepiota.data"
    # Ex. boost 1 10

    # Get the ensemble type
    # entype = sys.argv[1]
    entype = 'bag'
    # Get the depth of the trees
    # tdepth = int(sys.argv[2])
    tdepth = 3
    # Get the number of bags or trees
    # nummodels = int(sys.argv[3])
    nummodels = 1
    # Get the location of the data set
    # datapath = sys.argv[4]
    datapath = ''

    # Code to read the training input file
    inputdata = pd.read_csv(datapath+'train.csv')
    trainrecords=inputdata.shape[0]
    inputdata=inputdata.rename(columns = {'poisonous':'class'})
    newcolnames = inputdata.columns

    # Code to read the test input file
    testdata = pd.read_csv(datapath+'test.csv')
    testrecords=testdata.shape[0]
    testdata=testdata.rename(columns = {'poisonous':'class'})
    testdata['predclass'] = 'NA'
    testdata.is_copy = False
    # print(testdata)

    # Check which type of ensemble is to be learned
    if entype == "bag":
        # Learned the bagged decision tree ensemble
        accuracy_bagEnsemble = learn_bagged(tdepth, nummodels, datapath)
        print("Accuracy of the Ensemble is:",accuracy_bagEnsemble)
    else:
        # Learned the boosted decision tree ensemble
        learn_boosted(tdepth, nummodels, datapath)


