# 4th New code with corss validation

library(cvTools) # Using the folds method to split the data randomly into 10 folds
# Code to read input file
input_dataset<-read.csv(file.choose(),header = FALSE)
colnames(input_dataset)<- c("C1","C2","C3","C4","Class")
cnames=colnames(input_dataset)
new_dataset<-input_dataset # Create a temp dataset
colnames(new_dataset)<-cnames
allaccuracies=NULL
#print(new_dataset)

folds=10

myunif <- runif(nrow(input_dataset))
unif_dataset<- input_dataset[order(myunif),]
nrow(unif_dataset)
allfolds <- cvFolds(NROW(unif_dataset), K=folds)
unif_dataset$predict <- rep(0,nrow(unif_dataset))
accuracy_sum=0

#Gini function definition

get_gini<- function(new_dataset){
  allgini<-NULL
  i=1
  j=1
  k=1
  for (i in 1:(ncol(new_dataset)-1)){
    c_unqiuevalues=unique(new_dataset[,i])
    if(length(c_unqiuevalues)>1){
      for (j in 1:(length(c_unqiuevalues)-1)){
        mean1=(c_unqiuevalues[j]+c_unqiuevalues[j+1])/2
        dataset_left=new_dataset[new_dataset[,i]<mean1,]
        dataset_right=new_dataset[new_dataset[,i]>mean1,]
        for (k in 1:length(total_classes)){
          lvalclass=nrow(dataset_left[dataset_left$Class==total_classes[k],])
          probval=(lvalclass/nrow(dataset_left))^2
          probsum=probsum+probval
        }
        probsuml=1-(probsum)
        probsum=0
        for (k in 1:length(total_classes)){
          rvalclass=nrow(dataset_right[dataset_right$Class==total_classes[k],])
          probval=(rvalclass/nrow(dataset_right))^2
          probsum=probsum+probval
        }
        probsumr=1-(probsum)
        probsum=0
        colprobvall=(nrow(dataset_left)/nrow(new_dataset))*(probsuml)
        colprobvalr=(nrow(dataset_right)/nrow(new_dataset))*(probsumr)
        colprobvalsum=colprobvall+colprobvalr
        probfullcol= rbind(probfullcol,data.frame(mean = mean1,Gini = colprobvalsum))
        probsuml=0
        probsumr=0
      }
      probfullcol_min=probfullcol[probfullcol$Gini == min(probfullcol$Gini),]
      allgini=rbind(allgini,data.frame(column.name = i,mean= probfullcol_min$mean,Gini=probfullcol_min$Gini))
      probfullcol=data.frame(mean = double(),Gini = double())
    }
    
  }
  return(allgini)
}

# Get information gain function definition

get_Informationgain<- function(new_dataset){
  allgini<-NULL
  i=1
  j=1
  k=1
  allmin=0.0001
  for (i in 1:(ncol(new_dataset)-1)){
    c_unqiuevalues=unique(new_dataset[,i])
    if(length(c_unqiuevalues)>1){
      for (j in 1:(length(c_unqiuevalues)-1)){
        mean1=(c_unqiuevalues[j]+c_unqiuevalues[j+1])/2
        dataset_left=new_dataset[new_dataset[,i]<mean1,]
        dataset_right=new_dataset[new_dataset[,i]>mean1,]
        for (k in 1:length(total_classes)){
          lvalclass=nrow(dataset_left[dataset_left$Class==total_classes[k],])
          if (lvalclass==0){
            probval=0
            probsum=probsum+probval 
          }
          else{
            probval=-(lvalclass/nrow(dataset_left))*log((lvalclass/nrow(dataset_left)+allmin))
            probsum=probsum+probval  
          }
          
        }
        probsuml=probsum
        probsum=0
        for (k in 1:length(total_classes)){
          rvalclass=nrow(dataset_right[dataset_right$Class==total_classes[k],])
          if(rvalclass==0){
            probval=0
            probsum=probsum+probval
          }
          else{
            probval=(rvalclass/nrow(dataset_right))*log((lvalclass/nrow(dataset_left)+allmin))
            probsum=probsum+probval
          }
        }
        probsumr=probsum
        probsum=0
        colprobvall=(probsuml)*(nrow(dataset_left)/nrow(new_dataset))
        colprobvalr=(probsumr)*(nrow(dataset_right)/nrow(new_dataset))
        colprobvalsum=colprobvall+colprobvalr
        probfullcol= rbind(probfullcol,data.frame(mean = mean1,Gini = colprobvalsum))
        probsuml=0
        probsumr=0
      }
      probfullcol_min=probfullcol[probfullcol$Gini == min(probfullcol$Gini),]
      allgini=rbind(allgini,data.frame(column.name = i,mean= probfullcol_min$mean,Gini=probfullcol_min$Gini))
      probfullcol=data.frame(mean = double(),Gini = double())
    }
    
  }
  return(allgini)
}


# Get Prediction Function

get_prediction<-function(subjectnode){
  predicted_value=NULL
  nextnode=1
  vwvalue=unique(search_tree$visited)
  vwvaluecount=length(unique(search_tree$visited))
  searchnode<-NULL
  while(found==0 | (vwvaluecount==1 && vwvalue==1)){
    searchnode<-NULL
    searchnode=search_tree[nextnode,]
    evalclass=searchnode$classval
    if(is.na(evalclass)==TRUE){
      
      searchnode$visited=1
      evalcol=searchnode$colval
      evalmean=searchnode$meanval
      childleft=searchnode$childleft
      childright=searchnode$childright
      
      if(subjectnode[1,evalcol]<evalmean){
        nextnode=childleft
      }
      else{
        nextnode=childright
      }
      
    }
    else{
      predicted_value=searchnode$classval
      prediction<-rbind(prediction,data.frame(predval=predicted_value))
      vwvalue=NULL
      vwvaluecount=NULL
      found=1
      break
    }
    
  }
  if(is.null(predicted_value)==TRUE){
    predicted_value=NA
    #print(predicted_value)
    prediction<-rbind(prediction,data.frame(predval=predicted_value))
    vwvalue=NULL
    vwvaluecount=NULL
    found=0
    return(prediction)
  }
  else{
    return(prediction)
  }
}

for(i in 1:folds){
  train_dataset <- unif_dataset[allfolds$subsets[allfolds$which != i], ] #Set the training set
  new_dataset<-train_dataset[,-ncol(train_dataset)]
  validation_dataset <- unif_dataset[allfolds$subsets[allfolds$which == i], ] #Set the validation
  names(train_dataset)<- cnames # Data col names
  names(validation_dataset)<- cnames # Data col names
  nrow(train_dataset)
  nrow(validation_dataset)
  
  # variables to store the split results and new data sets
  new_dataset_right<-NULL
  new_dataset_left<-NULL
  nodecount=0
  nodenumber=0
  classvalue_right=NA
  classvalue_left=NA
  all_dflist<-list()
  node_list<-data.frame(nodenumber=double(),meanval=double(),colval=double(),childleft=double(),childright=double(),classval=double())
  
  # Initialize gini values
  gini_values<-numeric(0)
  mean_values<-numeric(0)
  gini_min=0
  gini_mincol=0
  all_values<-list(0)
  giniclass_sum=0
  temp_dataset<-data.frame()
  coldataset<-data.frame()
  dataset_left<-data.frame()
  dataset_left<-data.frame()
  probsum=0
  probsumr=0
  probsuml=0
  colprobvalsum=0
  probcollist=NULL
  probfullcol=data.frame(mean = double(),Gini = double())
  allgini=data.frame(column.name=double(),mean = double(),Gini = double())
  nodesplitleft=data.frame(class1=double(),class2=double())
  nodesplitright=data.frame(class1=double(),class2=double())

  lflag=0
  rflag=0
  inc=0
  mylist=NULL
  # Add the first data set as the root node
  nodecount=nodecount+1
  all_dflist[[length(all_dflist)+1]]<-new_dataset
  total_classes=unique(new_dataset$Class)
  
  for (i in 1:length(total_classes)){
    temp_dataset<-new_dataset[new_dataset$Class==total_classes[i],]
    ginival=(nrow(temp_dataset)/nrow(new_dataset))^2
    giniclass_sum=giniclass_sum+ginival
  }
  classgini=1-(giniclass_sum)
  
  while(length(all_dflist)!=0){
    new_dataset_left<-NULL
    new_dataset_right<-NULL
    inc=inc+1
    cleft=NULL
    cright=NULL
    # Function call to capture gini function return values
    if(length(unique(new_dataset$Class))>1){
      
      # Make the gini and information gain function calls
      #allgini_datset<-get_gini(new_dataset)
      allgini_datset<-get_Informationgain(new_dataset)
      if(is.null(allgini_datset)==TRUE){
        #nodenumber=nodenumber+1
        #cval=new_dataset[1,"Class"]
        #node_list<-rbind(node_list,data.frame(nodenumber=nodenumber,meanval=NA,colval=NA,childleft=NA,childright=NA,classval=cval))
        #all_dflist[[length(all_dflist)+1]]<-new_dataset
        all_dflist[1]<-NULL
        #print(node_list)
      }
      else{
        allgini_datset_min=allgini_datset[allgini_datset$Gini == min(allgini_datset$Gini),]
        split_col=allgini_datset_min[1,1]
        split_mean=allgini_datset_min[1,2]
        split_gini=allgini_datset_min[1,3]
        
        # Rows to left and rows to right
        allrows_left=which(new_dataset[,split_col]<=split_mean)
        allrows_right=which(new_dataset[,split_col]>split_mean)
        
        
        # Create the split data sets
        for (i in allrows_left){
          new_dataset_left<-rbind(new_dataset_left,new_dataset[i,])
        }
        
        for (j in allrows_right){
          new_dataset_right<-rbind(new_dataset_right,new_dataset[j,])
        }
        
        
        # Append all the new data sets and increment node count value
        if(is.null(new_dataset_left)!=TRUE)
        {
          all_dflist[[length(all_dflist)+1]]<-new_dataset_left
          for (ccleft in 1:length(total_classes)){
            ccvalueleft=nrow(new_dataset_left[new_dataset_left$Class==total_classes[ccleft],])
            cleft=c(cleft,ccvalueleft)
          }
          # Check for class values to be same
          if (length(unique(new_dataset_left$Class))==1){
            #print("left leaf found")
            lflag=1
            classvalue_left=unique(new_dataset_left$Class)
          }
        }
        if(is.null(new_dataset_right)!=TRUE)
        {
          all_dflist[[length(all_dflist)+1]]<-new_dataset_right
          for (ccright in 1:length(total_classes)){
            ccvalueright=nrow(new_dataset_right[new_dataset_right$Class==total_classes[ccright],])
            cright=c(cright,ccvalueright)
          }
          # Check for class values to be same
          if (length(unique(new_dataset_right$Class))==1){
            #print("right leaf found")
            rflag=1
            classvalue_right=unique(new_dataset_right$Class)
          }
        }
        # Create the node_gini values list for every node
        if(inc==1){
          nodenumber=nodenumber+1
          nodecount=nodecount+2
          node_list<-rbind(node_list,data.frame(nodenumber=nodenumber,meanval=split_mean,colval=split_col,childleft=nodecount-1,childright=nodecount,classval=NA))
          nodesplitleft<-rbind(nodesplitleft,data.frame(class1=cleft[1],class2=cleft[2]))
          nodesplitright<-rbind(nodesplitright,data.frame(class1=cright[1],class2=cright[2]))
        }
        if (lflag==0 && rflag==0 && inc!=1){
          nodenumber=nodenumber+1
          nodecount=nodecount+2
          node_list<-rbind(node_list,data.frame(nodenumber=nodenumber,meanval=split_mean,colval=split_col,childleft=nodecount-1,childright=nodecount,classval=NA))
          nodesplitleft<-rbind(nodesplitleft,data.frame(class1=cleft[1],class2=cleft[2]))
          nodesplitright<-rbind(nodesplitright,data.frame(class1=cright[1],class2=cright[2]))
        }
        if(lflag==1){
          nodenumber=nodenumber+1
          node_list<-rbind(node_list,data.frame(nodenumber=nodenumber,meanval=split_mean,colval=split_col,childleft=NA,childright=NA,classval=classvalue_left))
          nodesplitleft<-rbind(nodesplitleft,data.frame(class1=cleft[1],class2=cleft[2]))
          nodesplitright<-rbind(nodesplitright,data.frame(class1=cright[1],class2=cright[2]))
        }
        if(rflag==1){
          nodenumber=nodenumber+1
          node_list<-rbind(node_list,data.frame(nodenumber=nodenumber,meanval=split_mean,colval=split_col,childleft=NA,childright=NA,classval=classvalue_right))
          nodesplitleft<-rbind(nodesplitleft,data.frame(class1=cleft[1],class2=cleft[2]))
          nodesplitright<-rbind(nodesplitright,data.frame(class1=cright[1],class2=cright[2]))
        }
        #print(node_list)
        #print(length(all_dflist))
        lflag=0
        rflag=0
        all_dflist[1]<-NULL
        new_dataset=as.data.frame(all_dflist[1])
        length(all_dflist) 
      }
    }
    else{
      all_dflist[1]<-NULL
      #print(length(all_dflist))
      new_dataset=as.data.frame(all_dflist[1])
      length(all_dflist)
    }
  }
  #print("Final Decision tree is:")
  #print(node_list)
  #print(c("Nodesplit left",nodesplitleft))
  #print(c("Nodesplit right",nodesplitright))
  #print(length(unique(node_list$nodenumber)))
  all_dflist<-NULL
  # Make the node list NULL at the end of the cross validation loop
  
  # Prediction code begins here
  
  test_dataset=data.frame()
  test_dataset<-validation_dataset
  test_dataset<-test_dataset[,-ncol(test_dataset)] # Create a temp dataset
  temp_dataset<-test_dataset
  temp_dataset$Class<-NULL
  #print(temp_dataset)
  
  # Initializing values for prediction
  search_tree<-node_list
  search_tree[,ncol(search_tree)+1]<-0
  colnames(search_tree)[ncol(search_tree)]<-"visited"
  nextnode=1
  evalcol=0
  subjectnode<-data.frame()
  prediction<-data.frame(predval=double())
  vwvaluecount=NULL
  vwvalue=NULL
  found=0
  predicted_value=NULL
  searchnode=data.frame(nodenumber=double(),meanval=double(),colval=double(),childleft=double(),childright=double(),classval=double())
  predictiondf<-data.frame()
  
  
  for(i in 1:nrow(temp_dataset)){
    subjectnode=temp_dataset[i,]
    prediction_result<-get_prediction(subjectnode)
    predictiondf<-rbind(predictiondf,prediction_result)
  }
  sum=0
  i=1
  for (i in 1:nrow(predictiondf)){
    left=as.character(predictiondf[i,1])
    right=as.character(test_dataset[i,ncol(test_dataset)])
    #print(i)
    if(left==right){
      sum=sum+1
    }
  }
  accuracy=sum/nrow(test_dataset)
  allaccuracies=c(allaccuracies,accuracy)
  print(c("Accuracy is: ",accuracy))
  predictiondf<-NULL
  }
print(mean(allaccuracies))