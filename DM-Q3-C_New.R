

# Code to read input file
input_dataset<-read.csv(file.choose(),header = FALSE)

# Prompt the user for the min support and the K value:

n <- readline(prompt="Enter an minimum support: ")
n<-as.numeric(n)
if (is.na(n)){
  prompt("Invalid input from the user for the minimum support")
}



# Count the number of items every user buys for general information to set k value
matrix_counts<-apply(input_dataset, 1, function(c)sum(c!=0))

# Remove all zero rows 
new_array <- apply(input_dataset, 1, function(x){sum(x)!=0})
new_dataset<-input_dataset[new_array==TRUE,]

sup=ceiling(nrow(new_dataset)*n)
print("Minimum Support is:")
sup


# Generate Size 1 itemsets
mylist<-c()

for (i in 1:nrow(new_dataset)){
  for (j in 1:ncol(new_dataset)){
    if(new_dataset[i,j]==1){
      val=j
      mylist=c(mylist,val)
    }
  }
}

# Conveting to Table
mylist_table<-table(mylist)
#print(mylist_table)

# converting contingency table to Data Frame
item_1<-as.data.frame(mylist_table)
colnames(item_1)<-c("Product","Cnt")

xxx<-new_dataset

k_one<-function(new_dataset){
  xxx<-data.frame(combn(as.vector(item_1$Product),itemset_size))
  xxx<-data.frame(t(xxx))
  c_values<-c()
  
  # Addind a Count col to the Data Frame
  xxx$Cnt <- 0
  
  yyy<- as.matrix(xxx)
  flag=0
  
  for (i in 1:nrow(yyy)){
    
    flag=0
    
    for (j in 1:nrow(new_dataset)){
      
      flag=0
      
      for (iter in 1:itemset_size){
        
        if(new_dataset[j,as.numeric(yyy[i,iter])]!=1){
          flag=1
        }
      }
      if(flag==0){
        xxx[i,ncol(xxx)]=xxx[i,ncol(xxx)]+1
      }
      
    }
  }
  return(xxx)
}


# Generate close itemsets from level 1 and above by implementing K-1 X K-1 combinations 

closeflag=0
loopiter=ncol(new_dataset)-1
for (allcols in 1:loopiter){
  # Generate current level Freq itemets
  itemset_size=allcols
  c_l<-k_one(new_dataset)# current level itemssets
  c_level<-c_l[which(c_l$Cnt>=sup),]
  if (nrow(c_level)>0)
  {
    c_levelresult<-c_level
    c_levelresult$Closed<-0
    
    # Generate next level itemsets
    itemset_size=allcols+1
    n_level<-k_one(new_dataset) # next level itemsets
    
    # evaluate the closeed itemsets for itemsets at current level
    c_levelprod<-c_level[-ncol(c_level)]
    n_levelprod<-n_level[-ncol(n_level)]
    
    for (crows in 1:nrow(c_levelprod)){
      carray<-c()
      closeflag=0
      carray<-as.vector(as.matrix(c_levelprod[crows,]))
      for(nrows in 1:nrow(n_levelprod)){
        narray<-c()
        narray<-as.vector(as.matrix(n_levelprod[nrows,]))
        if(all(carray%in%narray)){
          if(n_level[nrows,'Cnt']>=c_level[crows,'Cnt']){
            #print("Flag Flipped at")
            closeflag=1
            break
          }
        }
      }
      if(closeflag==1){
        c_levelresult[crows,'Closed']=1
      }
    }
    print(c_levelresult[c_levelresult[,'Closed']=='0',])
  }else{
    print("No closed item sets at this level")
  }
}


# Generate Maximal Frequent itemsets from level 1 and above by implementing K-1 X K-1 combinations 

closeflag_m=0
loopiter=ncol(new_dataset)-1
for (allcols in 1:loopiter){
  # Generate current level Freg itemets
  itemset_size=allcols
  c_l<-k_one(new_dataset)# current level itemssets
  c_level<-c_l[which(c_l$Cnt>=sup),]
  if (nrow(c_level)>0)
  {
    c_levelresult_m<-c_level
    c_levelresult_m$Maximal<-1
    
    # Generate next level itemsets
    itemset_size=allcols+1
    n_level<-k_one(new_dataset) # next level itemsets
    
    # evaluate the closeed itemsets for itemsets at current level
    c_levelprod<-c_level[-ncol(c_level)]
    n_levelprod<-n_level[-ncol(n_level)]
    
    for (crows in 1:nrow(c_levelprod)){
      carray<-c()
      closeflag_m=0
      carray<-as.vector(as.matrix(c_levelprod[crows,]))
      for(nrows in 1:nrow(n_levelprod)){
        narray<-c()
        narray<-as.vector(as.matrix(n_levelprod[nrows,]))
        if(all(carray%in%narray)){
          if(n_level[nrows,'Cnt']>=sup){
            #print("Flag Flipped")
            closeflag_m=1
            break
          }
        }
      }
      if(closeflag_m==1){
        c_levelresult_m[crows,'Maximal']=0
      }
    }
    print(c_levelresult_m[c_levelresult_m[,'Maximal']=='1',])
  }else{
    print("No Maximal itemsets at this level")
  }
}
