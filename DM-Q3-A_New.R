
# Code to read input file
input_dataset<-read.csv(file.choose(),header = FALSE)

# Prompt the user for the min support and the K value:

n <- readline(prompt="Enter an minimum support: ")
n<-as.numeric(n)
if (is.na(n)){
  prompt("Invalid input from the user for the minimum support")
}
# 
# k <- readline(prompt="Enter the K value (size of frequent itemsets): ")
# k <- as.numeric(k)
# if (is.na(k)){
#   prompt("Invalid input from the user for K value")
# }
# 
# k<-ceiling(k)
# itemset_size=k-1


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
#summary(item_1)

print("1 size before==========================================")
print(nrow(item_1))
#Generate size 1 Freq item sets with min sup
item_1_new<-item_1[which(item_1$Cnt>=sup),]
print("1 size after==========================================")
nrow(item_1_new)

# Converting Table to Data Frame
freq_df<-data.frame(item_1_new)
colnames(freq_df)<-c("Product","Cnt")

print("F_1 Frequent item sets:")
print(freq_df)
#ncol(freq_df)
#nrow(freq_df) # freq_df is the set of all 1-freq itemsets with their support counts

# Generate K-1 Frequent itemsets

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
      
      #if (new_dataset[j,as.numeric(yyy[i,1])]==1 & new_dataset[j,as.numeric(yyy[i,2])]==1){
       # xxx[i,ncol(xxx)]=xxx[i,ncol(xxx)]+1
      }
  }
  return(xxx)
  }



# Implementing F_K-1 Apriori Algorithm -----------------------


for(attr in 1:(ncol(new_dataset)-1)){
  itemset_size=attr
  
  
  # freq_df is the set of all K-1 freq itemsets with their support counts
  my_ds<-k_one(new_dataset)
  #print(my_ds)
  my_ds<-my_ds[which(my_ds$Cnt>=sup),]
  
  #print("=====================================================")
  #print("Frequent F_k-1 Frequent itemsets:")
  #print(my_ds)
  
  if(nrow(my_ds)==0){
    break
  }
  # Now generate K Freq item sets
  
  one_set<-as.matrix(freq_df[-ncol(freq_df)])
  k_minus_set<-as.matrix(my_ds[-ncol(my_ds)])
  
  flag2=1
  myvec<-c()
  
  result_df <- data.frame(matrix(0, ncol = itemset_size+1, nrow = 1))
  
  
  for (rows in 1:nrow(k_minus_set)){
    flag2=0
    
    for(line in 1:nrow(one_set)){
      flag2=0
      
      for (iter in 1:itemset_size){
        if((k_minus_set[rows,iter] == one_set[line,1])){
          flag2=1
        }
      }
      if(flag2==0){
        myvec<-c()
        for (iter in 1:itemset_size){
          myvec<-c(myvec,as.numeric(k_minus_set[rows,iter]))
        }
        myvec<-c(myvec,as.numeric(one_set[line,1]))
        result_df= rbind(result_df,myvec)
        
      }
    }
  }
  
  result_df<-result_df[-1,]
  
  #print(result_df)
  for (i in 1:nrow(result_df)){
    result_df[i,]<-sort(result_df[i,])
  }
  
  result_df<-unique(result_df)
  
  # Find support of the K itemsets candidate sets
  
  result_df$Cnt<-0
  all_result<-result_df
  flag3=0
  mysum=0
  temp<-as.matrix(result_df)
  
  for (i in 1:nrow(temp)){
    
    flag3=0
    mysum=0
    for (j in 1:nrow(new_dataset)){
      
      flag3=0
      
      for (iter in 1:(itemset_size+1)){
        
        if(new_dataset[j,as.numeric(temp[i,iter])]!=1){
          flag3=1
        }
      }
      if(flag3==0){
        mysum=mysum+1
      }
    }
    all_result[i,ncol(all_result)]=all_result[i,ncol(all_result)]+mysum
  }
  print("Size k before =============")
  print(nrow(all_result))
  #print(all_result)
  # Output only Frequent itemsets
  all_result<-all_result[which(all_result$Cnt>=sup),]
  
  print("Frequent F_K item sets using F_1 x F_k-1 apriori algorithm:")
  print(all_result)
  print("Size k after =============")
  print(nrow(all_result))
  print("===================================================================")
  if(nrow(all_result)==0){
    break
  }
}




# Implementing K-1 and K-1 Apriori Algorithm


for(attr in 1:(ncol(new_dataset)-1)){
  itemset_size=attr
  
  ds02<-k_one(new_dataset)
  ds02<-ds02[which(ds02$Cnt>=sup),]
  my_ds02<-ds02[-ncol(ds02)]
  if(nrow(ds02)==0){
    break
  }
  ds03<-k_one(new_dataset)
  ds03<-ds03[which(ds03$Cnt>=sup),]
  my_ds03<-ds03[-ncol(ds03)]
  
  kk_set <- data.frame(matrix(0, ncol = 2*(itemset_size), nrow = 1))
  
  for(r2 in 1:nrow(my_ds02)){
    
    for(r3 in 1:nrow(my_ds03)){
      
      if(as.matrix(my_ds02[r2,1])!=as.matrix(my_ds03[r3,1])){
        kkvec<-c(as.matrix(my_ds02[r2,]),as.matrix(my_ds03[r3,]))
        kk_set<-rbind(kk_set,kkvec)
      }
    }
  }
  
  for (r in 1:nrow(kk_set)){
    kk_set[r,]<-sort(kk_set[r,])
  }
  
  
  kk_set<-unique(kk_set[-1,])
  
  kk_matrix<-as.matrix(kk_set)
  val_vec<-c()
  
  ksets<-data.frame()
  
  for (rows in 1:nrow(kk_set)){
    val_vec<-c(val_vec,kk_matrix[rows,])
    val_vec<-unique(val_vec)
    temp_ds<-t(combn(val_vec,itemset_size+1))
    ksets<-rbind(ksets,data.frame(temp_ds))
  }
  
  uniq_ksets<-unique(ksets)
  
  uniq_ksets$Cnt<-0
  uniq_ksetsresult<-uniq_ksets
  flag4=0
  mysum1=0
  temp1<-as.matrix(uniq_ksetsresult)
  
  for (i in 1:nrow(temp1)){
    
    flag4=0
    
    for (j in 1:nrow(new_dataset)){
      
      flag4=0
      
      for (iter1 in 1:(itemset_size+1)){
        
        if(new_dataset[j,as.numeric(temp1[i,iter1])]!=1){
          flag4=1
        }
      }
      if(flag4==0){
        uniq_ksetsresult[i,ncol(uniq_ksetsresult)]=uniq_ksetsresult[i,ncol(uniq_ksetsresult)]+1
      }
    }
    
  }
  print("Size k before =============")
  print(nrow(uniq_ksetsresult))
  # Output only Frequent itemsets
  #print(uniq_ksetsresult)
  uniq_ksetsresult<-uniq_ksetsresult[which(uniq_ksetsresult$Cnt>=sup),]
  print("==================================================================")
  print("Frequent F_K item sets using F_k-1 x F_k-1 apriori algorithm:")
  print(uniq_ksetsresult)
  print("==================================================================")
  print("Size k after =============")
  print(nrow(uniq_ksetsresult))
  if(nrow(uniq_ksetsresult)==0){
    break
  }
}