
# Code to read input file
input_dataset<-read.csv(file.choose(),header = FALSE)

# Count the number of items every user buys for general information to set k value
matrix_counts<-apply(input_dataset, 1, function(c)sum(c!=0))

# Remove all zero rows 
new_array <- apply(input_dataset, 1, function(x){sum(x)!=0})
new_dataset<-input_dataset[new_array==TRUE,]



# Implementing F_k and F_K-1 Apriori Algorithm

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
summary(item_1)

#Generate size 1 Freq item sets with min sup = 5
sup=3
item_1_new<-item_1[which(item_1$Cnt>=sup),]
nrow(item_1_new)

# Converting Table to Data Frame
freq_df<-data.frame(item_1_new)
colnames(freq_df)<-c("Product","Cnt")
print(freq_df)
#ncol(freq_df)
#nrow(freq_df) # freq_df is the set of all 1-freq itemsets with their support counts

# Generate K-1 Frequent itemsets with the support count values
itemset_size=3 # ====== Will generate itemsets of size iremset_size

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

# # freq_df is the set of all F_1 freq itemsets with their support counts
# my_ds<-k_one(new_dataset)
# print(my_ds)
# 
# level1_items<-item_1 # All F_1 items sets
# level1_items$closed<-1
# level1_items<-as.matrix(level1_items)
# 
# level2_items<-as.matrix(my_ds) # All F_K+1 item sets 
# mlevel<-level2_items[,-ncol(level2_items)]
# 
# # Generate closed freq itemset for at F_1 level comparing items at F_2 level
# 
# lflag=1
# 
# for (l1r in i:nrow(level1_items)){
#  l1item=as.numeric(level1_items[l1r,1])
#  l1support=as.numeric(level1_items[l1r,'Cnt'])
#  lflag=1
#  myc<-c()
#  for(j in 1:ncol(mlevel)){
#    cc1<-which(mlevel[,j]==l1item)
#    myc<-c(myc,cc1)
#  }
#  for(k in myc){
#    if(as.numeric(level2_items[k,'Cnt'])>=l1support){
#      lflag=0
#    }
#  }
#  if(lflag==0){
#    level1_items[l1r,'closed']=0
#  }
# }
# 
# print(level1_items) # Closed item sets will have closed = 1 in the data frame


# Generate close itemsets from level 1 and above by implementing K-1 X K-1 combinations 

closeflag=0
for (allcols in 1:ncol(new_dataset)){
  # Generate current level Freg itemets
  itemset_size=allcols
  c_l<-k_one(new_dataset)# current level itemssets
  c_level<-c_l[which(c_l$Cnt>=sup),]
  c_levelresult<-c_level
  c_levelresult$Closed<-0
  print(c_levelresult)
  # Generate next level itemsets
  itemset_size=allcols+1
  n_level<-k_one(new_dataset) # next level itemsets
  
  # evaluate the closeed itemsets for itemsets at current level
  c_levelprod<-c_level[-ncol(c_level)]
  n_levelprod<-n_level[-ncol(n_level)]
  
  for (crows in 1:nrow(c_levelprod)){
    closeflag=0
    carray<-as.vector(as.matrix(c_levelprod[crows,]))
    
    for(nrows in 1:nrow(n_levelprod)){
      narray<-as.vector(as.martix(n_levelprod[nrows,]))
      if(carray%in%narray){
        if(n_level$Cnt>=c_level$Cnt){
          closeflag=1
        }
      }
    }
    if(closeflag==1){
      c_levelresult[crows,'Closed']=1
    }
  }
  print(c_levelresult)
}



#-------------------------------------------------------------------
# Implementing K-1 and K-1 Apriori Algorithm


ds02<-k_one(new_dataset)
ds02<-ds02[which(ds02$Cnt>=sup),]
my_ds02<-ds02[-ncol(ds02)]

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

# Output only Frequent itemsets
print(uniq_ksetsresult)














































my_ds<-my_ds[which(my_ds$Cnt>=sup),]
print(my_ds)


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

print(all_result)
# Output only Frequent itemsets
all_result<-all_result[which(all_result$Cnt>=sup),]
print(all_result)

# Implementing K-1 and K-1 Apriori Algorithm


ds02<-k_one(new_dataset)
ds02<-ds02[which(ds02$Cnt>=sup),]
my_ds02<-ds02[-ncol(ds02)]

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

# Output only Frequent itemsets
print(uniq_ksetsresult)
uniq_ksetsresult<-uniq_ksetsresult[which(uniq_ksetsresult$Cnt>=sup),]
print(uniq_ksetsresult)






