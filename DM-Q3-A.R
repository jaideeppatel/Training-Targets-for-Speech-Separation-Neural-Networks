
library(dplyr)

# Code to read input file
input_dataset<-read.csv(file.choose(),header = FALSE)

# Count the number of items every user buys for general information to set k value
matrix_counts<-apply(input_dataset, 1, function(c)sum(c!=0))

# Remove all zero rows 
new_dataset <- input_dataset 

for (i in 1:nrow(input_dataset)){
  y =(sum(input_dataset[i,]))
  if(y==0){
    new_dataset<-new_dataset[-i,]
  }
}
nrow(new_dataset)

# Generate Size 1 itemsets
mylist<-c()
head(new_dataset)

for (i in 1:nrow(new_dataset)){
  for (j in 1:ncol(new_dataset)){
    if(new_dataset[i,j]==1){
      val=j
      mylist=c(mylist,val)
    }
  }
}
print(mylist)
# Conveting to Table
mylist_table<-table(mylist)

# converting contingency table to Data Frame
item_1<-as.data.frame(mylist_table)
colnames(item_1)<-c("Product","Cnt")


#Generate size 1 Freq item sets with min sup = 3
item_1_new<-item_1[which(item_1$Cnt>=3),]
nrow(item_1_new)

# Converting Table to Data Frame
freq_df<-data.frame(item_1_new)
colnames(freq_df)<-c("Product","Cnt")
ncol(freq_df)
nrow(freq_df)

# Generate K-1 Frequent itemsets
# When K = 2
itemset_size=2

xxx<-new_dataset

k_one<-function(new_dataset){
  xxx<-data.frame(combn(as.vector(item_1$Product),itemset_size))
  xxx<-data.frame(t(xxx))
  c_values<-c()
  
  # Addind a Count col to the Data Frame
  xxx$Cnt <- 0
  
  yyy<- as.matrix(xxx)
  
  for (i in 1:nrow(yyy)){
    
    for (j in 1:nrow(new_dataset)){
      if (new_dataset[j,as.numeric(yyy[i,1])]==1 & new_dataset[j,as.numeric(yyy[i,2])]==1){
        xxx[i,ncol(xxx)]=xxx[i,ncol(xxx)]+1
      }
    }
  }
  
  # Frequent 1 Itemsets
  xxx<-xxx[which(xxx$Cnt>=3),]
  return(xxx)
}

my_ds<-k_one(new_dataset)

# Now we need to eliminate non sup items from K-1 set
my_ds_new<-as.matrix(my_ds[-ncol(my_ds)])
all_2items<-c()

for (i in 1:nrow(my_ds_new)){
  for (j in 1:ncol(my_ds_new)){
    val=as.numeric(my_ds_new[i,j])
    all_2items<-c(all_2items,val)
  }
}

all_2items_table<-table(all_2items)

# converting contingency table to Data Frame
all_2items_df<-as.data.frame(all_2items_table)
colnames(all_2items_df)<-c("Product","Cnt")


#Generate size 1 Freq item sets with min sup = 2
all_2items_dfnew<-all_2items_df[which(all_2items_df$Cnt>=2),]
nrow(all_2items_dfnew)

# Converting Table to Data Frame
freq_df<-data.frame(item_1_new)
colnames(freq_df)<-c("Product","Cnt")
ncol(freq_df)
nrow(freq_df)



















item_1_new<-as.matrix(item_1_new)
return (item_1_new)

items<-c()

for (i in 1:ncol(xxx)-1){
  items<-c(items,as.vector(xxx[i,]))
}
unique(items)




# Generate K itemsets 
zzz<-as.matrix(xxx[-ncol(xxx)])
result_df<-data.frame(A1=double(),A2=double(),A3=double())


for (rows in 1:nrow(zzz)){
  
  for(line in 1:nrow(item_1_new)){
    
    if((zzz[rows,1] != item_1_new[line,1]) & (zzz[rows,2] != item_1_new[line,1])){
      
      result_df= rbind(result_df,data.frame(A1=as.numeric(zzz[rows,1]),A2=as.numeric(zzz[rows,2]),A3=as.numeric(item_1_new[line,1])))
      
    }
  }
}


print(result_df)
for (i in 1:nrow(result_df)){
  result_df[i,]<-sort(result_df[i,])
}
print(unique(result_df))

abc<- unique(result_df)
abc$Cnt<-0
yyy<- as.matrix(abc)

for (i in 1:nrow(abc)){
  
  for (j in 1:nrow(new_dataset)){

    if (new_dataset[j,as.numeric(yyy[i,1])]==1 & new_dataset[j,as.numeric(yyy[i,2])]==1 & new_dataset[j,as.numeric(yyy[i,3])]==1){
      abc[i,ncol(abc)]=abc[i,ncol(abc)]+1
    }
  }
}





