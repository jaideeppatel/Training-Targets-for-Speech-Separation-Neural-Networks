
# Prompt the user for the min support and the K value:

n <- readline(prompt="Enter an minimum support: ")
n<-as.numeric(n)
if (is.na(n)){
  prompt("Invalid input from the user for the minimum support")
}

cc <- readline(prompt="Enter an minimum confidence: ")
cc<-as.numeric(cc)
if (is.na(cc)){
  prompt("Invalid input from the user for the minimum support")
}

min_conf=cc

k <- readline(prompt="Enter the K value (size of frequent itemsets): ")
k <- as.numeric(k)
if (is.na(k)){
  prompt("Invalid input from the user for K value")
}


k<-ceiling(k)
itemset_size=k-1

# Code to read input file
input_dataset<-read.csv(file.choose(),header = FALSE)

# Count the number of items every user buys for general information to set k value
matrix_counts<-apply(input_dataset, 1, function(c)sum(c!=0))

# Remove all zero rows 
new_array <- apply(input_dataset, 1, function(x){sum(x)!=0})
new_dataset<-input_dataset[new_array==TRUE,]



# Implementing F k and F K-1 Apriori Algorithm

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
sup=2
item_1_new<-item_1[which(item_1$Cnt>=sup),]
nrow(item_1_new)

# Converting Table to Data Frame
freq_df<-data.frame(item_1_new)
colnames(freq_df)<-c("Product","Cnt")
print(freq_df)
#ncol(freq_df)
#nrow(freq_df) # freq_df is the set of all 1-freq itemsets with their support counts

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

# freq_df is the set of all K-1 freq itemsets with their support counts
my_ds<-k_one(new_dataset)
#print(my_ds)
my_ds<-my_ds[which(my_ds$Cnt>=sup),]
#print(my_ds)


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

#print(all_result)
# Output only Frequent itemsets
all_result<-all_result[which(all_result$Cnt>=sup),]
print('======== All Frequent Item sets ==========')
print(all_result)


# Implement rule generation 



# implement rule generation for all the Freq itemsets
freq_counts<-all_result
freqsets<-all_result[-ncol(all_result)]
iter=ncol(freqsets)-1
alln=ncol(freqsets)

rules1<-data.frame()
rules_set<-data.frame()


# Loop from level 1 cons to level n-1 consequents 
for (rows in 1:nrow(freqsets)){
  
  # Get the string items of the frequent set
  main<-toString(freqsets[rows,])
  main<-noquote(main)
  main<-gsub(',','',main)
  main<-as.character(as.vector(freqsets[rows,]))
  
  #Generationg rules from the frequent item set
  for(cols in iter:1){
    black_list<-c()
    rules1<-data.frame(combn(as.vector(freqsets[rows,]),cols))
    rules1<-t(rules1)
    
    # Generating antecendent and consequent for the rule
    for(ruler in 1:nrow(rules1)){
      antecedent<-gsub(',','',as.vector(rules1[ruler,]))
      consequent<-toString(setdiff(main,antecedent))
      antecedent<-toString(antecedent)
      rules_set<-rbind(rules_set,data.frame(Ante=antecedent,Cons=consequent,Conf=NA))
      # check if the rule needs to be pruned or not
      # Fetch rule support
      conseq_chk<-setdiff(main,antecedent)
      
      
      conseq_sup=freq_counts[rows,ncol(freq_counts)]
      # Fetch antecedent support
      ante_df<-data.frame()
      ante_df<-rbind(ante_df,data.frame(rules1[ruler,]))
      
      ante_df$Cnt<-0
      ante_all_result<-data.frame()
      ante_all_result<-ante_df
      flagc=0
      mysum=0
      temp<-as.matrix(ante_df)
      
      for (j in 1:nrow(new_dataset)){
        
        flag3=0
        
        for (x in 1:cols){
          
          if(new_dataset[j,as.numeric(temp[1,x])]!=1){
            flag3=1
          }
        }
        if(flag3==0){
          mysum=mysum+1
        }
      }
      ante_all_result[1,ncol(ante_all_result)]=ante_all_result[1,ncol(ante_all_result)]+mysum
      ante_sup=ante_all_result[1,ncol(ante_all_result)]
      rule_confidence=conseq_sup/ante_sup
      all_conf<-c(all_conf,rule_confidence)
      rules_set[nrow(rules_set),ncol(rules_set)]=rule_confidence
    }
  }
}




# Select top 5 rules based on the suport of the rules


# Converting string to vector values and find the antecedent and the cosequent
# Finding the ante and the cons of the rules

rules_set$Sup<-0

for(irow in 1:nrow(rules_set)){
  val1_ante<-toString(rules_set[irow,1])
  val1_ante<-as.numeric(unlist((strsplit(val1_ante,"[,]"))))
  val1_cons<-toString(rules_set[irow,2])
  val1_cons<-as.numeric(unlist((strsplit(val1_cons,"[,]"))))
  
  final_val<-c()
  final_val<-c(val1_ante,val1_cons)
  final_val<-as.numeric(final_val)
  print(final_val)
  
  # Find support of Association Rule
  flag3=0
  mysum=0
  cols=length(final_val)
  for(rr in 1:nrow(new_dataset)){
    flag3=0
    for (iter in 1:cols){
      if(new_dataset[rr,final_val[iter]]!=1){
        flag3=1
      }
    }
    if(flag3==0){
      mysum=mysum+1
    }
  }
  rules_set[irow,ncol(rules_set)]=rules_set[irow,ncol(rules_set)]+mysum
}



print(rules_set)
rules_set$lift<-0
rules_set$lift<-rules_set$Conf/rules_set$Sup

# All association rules with conf, sup, lift values are:

rules_set

