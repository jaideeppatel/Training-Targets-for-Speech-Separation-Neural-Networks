# Generating Sparse matrix ...

# Code to read airfoil input file
input_dataset<-read.csv(file.choose(),header = FALSE)

maxim<-c()
mini<-c()

for(cols in 1:ncol(input_dataset)){
  maxim<-c(maxim,max(input_dataset[,cols]))
  mini<-c(mini,min(input_dataset[,cols]))
}

tempds<-as.matrix(input_dataset)

for(row in 1:nrow(input_dataset)){
  for(col in 1:ncol(input_dataset)){
    val=as.numeric((tempds[row,col]-mini[col])/(maxim[col]-mini[col]))
    if(val<=0.3){
      input_dataset[row,col]=0
    }else{
      input_dataset[row,col]=1
    }
  }
}

write.table(input_dataset,file="Census_Sparse_Test.csv",sep=",")
