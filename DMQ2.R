#getwd()
# Reading the input data set iris.data csv file

winedata <- read.csv("C:/Users/jaide/Documents/Data Mining/Homeworks/HW-2_Data/Wine/wine.data",header=FALSE)
#print(winedata)
ncol(winedata)
colnames(winedata) <- c("Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid phenold", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline")
print(winedata)
cor_matrix = matrix(0,13,13)
for (i in seq(2,14)){
  for (j in seq(2,14)){
    if(i==j){}
    else{
    x<-winedata[,i]
    y<- winedata[,j]
    cor_value=cor(x,y,method = "pearson")
    cor_matrix[i-1,j-1]=cor_value 
    print(cor_value)
    print(i)
    print(j)
    }
  }
}
cor_matrix<-abs(cor_matrix)
cor_matrix[lower.tri(cor_matrix)]<-0
print(cor_matrix)
winedf<-data.frame(cor_matrix)
print(winedf)

mosthighlycorrelated <- function(cor_matrix,numtoreport)
{
  # find the correlations
  #cormatrix <- cor(mydataframe)
  # set the correlations on the diagonal or lower triangle to zero,
  # so they will not be reported as the highest ones:
  diag(cor_matrix) <- 0
  cor_matrix[lower.tri(cor_matrix)] <- 0
  # flatten the matrix into a dataframe for easy sorting
  fm <- as.data.frame(as.table(cor_matrix))
  # assign human-friendly names
  names(fm) <- c("First.Variable", "Second.Variable","Correlation")
  # sort and print the top n correlations
  head(fm[order(abs(fm$Correlation),decreasing=TRUE),],n=numtoreport)
}
resulttop4<-mosthighlycorrelated(cor_matrix, 4)
relations=data.matrix(resulttop4[,c(1,2)])
print(relations)

for (x in 1:nrow(relations)){
  print(x)
  col1=winedata[,relations[x,1]+1]
  col2=winedata[,relations[x,2]+1]
  print(col1)
  print(col2)
  xlabel=colnames(winedata)[relations[x,1]+1]
  ylabel=colnames(winedata)[relations[x,2]+1]
  #xlabel=colnames(winedata[,relations[x,1]+1])
  #ylabel=colnames(winedata[,relations[x,2]+1])
  print("xlabel name is:")
  print(xlabel)
  print("Ylabel name is:")
  print(ylabel)
  plot(col1,col2,xlab=xlabel,ylab=ylabel,col=c("red","blue"),main=paste(xlabel,"Vs",ylabel))
  abline(lm(col1~col2), col="red")
  lines(lowess(col1~col2), col="blue")
}

mostleastcorrelated <- function(cor_matrix,numtoreport)
{
  # find the correlations
  #cormatrix <- cor(mydataframe)
  # set the correlations on the diagonal or lower triangle to zero,
  # so they will not be reported as the highest ones:
  diag(cor_matrix) <- 0
  cor_matrix[lower.tri(cor_matrix)] <- 0
  # flatten the matrix into a dataframe for easy sorting
  fm <- as.data.frame(as.table(cor_matrix))
  # assign human-friendly names
  names(fm) <- c("First.Variable", "Second.Variable","Correlation")
  fm<-subset(fm,Correlation>0)
  # sort and print the top n correlations
  head(fm[order(abs(fm$Correlation)),],n=numtoreport)
}

resultleast4<-mostleastcorrelated(cor_matrix, 4)
relations=data.matrix(resultleast4[,c(1,2)])
print(relations)

for (x in 1:nrow(relations)){
  print(x)
  col1=winedata[,relations[x,1]+1]
  col2=winedata[,relations[x,2]+1]
  print(col1)
  print(col2)
  xlabel=colnames(winedata)[relations[x,1]+1]
  ylabel=colnames(winedata)[relations[x,2]+1]
  #xlabel=colnames(winedata[,relations[x,1]+1])
  #ylabel=colnames(winedata[,relations[x,2]+1])
  print("xlabel name is:")
  print(xlabel)
  print("Ylabel name is:")
  print(ylabel)
  plot(col1,col2,xlab=xlabel,ylab=ylabel,col=c("red","blue"),main=paste(xlabel,"Vs",ylabel))
  abline(lm(col1~col2), col="red")
  lines(lowess(col1~col2), col="blue")
}











