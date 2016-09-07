#getwd()
# Reading the input data set iris.data csv file

winedata <- read.csv("C:/Users/jaide/Documents/Data Mining/Homeworks/HW-2_Data/Wine/wine.data",header=FALSE)
#print(winedata)
ncol(winedata)
total_class1=nrow(winedata[winedata[,1]==1,])
total_class2=nrow(winedata[winedata[,1]==2,])
total_class3=nrow(winedata[winedata[,1]==3,])
colnames(winedata) <- c("Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid phenold", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline")
print(winedata)
data_matrix=as.matrix(winedata)
data_matrix<-data_matrix[,-1]
print(data_matrix)
relation_matrix<-dist(data_matrix,method = "euclidean",diag = FALSE, upper = TRUE)
relation_matrix=as.matrix(relation_matrix)
print(NROW(relation_matrix))
same_class=0
for (i in 1:NROW(relation_matrix)){
  row_data=as.data.frame(relation_matrix[i,])
  min_row=which(row_data==min(row_data[row_data>0]))
  print(min_row)
  if (winedata[min_row,1]==winedata[i,1]){
    same_class=same_class+1
  }
}
print(same_class)
class_percent=(same_class/nrow(winedata))*100
print(class_percent)
#Classwise
class1=0
class2=0
class3=0
for (i in 1:NROW(relation_matrix)){
  row_data=as.data.frame(relation_matrix[i,])
  min_row=which(row_data==min(row_data[row_data>0]))
  print(min_row)
  if (winedata[min_row,1]==1 && winedata[i,1]==1){
    class1=class1+1
  }
  else if (winedata[min_row,1]==2 && winedata[i,1]==2){
    class2=class2+1
  }
  else if (winedata[min_row,1]==3 && winedata[i,1]==3){
    class3=class3+1
  }
}
print(class1)
print(class2)
print(class3)
print(total_class1)
print(total_class2)
print(total_class3)
class1_percent=(class1/total_class1)*100
class2_percent=(class2/total_class2)*100
class3_percent=(class3/total_class3)*100
print(class1_percent)
print(class2_percent)
print(class3_percent)
