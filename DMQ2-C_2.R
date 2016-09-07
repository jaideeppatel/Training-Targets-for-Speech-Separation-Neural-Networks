#getwd()
# Reading the input data set iris.data csv file

winedata <- read.csv("C:/Users/jaide/Documents/Data Mining/Homeworks/HW-2_Data/Wine/wine.data",header=FALSE)
print(winedata)
ncol(winedata)
newwinedata<-scale(winedata[2:14],center = TRUE,scale = TRUE)
boxplot(newwinedata,horizontal = TRUE, ylab="Feature Values",xlab="Features",main="Box Plot for Wine Features",names = c("Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid phenold", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"))
print(newwinedata)
classes=unique(newwinedata[,1])
print(classes[1])
total_class1=nrow(newwinedata[newwinedata[,1]==classes[1],])
total_class2=nrow(newwinedata[newwinedata[,1]==classes[2],])
total_class3=nrow(newwinedata[newwinedata[,1]==classes[3],])
colnames(newwinedata) <- c("Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid phenold", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline")
print(newwinedata)

data_matrix=as.matrix(newwinedata)
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
  if (newwinedata[min_row,1]==newwinedata[i,1]){
    same_class=same_class+1
  }
}
print(same_class)
class_percent=(same_class/nrow(newwinedata))*100
print(class_percent)
#Classwise
class1=0
class2=0
class3=0
for (i in 1:NROW(relation_matrix)){
  row_data=as.data.frame(relation_matrix[i,])
  min_row=which(row_data==min(row_data[row_data>0]))
  print(min_row)
  if (newwinedata[min_row,1]==classes[1] && newwinedata[i,1]==classes[1]){
    class1=class1+1
  }
  else if (newwinedata[min_row,1]==classes[2] && newwinedata[i,1]==classes[2]){
    class2=class2+1
  }
  else if (newwinedata[min_row,1]==classes[3] && newwinedata[i,1]==classes[3]){
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

