# getwd()
# Reading the input data set iris.data csv file

Iris <- read.csv("C:/Users/jaide/Documents/Data Mining/Homeworks/HW-2_Data/Iris/iris.data",header=FALSE)
print(Iris)
# Generating mean and standard deviation for 4 features of the Flowers
# Sepal Lenth - Mean
meanV1=mean(Iris[["V1"]]) # To get the mean for V1 i.e. Sepal Length for all classes
SD1=sd(Iris[["V1"]]) # To get the standard deviation for V1 i.e. Sepal Length for all classes
print(meanV1)
print(SD1)
# Sepal Width - Mean
meanV2=mean(Iris[["V2"]]) # To get the mean for V2 i.e. Sepal Width for all classes
SD2=sd(Iris[["V2"]]) # To get the standard deviation for V2 i.e. Sepal Length for all classes
print(meanV2)
print(SD2)
# Petal Lenth - Mean
meanV3=mean(Iris[["V3"]]) # To get the mean for V3 i.e. Petal Length for all classes
SD3=sd(Iris[["V3"]]) # To get the standard deviation for V3 i.e. Petal Length for all classes
print(meanV3)
print(SD3)
# Petal Width - Mean
meanV4=mean(Iris[["V4"]]) # To get the mean for V4 i.e. Petal width for all classes
SD4=sd(Iris[["V4"]]) # To get the standard deviation for V4 i.e. Petal Length for all classes
print(meanV4)
print(SD4)

# Calculate V1 mean and SD by goruping for each flower
V1_allmeans=by(Iris$V1, Iris$V5, mean) # To get the mean for V1 i.e. Sepal Length for classwise
print(V1_allmeans)
V1_allsd=by(Iris$V1, Iris$V5, sd) # To get the SD for V1 i.e. Sepal Length for classwise
print(V1_allsd)
# Calculate V2 mean and SD by goruping for each flower
V2_allmeans=by(Iris$V2, Iris$V5, mean) # To get the mean for V2 i.e. Sepal width for classwise
print(V2_allmeans)
V2_allsd=by(Iris$V2, Iris$V5, sd) # To get the SD for V2 i.e. Sepal width for classwise
print(V2_allsd)
# Calculate V3 mean and SD by goruping for each flower
V3_allmeans=by(Iris$V3, Iris$V5, mean) # To get the mean for V3 i.e. Petal Length for classwise
print(V3_allmeans)
V3_allsd=by(Iris$V3, Iris$V5, sd) # To get the Sd for V3 i.e. Petal Length for classwise
print(V3_allsd)
# Calculate V4 mean and SD by goruping for each flower
V4_allMeans=by(Iris$V4, Iris$V5, mean) # To get the meab for V4 i.e. Petal width for classwise
print(V4_allMeans)
V4_allsd=by(Iris$V4, Iris$V5, sd)# To get the SD for V4 i.e. Petal width for classwise
print(V4_allsd)

# Box Plot for sepal length for all the 3 flowers
v1_sentosa<-Iris[which(Iris$V5=='Iris-setosa'),1] # Get Sepal length for Iris-Sentosa
print(v1_sentosa)
v1_versicolor<-Iris[which(Iris$V5=='Iris-versicolor'),1] # Get Sepal length for Iris-versicolor
print(v1_versicolor)
v1_virginica<-Iris[which(Iris$V5=='Iris-virginica'),1] # Get Sepal length for Iris-virginica
print(v1_virginica)

# Draw Box plot
boxplot(v1_sentosa,v1_versicolor,v1_virginica,medcol=c("red","blue","green"),ylab="Sepal Length in cms",xlab="Flower Type",main="Sepal Length for all Flower types",names = c("Iris-setosa","Iris-versicolor","Iris-virginica"))

# Box Plot for sepal width for all the 3 flowers
v2_sentosa<-Iris[which(Iris$V5=='Iris-setosa'),2] # Get Sepal width for Iris-Sentosa
print(v2_sentosa)
v2_versicolor<-Iris[which(Iris$V5=='Iris-versicolor'),2] # Get Sepal width for Iris-versicolor
print(v2_versicolor)
v2_virginica<-Iris[which(Iris$V5=='Iris-virginica'),2] # Get Sepal width for Iris-virginica
print(v2_virginica)

#Draw box plot
boxplot(v2_sentosa,v2_versicolor,v2_virginica,medcol=c("red","blue","green"),ylab="Sepal Width in cms",xlab="Flower Type",main="Sepal Width for all Flower types",names = c("Iris-setosa","Iris-versicolor","Iris-virginica"))

# Box Plot for petal length for all the 3 flowers
v3_sentosa<-Iris[which(Iris$V5=='Iris-setosa'),3] # Get Petal length for Iris-Sentosa
print(v3_sentosa)
v3_versicolor<-Iris[which(Iris$V5=='Iris-versicolor'),3] # Get Petal length for Iris-versicolor
print(v3_versicolor)
v3_virginica<-Iris[which(Iris$V5=='Iris-virginica'),3] # Get Petal length for Iris-virginica
print(v3_virginica)

#Draw box plot
boxplot(v3_sentosa,v3_versicolor,v3_virginica,medcol=c("red","blue","green"),ylab="Petal Length in cms",xlab="Flower Type",main="Petal Length for all Flower types",names = c("Iris-setosa","Iris-versicolor","Iris-virginica"))

# Box Plot for petal width for all the 3 flowers
v4_sentosa<-Iris[which(Iris$V5=='Iris-setosa'),4] # Get Petal width for Iris-Sentosa
print(v4_sentosa)
v4_versicolor<-Iris[which(Iris$V5=='Iris-versicolor'),4] # Get Petal width for Iris-versicolor
print(v4_versicolor)
v4_virginica<-Iris[which(Iris$V5=='Iris-virginica'),4] # Get Petal width for Iris-virginica
print(v4_virginica)

#Draw box plot
boxplot(v4_sentosa,v4_versicolor,v4_virginica,medcol=c("red","blue","green"),ylab="Petal Width in cms",xlab="Flower Type",main="Petal Width for all Flower types",names = c("Iris-setosa","Iris-versicolor","Iris-virginica"))
