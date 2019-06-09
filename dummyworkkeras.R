library(tensorflow)
library(tfestimators)
library(tfdatasets)
library(tfdeploy)
library(tfruns)
library(cloudml)
library(dplyr)
#library(tidyverse)
library(keras)
library(corrplot)





#load the iris data
data(iris)

# this data has 150 rows
nrow(iris)

# look at the first few
head(iris)

# splitdf function will return a list of training and testing sets
splitdf <- function(dataframe, seed=NULL) {
  if (!is.null(seed)) set.seed(seed)
  index <- 1:nrow(dataframe)
  trainindex <- sample(index, trunc(length(index)/2))
  trainset <- dataframe[trainindex, ]
  testset <- dataframe[-trainindex, ]
  list(trainset=trainset,testset=testset)
}

#apply the function
splits <- splitdf(iris, seed=808)

#it returns a list - two data frames called trainset and testset
str(splits)

# there are 75 observations in each data frame
lapply(splits,nrow)

#view the first few columns in each data frame
lapply(splits,head)

# save the training and testing sets as data frames
training <- splits$trainset
testing <- splits$test


mat.training <-as.matrix(training)

test.target <- testing[,5]
train.target <- training[,5]


iris.trainLabels <- to_categorical(train.target)

# One hot encode test target values
iris.testLabels <- to_categorical(test.target)







