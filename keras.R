#install.packages("keras")
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
library(caret)

data("iris")
dim(iris)








names(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")

plot(iris$Petal.Length, 
     iris$Petal.Width, 
     pch=21, bg=c("red","green3","blue")[unclass(iris$Species)], 
     xlab="Petal Length", 
     ylab="Petal Width")



cor(iris$Petal.Length, iris$Petal.Width)


# Store the overall correlation in `M`
M <- cor(iris[,1:4])

# Plot the correlation plot with `M`
corrplot(M, method="circle")



## Random Sampling 
##%70 to 30% split

data("iris")

N <- nrow(iris)
trainIndex <- base:: sample(x = N , size = trunc(0.7*N), replace = FALSE)
trainIndex


# Stratified sampling based 





# 
# Normalizing Your Data With A User Defined Function (UDF)
# From the result of the summary() function in the DataCamp Light chunk above, you see that the Iris data set doesn't need to be normalized: 
#the Sepal.Length attribute has values that go from 4.3 to 7.9 and Sepal.Width contains values from 2 to 4.4, while Petal.Length's values 
#range from 1 to 6.9 and Petal.Width goes from 0.1 to 2.5. In other words, all values of all the attributes of the Iris data set are contained 
#within the range of 0.1 and 7.9, 
#which you can consider acceptable.


# Build your own `normalize()` function
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

# Normalize the `iris` data
iris_norm <- as.data.frame(lapply(iris[1:4], normalize))

# Return the first part of `iris` 
head(iris)



# Tip use the hist() function in the R console to study the distribution of the Iris data before (iris) and after the normalization (iris_norm).



# To use the normalize() function from the keras package, you first need to make sure that you're working with a matrix. 
# As you probably remember from earlier, the characteristic of matrices is that the matrix data elements are of the same basic type; 
# In this case, you have target values that are of type factor, while the rest is all numeric.
#This needs to change first.


#You can use the as.numeric() function to convert the data to numbers:



iris[,5] <- as.numeric(iris[,5])

# Turn `iris` into a matrix
iris <- as.matrix(iris)

iris
# Set iris `dimnames` to `NULL`
dimnames(iris) <- NULL





# A numerical data frame is alright, but you'll need to convert the data to an array or a matrix if you want to make use of the keras package. You can easily do this with the as.matrix() function; Don't forget here to set the dimnames to NULL.
# 
# As you might have read in the section above, normalizing the Iris data is not necessary. Nevertheless, it's still a good idea to study normalization and its effect, and to see how this can not only be done with a UDF but also with the keras built-in normalize() function.
# 
# With your data converted to a matrix, you can indeed also use the keras package to study the effect of a possible normalization on your data:




#load iris data set



# Normalize the `iris` data
iris <- normalize(iris[,1:4])

# Return the summary of `iris`
summary(iris)


# Note that here, you use dimnames() to set the dimnames of iris to NULL. This ensures that there are no column names in your data.


# iris_original <- iris
# iris <- normalize(iris[,1:4])

# Determine sample size
ind <- sample(nrow(iris), size = nrow(iris)* 0.66666)

# Split the `iris` data
iris.training <- iris[ind, 1:4]
iris.test <- iris[ind, 1:4]




# Split the class attribute
iris.trainingtarget <- iris[ind, 5]
iris.testtarget <- iris[ind, 5]


# Calling as numerica
iris.testtarget <- iris.testtarget %>%as.numeric()


# x_train <- iris$train$x
# y_train <- iris$train$y
# x_test <- iris$test$x
# y_test <- iris$test$y
  


 
# Luckily, the keras package has a to_categorical() function that will do all of this for you; Pass in the iris.trainingtarget and the iris.testtarget to this function and store the result in iris.trainLabels and iris.testLabels:


# One hot encode training target values
iris.trainLabels <- to_categorical(iris.trainingtarget)

# One hot encode test target values
iris.testLabels <- to_categorical(iris.testtarget)

# Print out the iris.testLabels to double check the result
print(iris.testLabels)



# # Initialize a sequential model
# model <- keras_model_sequential() 
# 
# # Add layers to the model
# model %>% 
#   layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
#   layer_dense(units = 3, activation = 'softmax')





model<- keras_model_sequential()%>%
  layer_dense(units = 128, activation = 'relu',input_shape = c(784))%>%
  layer_dropout(rate = 0.4)%>%
  layer_dense(units = 128 ,activation = 'relu')%>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
  
)

# 
# model %>% compile(
#   loss = 'categorical_crossentropy',
#   optimizer = 'adam',
#   metrics = 'accuracy'
# )




history <-model %>% fit(
  iris.training, 
  iris.trainLabels, 
  epochs = 20, 
  batch_size = 5, 
  validation_split = 0.2
)


# x_train = iris.training
# y_train = iris.trainLabels

# 
# history <- model %>%fit(
#   x_train,y_train,
#   batch_size = 128,
#   epochs = 10,
#   validation_split =0.2
#   
#   
# )


summary(model)

plot(history)

model %>% evaluate(x_test,y_test)

model %>% predict_classes(x_test[1:100,])




# # Print a summary of a model
# summary(model)
# 
# # Get model configuration
# get_config(model)
# 
# # Get layer configuration
# get_layer(model, index = 1)
# 
# # List the model's layers
# model$layers
# 
# # List the input tensors
# model$inputs
# 
# # List the output tensors
# model$outputs

