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




c(c(x_train , y_train), c(x_test, y_test)) %<-% dataset_mnist()


x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test , c(nrow(x_test) , 784))
x_train <- x_train / 255
x_test <- x_test / 255


y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test ,10)




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



history <-model %>% fit(
  x_train, 
  y_train, 
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




# Print a summary of a model
summary(model)

# Get model configuration
get_config(model)

# Get layer configuration
get_layer(model, index = 1)

# List the model's layers
model$layers

# List the input tensors
model$inputs

# List the output tensors
model$outputs




