#To download keras (tensorflow included) using RSTudio 4.3.2 and virtual environment with tensorflow 
install.packages("keras")
install_keras()

#Get packages
library(reticulate)
use_python("C:/Users/PDog/AppData/Local/r-miniconda/envs/r-tensorflow/python.exe")
library(keras)
library(tensorflow)
library (magick)
library (fs)
#np <- import("numpy")
#plt <- import("matplotlib.pyplot")

#1 Create Data

#Construct Circle Stimulus save as SOURCEPLOT.jpeg
xline <- c(0,1.053)
yline <- c(0,0)
plot(x = 0, y = 0, xlim = c(-1, 1), ylim = c(-1, 1), asp = 1, lwd=3, col= "gray",
cex = 115., las = 1,xlab = "", ylab = "",axes= FALSE, lines(xline,yline,lwd=3))

#Set directory and read one image for loop manipulation
directory <- "C:/GenerateData"
generatedata <- image_read(file.path(directory,"SOURCEPLOT.jpeg"))

#For loop creating 1 image every 1 degree rotation 360 times in separate folders
for(degree in 0:359){
  setwd("C:/GenerateData")
  rotation <- image_rotate(generatedata,degree)
  rotation <- image_trim(rotation,fuzz=0)
  rotation <- image_resize(rotation,geometry="244x244")
  new_dir <- (paste0(degree))
  dir.create(new_dir)
  setwd(new_dir)
  filename <- paste0(degree,".jpeg")
  image_write(rotation,filename)
}


#2 Custom Model

Kustom_CNN36 <- keras_model_sequential() %>% 
  layer_conv_2d(filters= 32, kernel_size= c(3,3), padding= 'same', activation= "relu", input_shape= c(244,244,1)) %>%
  layer_conv_2d(filters= 64, kernel_size= c(3,3), padding= 'same', activation= "relu") %>%
  layer_max_pooling_2d(pool_size= c(2,2), padding= 'same') %>%
  layer_conv_2d(filters= 64, kernel_size= c(3,3), padding= 'same', activation= "relu") %>%
  layer_max_pooling_2d(pool_size= c(2,2), padding= 'same') %>%
  layer_flatten() %>%
  layer_dense(units=510,activation = "relu")%>%
  layer_dense(units=480,activation = "relu")%>%
  layer_dense(units=420,activation = "relu")%>%
  layer_dense(units=360,activation = "softmax")

compile(Kustom_CNN36,
        optimizer= optimizer_rmsprop(learning_rate= .0001),
        loss= "sparse_categorical_crossentropy",
        metrics= "sparse_categorical_accuracy")

#3 Create Batch Dataset (can use image to array loop) ================================================================================================

#Create Dataset for Keras
CNNDataset <- image_dataset_from_directory(
  directory="C:/TestingData",
  labels= "inferred",
  label_mode = "int",
  color_mode = "grayscale",
  batch_size= 1,
  image_size = c(244,244),
  shuffle= FALSE,
  seed= NULL,
  validation_split= NULL,
  interpolation= "bilinear",
  follow_links= FALSE,
  crop_to_aspect_ratio= TRUE
)
class_names <- CNNDataset$class_names
print(class_names)


#Visualize dataset (work in progress)
#plt$figure(figsize=c(10,10))
#  for (i in range(8)){
#  plt$subplot(4,8,i+1)
#  plt$imshow(images[[i]]$numpy()$astype("uint8"))
#  plt$axis("off")
#}  
#for(i in 0:8 (images)){
#  plt$subplot(4,8,i+1)
#}

#data_subset <- CNNDataset$unbatch %>%
#CNNDataset$take(1)

#4 Train Model
modeltrain36 <- fit(
  Kustom_CNN36,
  x= CNNDataset,
  epochs= 13,
  verbose = 2,
  callbacks = NULL,
  validation_split = 0,
  validation_data = NULL,
  shuffle = TRUE,
  class_weight = NULL,
  sample_weight = NULL,
  initial_epoch = 0,
  steps_per_epoch = 360,
  validation_steps = NULL,
  )

#save_model_tf(object = Kustom_CNN11,filepath = "C:/Saved_Models", overwrite = TRUE, include_optimizer = TRUE)

#5 Evaluation predictions
predictions= Kustom_CNN36 %>% predict(x= CNNDataset, batch_size = 1, verbose = 2, steps = 10 ) %>% k_argmax(axis= -1)
predicted_class_names <- class_names[predicted_class_indices + 1]
