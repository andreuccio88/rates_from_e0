#######################################################################################
##                                                                                   ## 
## Leveraging deep neural networks to estimate age specific                          ## 
##    mortality from life expectancy at birth (Nigri, Levantesi, Aburto)             ## 
##                                                                                   ## 
##                                                                                   ## 
##                                                                                   ## 
## This script will be useful to train the Network                                   ##
##   based on its hyperparameter(number of units and epochs).                        ## 
##                                                                                   ##
## cnt = Selected Country                                                            ##
## age = 0:100,                                                                      ##
## ys = First Year of training                                                       ##
## yf = Last Year of training,                                                       ##
## yv = Last Year of Test (Out of Sample)                                            ##
## ep = epochs,                                                                      ##
## un = Unit                                                                         ##
##                                                                                   ## 
#######################################################################################



source("fun_DNN.R")
source("fun_Error.R")


library(keras) 
library(tensorflow) 
library(demography)
library(data.table)
library(reshape2)
library(ggplot2)
library(tidyverse)
library(splines)

set.seed(123)



# Load Hyperparameter for training (neurons and epochs)
load(file = "H_Parameter.RData")
setting <- s %>% filter(country=="ITA",gender=="M",period==2)

# period 1 = 1950:1980
# period 2 = 1960:1990
# period 3 = 1970:2000


##--------------------------------------------------------------------
##
##  Life Table Loading
##  
##--------------------------------------------------------------------

load("df_f.Rdata")
load("df_m.Rdata")


DNN <- Fit_DNN(cnt=setting$country,age = 0:100,ys = setting$ys,yf = setting$yf,yv=setting$yv,
                 ep=setting$ep,un=setting$un)

matr.NN<- DNN[[1]]
matr.val_predi<- DNN[[2]]
matr.val<- DNN[[3]]

mae(matr.NN,matr.val)
mae(matr.val_predi,matr.val)
rmse(matr.NN,matr.val)
rmse(matr.val_predi,matr.val)

