#######################################################################################
##                                                                                   ## 
## Leveraging deep neural networks to estimate age specific                          ## 
##    mortality from life expectancy at birth (Nigri, Levantesi, Aburto)             ## 
##                                                                                   ## 
##                                                                                   ## 
##                                                                                   ## 
## This script will be useful to perform auto-tuning procedure i.e.                  ## 
## to obtain the best hyperparameters (number of units and epochs).                  ## 
##                                                                                   ## 
## The greater the Hyperparameter space,                                             ## 
## the higher the chance to find suitable combinations to reduce the error.          ## 
## In this script, we provide an example based on the ITA male population            ## 
## in the period 1970-2000 and out of sample 2001-2015.                              ## 
## From the paper we found the following best combination:                           ## 
## ep = 800; un =   200;  using earlystop regularization.                            ## 
##                                                                                   ## 
## Let's try to explore a wider space, let's say ep = 800                            ##
## un =   c(50,100,150,200,250)                                                      ## 
##                                                                                   ## 
#######################################################################################




source("FUN_AutoTuning.R")

library(keras) 
library(tensorflow) 
library(demography)
library(data.table)
library(reshape2)
library(ggplot2)
library(tidyverse)
library(splines)

set.seed(123)

##--------------------------------------------------------------------
##
##  Data Loading
##  
##--------------------------------------------------------------------

load("df_f.Rdata")
load("df_m.Rdata")

head(df_f)
head(df_m)

# It will take approximately 2 minutes for each combination
ep = 800
un =   c(50,100,150,200,250)




DNN <- Autotuning_DNN(cnt="ITA",age = 0:100,gender="M",
                      ys = 1970,yf = 2000,yv=2015,
                      un = un,
                      ep = ep,reg=1)


DNN
