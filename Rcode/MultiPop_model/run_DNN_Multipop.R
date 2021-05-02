
#######################################################################################
##                                                                                   ## 
## Leveraging deep neural networks to estimate age specific                          ## 
##    mortality from life expectancy at birth (Nigri, Levantesi, Aburto)             ## 
##                                                                                   ## 
##                                                                                   ## 
##                                                                                   ## 
## This script will be useful to train the MultiPop Network.                         ##
#                                                                                    ##
##                                                                                   ## 
#######################################################################################

# Load Data

library(data.table)
library(reshape2)
library(ggplot2)
library(tidyverse)
library(keras)
library(reticulate)

source("fun_DNN_Multipop.R")

# Load log format HMD data
load("HMD_Long.RData")
head(DD)

#Filter years according the study period
DD <- DD %>%  filter(Year>1969) # for the last period 1970-2016; test= 2001-2015

train <-  DD[Year < 2001]
test <- DD[Year >= 2001]

# Model
test <- DNN_Multipop(train,test)

test %>% filter(Country%in%c("USA","ITA","RUS","JPN","")) %>% 
  group_by(Sex,Country) %>%
  summarise(MAE = mae(logmx,log(mx_DNN)),
            RMSE = rmse(logmx,log(mx_DNN)))

