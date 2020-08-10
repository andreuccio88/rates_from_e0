########################################
##                                    ##
##       Andrea Nigri                 ##
##                                    ##
##         andrea.nigri@uniroma1.it   ##
##                                    ##
########################################

### --------------------------------------------------------------------------
### R Code: DNN mx from e0
### --------------------------------------------------------------------------

## For each code, please check technical info about packages for DNN implementation:

## R version 4.0.1 (2020-06-06)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## Running under: Windows 10 x64 (build 18363)
## Intel(R) Core(TM) i7 - 8550U CPU @ 1.80GHz 1.99GHz - RAM 16 GB

### --- Technical info about packages for DNN implementation
### 
### Keras version (R interface): 2.3.0.0
### Tensorflow version (R interface): 2.2.0
### Tensorflow version (Python module for backend tensor operations): 1.10.1 (Python module)
### Python version : 3.6

### Data source : Mortality Rates from Human Mortality Database
### Period : 1947-2014
### Country : ITA
### Gender : Males
### Forecasting period : 1950-2014

# Library, seed and error function -----------------------------------------------------------------

cnt <-"FRACNP" 
#un <- 700
#ep <- 350

un <- 80
ep <- 450




library(data.table)
library(reshape2)
library(ggplot2)
library(tidyverse)
library(splines)
library(LifeTables)
library(StMoMo)
#install.packages("devtools")
library(devtools)
#install_github("mpascariu/MortalityEstimate")
library(MortalityEstimate)


# Error functions 
rmse = function (truth, prediction)  {
  sqrt(mean((prediction - truth)^2))
}
mae = function(truth, prediction){
  mean(abs(prediction-truth))
}

options("scipen"=100, "digits"=4)
theme_set(theme_bw(base_family = "mono",base_size = 20))
set.seed(123)

##--------------------------------------------------------------------
##
##  Data Loading
##  
##--------------------------------------------------------------------

load("HMD_LT_f.Rdata")
load("HMD_LT_m.Rdata")

standardize <- . %>%  rename(year = Year, age = Age) %>% 
  mutate(lx = lx / 1e5,
         dx = dx / 1e5,
         Lx = Lx / 1e5,
         Tx = Tx / 1e5)

df_m <- HMD_LT_m$data %>% standardize()
df_f <- HMD_LT_f$data %>% standardize()
#table(df_m$country)
head(df_f)
head(df_m)

##--------------------------------------------------------------------
##
##  LEE CARTER
##  
##--------------------------------------------------------------------

D<- df_f %>%filter(year>=1950,year<=2014,age<=100,country==cnt)

D$Ex <- (D$dx*1e5)/D$mx
D$Dx <- D$Ex*D$mx

# We need to get Dx and Ex for StMoMo models

Dx <- D %>% select(year,age,Dx)
Ex <- D %>% select(year,age,Ex)

Dx <- spread(Dx,year,Dx)
Dx <- (Dx[c(1:101),])
Dx <- as.matrix(Dx[,-1])

Ex <- spread(Ex,year,Ex)
Ex <- (Ex[c(1:101),])
Ex <- as.matrix(Ex[,-1])

head(Dx)
head(Ex)

# StMoMo
age <- 0:100
year <- 1950:2005

wxt <- genWeightMat(ages = age, years = year,clip = 3)

LC.F<- fit(lc(link = "log"), Dxt = Dx[,1:56], Ext = Ex[,1:56], ages = , years = year,wxt = wxt)
forc <- forecast(LC.F, h = 9)

# We need to detach StMoMo to avoid conflicts with Keras
detach("package:StMoMo", unload = TRUE) 
library(keras) 
library(tensorflow) 

# Male population
D <- df_f %>% mutate(log_mx=log(mx))

D %>% filter(year==1950,country==cnt,age<=100) %>% select(log_mx,age) %>% ggplot(aes(age,log_mx))+geom_point()

##################################
##                              ##
## Country and Period selection ##
##                              ##
##################################

B <- D %>%filter(year>=1950,year<=2005,age<=100,country==cnt) %>%  select(year,age,log_mx,ex)

ex_ <- B %>% select(year,age,ex)
mx_ <- B %>% select(year,age,log_mx)

ex <- spread(ex_,age,ex) 
mx <- spread(mx_,age,log_mx) 

##--------------------------------------------------------------------
##
##  Linear Link Model. Pascariu et al.
##  
##--------------------------------------------------------------------
ages <- 0:100
years <- 1950:2005
rates <- exp(t(mx[,-1]))

# Fit Model
M <-LinearLink(ages,rates,years,theta = 0,method = "LSE")
M_pois <-LinearLink(ages,rates,years,theta = 0,method = "MLE")


##--------------------------------------------------------------------
##
##  Deep Neural Network
##  
##--------------------------------------------------------------------
set.seed(123)
# Row sampling for train-test split
smp=sample(1:nrow(ex),nrow(ex)/1.2)

#TRAIN
x_train <- as.matrix(ex[smp,c(1,2)]) #x_train
y_train <- as.matrix(mx[smp,]) #y_train
year_X_train <- sort(x_train[,1])
year_Y_train <- sort(y_train[,1])

##TEST
x_test <- as.matrix(ex[-smp,c(1,2)]) #x_test
y_test <- as.matrix(mx[-smp,]) #y_test

year_X_test <- sort(x_test[,1])
year_Y_test <- sort(y_test[,1])

# CHECK YEARS: TRAIN-TRAIN and TEST-TEST 
year_X_train + year_X_test==year_Y_train + year_Y_test

year_X_train== year_X_test
cbind(year_X_train ,year_X_test) # must be different

cbind(year_Y_train , year_Y_test) # must be different
year_Y_train == year_Y_test 

# DELETE FIRST COLUMN: YEAR
x_train <- x_train[,-c(1)] 
y_train <- y_train[,-c(1)]

x_test <- x_test[,-c(1)]
y_test <- y_test[,-c(1)]

# Normalize training data
x_train <- scale(x_train) 

# Use means and standard deviations from training set to normalize test set
col_means_train <- attr(x_train, "scaled:center") 
col_stddevs_train <- attr(x_train, "scaled:scale")
x_test <- scale(x_test, center = col_means_train, scale = col_stddevs_train)

##################################
##                              ##
##  DNN MODEL                   ##
##                              ##
##################################

use_session_with_seed(42)
build_model <- function() {
  
  model <- keras_model_sequential() %>%
    layer_dense(units =un, activation = 'relu',
                input_shape = dim(x_train)[2]) %>%
    layer_dense(units = un, activation = "relu") %>%
    layer_dense(units = un, activation = "relu") %>%
    layer_dense(units = un, activation = "relu") %>%
    layer_dense(units = un, activation = "relu") %>%
    layer_dense(units = un, activation = "relu") %>%
    layer_dense(units = un, activation = "relu") %>%
    layer_dense(units = 101)
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(), #Nel paper spiegare che ? un metodo proposto da geoffry hinton ecc..
    metrics = list("mean_absolute_error"))
  
  model}

model <- build_model()
model %>% summary()

# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  })    

epochs <- ep
# Fit the model and store training stats
history <- model %>% fit(
  x_train,
  y_train,
  epochs = epochs,
  batch_size = 1,
  validation_data = list(x_test, y_test),
  verbose = 2,
  callbacks = list(print_dot_callback))

plot(history, metrics = "mean_absolute_error", smooth = T) +
  coord_cartesian(ylim = c(0, 5))

history
test_predictions <- model %>% predict(x_test)
test_predictions


# Plot

i <- 2
NN <- (test_predictions[i,c(1:101)])
true<- (y_test[i,c(1:101)])


p <- data.frame(age=0:100,NN=NN,true=true)
p %>% ggplot(aes(age,true))+geom_line(size=1.7,alpha=0.8)+
  # geom_line(aes(age,NN),color="red",size=1.7,alpha=1)+
  stat_smooth(aes(age,NN),method = "lm",formula = y ~ bs(x,knots = c(1,10,15,25,35,70,75,90)),size = 1.5,alpha=0.8, se = F,color="red")+
  labs( title = "Italy Female, year:1965 (Smoothed)",
        subtitle = "Red: Neural Net, Black: Observed",
        y=expression("m"[x]))


#########################################
##                                     ##
## Validation - Backtesting: 2006-2014 ##
##                                     ##
#########################################

valid <- D %>% filter(year>2005&year<=2014,age<=100,country==cnt) %>%  
  select(year,age,log_mx,ex)

ex_valid <- valid %>% select(year,age,ex) 
mx_valid <- valid %>% select(year,age,log_mx)
ex_val <- spread(ex_valid,age,ex) 

# mx validation
mx_val <- as.matrix(spread(mx_valid,age,log_mx))

# e0 validation
x_val <- as.matrix(ex_val[,c(2)])
# e0 validation scaled
x_val_sc <- scale(x_val, center = col_means_train, scale = col_stddevs_train)

# NN prediction
val_predictions <- model %>% predict(x_val_sc)
val_predictions

# Lee Carter forecast
matr.lc <- t(log(forc$rates))

# Linear Link Prediction
LL_pred <- matrix(NA,length(x_val),101)
for (i in 1:length(x_val)) {
  p <- LinearLinkLT(M, ex = x_val[i])
  LL_pred[i,] <- log(p$lt$mx)
}


LL_POIS_pred <- matrix(NA,length(x_val),101)
for (i in 1:length(x_val)) {
  p <- LinearLinkLT(M_pois, ex = x_val[i])
  LL_POIS_pred[i,] <- log(p$lt$mx)
}


i <- 5
NN <- (val_predictions[i,c(1:101)])
true<- (mx_val[i,c(2:102)])
LC <- t(matr.lc)[i,c(1:101)]
LL <- LL_pred[i,c(1:101)]
LL_pois <- LL_POIS_pred[i,c(1:101)]


p <- data.frame(age=0:100,NN=NN,true=true,LC=LC,LL=LL)

p %>% ggplot(aes(age,true))+geom_point(size=1.1,alpha=0.8)+
  geom_line(aes(age,NN),color="red",size=1.1,alpha=1)+
  geom_line(aes(age,LC),color="black",size=1.1,alpha=1)+
  geom_line(aes(age,LL),color="blue",size=1.1,alpha=1)+
  geom_line(aes(age,LL_pois),color="yellow",size=1.1,alpha=1)+
  labs( title = "Italy Female, year:2007",
        subtitle = "Red: Neural Net, Black: Lee-Carter",
        y=expression("m"[x]))


##--------------------------------------------------------------------
##
##  SPLINE on VALIDATION
##  
##--------------------------------------------------------------------

library(demography)
ages <- unique(valid$age)
year <- unique(valid$year)
mx <- exp(val_predictions) # NN prediction
mx <- t(mx)
length(ages)

a <- demogdata(mx,pop = mx,ages = ages,years = year,type ="mortality",label = "prova",name="prova")
a$year
smooth <- smooth.demogdata(extract.years(a,2006:2014),method = "spline",weight = F,k = 80)
plot(smooth)

# Observed real values 2006-2014
matr.val <- t(mx_val[,-1])

# Lee Carter forecast 2006-2014
matr.lc <- log(forc$rates)

# NN+Smoothing forecast 2006-2014
matr.NN <- log(smooth$rate$prova[c(1:101),])

# NN (NO Smoothing) forecast 2006-2014
matr.val_predi <- t(val_predictions)

ll_pred <- t(LL_pred)
ll_POIS_pred <- t(LL_POIS_pred)

#View(LL_pred)
mae(matr.lc,matr.val)
mae(matr.NN,matr.val)
mae(matr.val_predi,matr.val)
mae(t(LL_pred),matr.val)

##--------------------------------------------------------------------
##
##  HEATMAP DELTA
##  
##--------------------------------------------------------------------

diff_lc <-( matr.lc- matr.val)/matr.val
#colnames(diff_lc) <- c(2006:2014)

diff_nn <-(matr.NN-matr.val)/matr.val
#colnames(diff_nn) <- c(2006:2014)

# PLOT LEE CARTER
delta_lc <- ggplot(melt((t(diff_lc)) ), aes(Var1,Var2, fill=value)) + 
  geom_raster()+scale_fill_viridis_c()+
  labs(title = "Italy Male",
       subtitle = "Observed Vs. Lee-Carter",
       y=expression("age"),x=expression("year"))+
  labs(fill="Delta")
delta_lc

# PLOT NN
delta_NN <-ggplot(melt((t(diff_nn)) ), aes(Var1,Var2, fill=value)) + 
  geom_raster()+scale_fill_viridis_c()+
  labs(title = "Italy Male",
       subtitle = "Observed Vs. DNN",
       y=expression("age"),x=expression("year"))+
  labs(fill="Delta")
delta_NN

# ERRORS
tab <- matrix(nrow=5, ncol=2)
dimnames(tab) <- list(c("NN","Smooth NN","LC","LL","LL_POIS"),c("MAE","RMSE"))
tab[,1] <-   c(mae(matr.val_predi,matr.val),  mae(matr.NN,matr.val),  mae(matr.lc,matr.val),mae((ll_pred),matr.val),mae((ll_POIS_pred),matr.val))
tab[,2] <-   c(rmse(matr.val_predi,matr.val),  rmse(matr.NN,matr.val),  rmse(matr.lc,matr.val), rmse((ll_pred),matr.val),rmse((ll_POIS_pred),matr.val))
tab

